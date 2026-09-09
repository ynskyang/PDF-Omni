import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.featurelayer import Conv2D
from module.backbone import PDFOmniEncoder
from module.volume_generator import Generator
from module.ge import GE
from module.update import UpdateBlock, DiGRU
from utils.common import *

autocast = partial(torch.amp.autocast, 'cuda')


class PDFOmni(torch.nn.Module):

    def __init__(self, varargin=None):
        super(PDFOmni, self).__init__()
        opts = Edict()
        opts.use_rgb = False
        self.opts = argparse(opts, varargin)

        self.use_ge = getattr(self.opts, "use_ge", False)
        self.use_digru = getattr(self.opts, "use_digru", False)
        self.distortion_mode = getattr(self.opts, "distortion_mode", "hyperbolic_dual_disk")

        self.encoder = PDFOmniEncoder(
            base_channel=self.opts.base_channel,
            dim_expansion=getattr(self.opts, "dim_expansion", 2),
            num_transformer=getattr(self.opts, "num_transformer", 3),
            downsample_twice=getattr(self.opts, "encoder_downsample_twice", False),
            pe_mode=getattr(self.opts, "pe_mode", "grid"),
        )
        context_dim = self.opts.base_channel
        hidden_dim  = self.opts.base_channel * 2

        self.volume_gen = Generator(self.opts)

        self.state_conv = Conv2D(context_dim, hidden_dim, 1, pad=0, relu=False)

        if self.use_ge:
            ge_scale = getattr(self.opts, "ge_scale", 10.0)
            ge_map_size = getattr(self.opts, "ge_map_size", 256)
            self.ge = GE(input_dim=3, mapping_size=ge_map_size, scale=ge_scale, out_dim=context_dim)
        else:
            self.ge = None

        if self.use_digru:
            self.update_block = DiGRU(self.opts, hidden_dim=hidden_dim, input_dim=context_dim)
        else:
            self.update_block = UpdateBlock(self.opts, hidden_dim=hidden_dim, input_dim=context_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def spherical_sweep(self, fisheye_feats, grids):
        bs = fisheye_feats[0].shape[0]
        grids_pad = [torch.cat([torch.zeros_like(grid[..., :1]), grid], dim=-1) for grid in grids]
        sph_feats = []
        for feat, grid in zip(fisheye_feats, grids_pad):
            sph_feat = F.grid_sample(feat[..., None], grid.repeat(bs, 1, 1, 1, 1), align_corners=True)
            sph_feats.append(sph_feat)

        sph_feats = sph_feats + [grid.permute(-1, 0, 1, 2).repeat(bs, 1, 1, 1, 1) for grid in grids]
        return sph_feats

    @staticmethod
    def _build_unit_sphere_grid(h, w, device, dtype):
        phi = torch.linspace(-math.pi / 2, math.pi / 2, h, device=device, dtype=dtype)
        lam = torch.linspace(-math.pi, math.pi, w, device=device, dtype=dtype)
        grid_phi, grid_lam = torch.meshgrid(phi, lam, indexing="ij")
        x = -torch.cos(grid_phi) * torch.cos(grid_lam)
        y = torch.sin(grid_phi)
        z = torch.cos(grid_phi) * torch.sin(grid_lam)
        return torch.stack([x, y, z], dim=0).unsqueeze(0)

    @staticmethod
    def _build_distortion_map(h, w, device, dtype, mode="hyperbolic_dual_disk"):
        if mode != "hyperbolic_dual_disk":
            raise ValueError(f"Unknown distortion mode: {mode}")

        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        grid_x_f = grid_x.to(compute_dtype)
        grid_y_f = grid_y.to(compute_dtype)
        eps_h = torch.as_tensor(1e-6, device=device, dtype=compute_dtype)

        def _acosh(x):
            if hasattr(torch, "acosh"):
                return torch.acosh(x)
            return torch.log(x + torch.sqrt(torch.clamp(x * x - 1.0, min=0.0)))

        def get_hyperbolic_distance_from_origin(u_norm, v_norm):
            x = 2.0 * u_norm - 1.0
            y = 2.0 * v_norm - 1.0

            r = torch.sqrt(x * x + y * y).clamp_min(eps_h)
            scale = (1.0 - eps_h) / torch.maximum(torch.ones_like(r), r)
            px = x * scale
            py = y * scale

            p2 = px * px + py * py
            denom = (1.0 - p2).clamp_min(eps_h)
            z = 1.0 + 2.0 * p2 / denom
            z = z.clamp_min(1.0 + eps_h)
            return _acosh(z)

        u = (grid_x_f + 1.0) / 2.0
        v = (grid_y_f + 1.0) / 2.0

        df = get_hyperbolic_distance_from_origin(u, v)
        db = get_hyperbolic_distance_from_origin((u + 0.5) % 1.0, v)

        balance = 1.0 - torch.abs(df - db) / (df + db + eps_h)
        min_norm = torch.tanh(torch.min(df, db))
        distortion = torch.tanh(min_norm * (1.0 + 0.5 * balance))

        return distortion.unsqueeze(0).unsqueeze(0)

    def upsample_invdepth_idx(self, invdepth, mask):
        bs, ch, h, w = invdepth.shape
        factor = 2 ** self.opts.num_downsample
        mask = mask.view(bs, 1, 9, factor, factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_invdepth = F.unfold(factor * invdepth, [3, 3], padding=1)
        up_invdepth = up_invdepth.view(bs, ch, 9, 1, 1, h, w)

        up_invdepth = torch.sum(mask * up_invdepth, dim=2)
        up_invdepth = up_invdepth.permute(0, 1, 4, 2, 5, 3)
        return up_invdepth.reshape(bs, ch, factor*h, factor*w)

    def volume_sample(self, feat_volume, invdepth_idx):
        bs, ch, h, w, n_invd = feat_volume.shape

        invdepth_idx_floor = torch.floor(invdepth_idx)
        invdepth_idx_ceil  = invdepth_idx_floor + 1
        invdepth_idx_floor = torch.clamp(invdepth_idx_floor, 0, n_invd - 1)
        invdepth_idx_ceil  = torch.clamp(invdepth_idx_ceil , 0, n_invd - 1)
        invdepth_idx       = torch.clamp(invdepth_idx      , 0, n_invd - 1)

        weight_floor = (invdepth_idx_ceil - invdepth_idx)
        weight_floor[weight_floor == n_invd - 1] = 1.0
        weight_ceil  = (invdepth_idx - invdepth_idx_floor)
        weight_ceil[invdepth_idx_ceil == 0] = 1.0

        invdepth_idx_floor = invdepth_idx_floor.long()
        invdepth_idx_ceil  = invdepth_idx_ceil.long()

        feat_floor = torch.gather(
            feat_volume, 4, invdepth_idx_floor.repeat(1, ch, 1, 1).unsqueeze(-1)
        )[..., 0]
        feat_ceil = torch.gather(
            feat_volume, 4, invdepth_idx_ceil.repeat(1, ch, 1, 1).unsqueeze(-1)
        )[..., 0]

        return weight_ceil * feat_ceil + weight_floor * feat_floor

    def forward(self, imgs, grids, iters=12, test_mode=False):
        with autocast(enabled=self.opts.mixed_precision):
            fisheye_feats = self.encoder(imgs)

        fisheye_feats   = [feat.float() for feat in fisheye_feats]
        spherical_feats = self.spherical_sweep(fisheye_feats, grids)

        with autocast(enabled=self.opts.mixed_precision):
            geo_sampler, init_disp, context_feat_volume, init_prob = self.volume_gen(spherical_feats)

        context_feat = context_feat_volume[..., 0]

        grid_feat = None
        if self.use_ge:
            grid_xyz = self._build_unit_sphere_grid(
                context_feat.shape[-2], context_feat.shape[-1],
                device=context_feat.device, dtype=context_feat.dtype
            ).expand(context_feat.shape[0], -1, -1, -1)
            with autocast(enabled=self.opts.mixed_precision):
                grid_feat = self.ge(grid_xyz)
                context_feat = context_feat + grid_feat

        with autocast(enabled=self.opts.mixed_precision):
            inp = torch.relu(context_feat)
            net = torch.tanh(self.state_conv(context_feat))  # hidden state

        Nd = context_feat_volume.shape[-1]
        invdepth_idx = torch.clamp(init_disp, 0, Nd - 1)

        invdepth_idx_predictions = []
        distortion_map = None
        if self.use_digru:
            distortion_map = self._build_distortion_map(
                context_feat.shape[-2], context_feat.shape[-1],
                device=context_feat.device, dtype=context_feat.dtype,
                mode=self.distortion_mode
            ).expand(context_feat.shape[0], -1, -1, -1)
        for itr in range(iters):
            invdepth_idx = invdepth_idx.detach()

            invdepth_idx_safe = torch.nan_to_num(
                invdepth_idx, nan=0.0, posinf=float(Nd - 1), neginf=0.0
            ).clamp(0.0, float(Nd - 1))

            geo_feat = geo_sampler(invdepth_idx_safe)

            if itr > 0:
                context_feat = self.volume_sample(context_feat_volume, invdepth_idx_safe)
                if self.use_ge and grid_feat is not None:
                    context_feat = context_feat + grid_feat
                inp = torch.relu(context_feat)

            with autocast(enabled=self.opts.mixed_precision):
                if self.use_digru:
                    net, delta_invdepth_idx, up_mask = self.update_block(
                        net, inp, geo_feat, invdepth_idx_safe, distortion_map=distortion_map,
                        no_upsample=(test_mode and itr < iters - 1)
                    )
                else:
                    net, delta_invdepth_idx, up_mask = self.update_block(
                        net, inp, geo_feat, invdepth_idx_safe,
                        no_upsample=(test_mode and itr < iters - 1)
                    )

            invdepth_idx = invdepth_idx + delta_invdepth_idx

            if up_mask is not None:
                invdepth_idx_up = self.upsample_invdepth_idx(invdepth_idx, up_mask)
                invdepth_idx_predictions.append(invdepth_idx_up)

        if test_mode:
            return torch.clamp(invdepth_idx_predictions[-1], 0, self.opts.num_invdepth - 1)

        if self.training:
            return invdepth_idx_predictions, init_prob
        return invdepth_idx_predictions
