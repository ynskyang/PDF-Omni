# The attention-volume construction and 3D hourglass aggregation are adapted
# from ACVNet (MIT): https://github.com/gangweiX/ACVNet
# The geometry encoding volume lookup follows IGEV-Stereo (MIT).

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, deconv=False, bn=True, relu=True):
        super().__init__()
        self.relu = relu
        self.use_bn = bn
        if deconv:
            self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch) if bn else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.leaky_relu(x, inplace=False)
        return x


class Hourglass3D(nn.Module):
    def __init__(self, in_ch=8, base=8, out_ch=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv3d(in_ch, base*2, 3, 2, 1, deconv=False, bn=True, relu=True),
            BasicConv3d(base*2, base*2, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.conv2 = nn.Sequential(
            BasicConv3d(base*2, base*4, 3, 2, 1, deconv=False, bn=True, relu=True),
            BasicConv3d(base*4, base*4, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.conv3 = nn.Sequential(
            BasicConv3d(base*4, base*6, 3, 2, 1, deconv=False, bn=True, relu=True),
            BasicConv3d(base*6, base*6, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.up3 = BasicConv3d(base*6, base*4, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), deconv=True, bn=True, relu=True)
        self.fuse2 = nn.Sequential(
            BasicConv3d(base*8, base*4, 1, 1, 0, deconv=False, bn=True, relu=True),
            BasicConv3d(base*4, base*4, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.up2 = BasicConv3d(base*4, base*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), deconv=True, bn=True, relu=True)
        self.fuse1 = nn.Sequential(
            BasicConv3d(base*4, base*2, 1, 1, 0, deconv=False, bn=True, relu=True),
            BasicConv3d(base*2, base*2, 3, 1, 1, deconv=False, bn=True, relu=True)
        )
        self.up1 = BasicConv3d(base*2, out_ch, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), deconv=True, bn=False, relu=False)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        u3 = self.up3(c3)
        f2 = torch.cat([u3, c2], dim=1)
        f2 = self.fuse2(f2)
        u2 = self.up2(f2)
        f1 = torch.cat([u2, c1], dim=1)
        f1 = self.fuse1(f1)
        out = self.up1(f1)
        return out


def groupwise_corr_5d(ref_vol, tgt_vol, num_groups=8):
    B, C, H, W, D = ref_vol.shape
    assert C % num_groups == 0, "channels must be divisible by num_groups"
    g = num_groups
    cpg = C // g

    ref_g = ref_vol.view(B, g, cpg, H, W, D)
    tgt_g = tgt_vol.view(B, g, cpg, H, W, D)
    cost = (ref_g * tgt_g).mean(dim=2)
    return cost.permute(0, 1, 4, 2, 3).contiguous()


def channel_corr_5d(ref_vol, tgt_vol):
    cost = (ref_vol * tgt_vol).sum(dim=1, keepdim=True)
    return cost.permute(0, 1, 4, 2, 3).contiguous()


class ACVAttention(nn.Module):
    def __init__(self, num_groups: int, base_ch: int = 32, att_heads: int = None, use_geo_acv: bool = False):
        super().__init__()
        self.use_geo_acv = use_geo_acv
        G = num_groups + 2 if use_geo_acv else num_groups
        s1 = max(1, G // 4)
        s2 = max(1, G // 2)
        s3 = max(1, G - s1 - s2)
        if s1 + s2 + s3 != G:
            s3 = G - s1 - s2

        self.slices = (s1, s2, s3)

        self.patch_all = nn.Conv3d(G, G, kernel_size=(1,3,3), padding=(0,1,1), groups=G, bias=False)
        self.patch_l1  = nn.Conv3d(s1, s1, kernel_size=(1,3,3), padding=(0,1,1), dilation=(1,1,1), groups=s1, bias=False)
        self.patch_l2  = nn.Conv3d(s2, s2, kernel_size=(1,3,3), padding=(0,2,2), dilation=(1,2,2), groups=s2, bias=False)
        self.patch_l3  = nn.Conv3d(s3, s3, kernel_size=(1,3,3), padding=(0,3,3), dilation=(1,3,3), groups=s3, bias=False)

        self.dres1_att = nn.Sequential(
            BasicConv3d(G, base_ch, 3, 1, 1),
            BasicConv3d(base_ch, base_ch, 3, 1, 1)
        )
        self.dres2_att = Hourglass3D(in_ch=base_ch, base=8, out_ch=base_ch)
        self.classif_att = nn.Sequential(
            BasicConv3d(base_ch, base_ch, 3, 1, 1),
            BasicConv3d(base_ch, 1, 3, 1, 1, bn=False, relu=False)
        )

    def forward(self, gwc):
        B, G, D, H, W = gwc.shape
        s1, s2, s3 = self.slices

        gwc_s = self.patch_all(gwc)

        xs = []
        p = 0
        if s1 > 0:
            xs.append(self.patch_l1(gwc_s[:, p:p+s1]))
            p += s1
        if s2 > 0:
            xs.append(self.patch_l2(gwc_s[:, p:p+s2]))
            p += s2
        if s3 > 0:
            xs.append(self.patch_l3(gwc_s[:, p:p+s3]))

        patch_vol = torch.cat(xs, dim=1)
        feat = self.dres1_att(patch_vol)
        feat = self.dres2_att(feat)
        att_logits = self.classif_att(feat)
        return att_logits


class CombinedGeoEncodingSampler:
    def __init__(self, geo_encoding_volume, init_corr_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        B, Cg, D, H, W = geo_encoding_volume.shape
        self.B, self.Cg, self.D, self.H, self.W = B, Cg, D, H, W

        self.geo_pyrs = []
        self.init_pyrs = []

        geo = geo_encoding_volume.permute(0, 3, 4, 1, 2).reshape(B*H*W, Cg, 1, D)
        init = init_corr_volume.permute(0, 3, 4, 1, 2).reshape(B*H*W, 1 , 1, D)

        self.geo_pyrs.append(geo)
        self.init_pyrs.append(init)

        for _ in range(1, num_levels):
            geo = F.avg_pool2d(geo, kernel_size=[1,2], stride=[1,2])
            init = F.avg_pool2d(init, kernel_size=[1,2], stride=[1,2])
            self.geo_pyrs.append(geo)
            self.init_pyrs.append(init)

    def __call__(self, disp_idx, coords=None):
        r = self.radius
        B, _, H, W = disp_idx.shape
        assert B == self.B and H == self.H and W == self.W

        flat = disp_idx.reshape(B*H*W, 1, 1, 1)
        windows = torch.linspace(-r, r, 2*r+1, device=disp_idx.device, dtype=disp_idx.dtype).view(1, 1, 2*r+1, 1)

        out_feats = []

        for lvl in range(self.num_levels):
            geo  = self.geo_pyrs[lvl]
            init = self.init_pyrs[lvl]

            x0 = (flat / (2**lvl)) + windows

            denom = max(1, geo.shape[-1] - 1)
            xn = 2.0 * x0 / float(denom) - 1.0
            yn = torch.zeros_like(xn)

            base_grid = torch.cat([xn, yn], dim=-1)

            grid_geo  = base_grid.to(dtype=geo.dtype)
            grid_init = base_grid.to(dtype=init.dtype)

            geo_samp  = F.grid_sample(geo,  grid_geo,  mode='bilinear', align_corners=True)
            init_samp = F.grid_sample(init, grid_init, mode='bilinear', align_corners=True)

            geo_samp  = geo_samp.view(B, H, W, self.Cg*(2*r+1)).contiguous()
            init_samp = init_samp.view(B, H, W,       (2*r+1)).contiguous()

            feat_lvl = torch.cat([geo_samp, init_samp], dim=-1)
            out_feats.append(feat_lvl)

        out = torch.cat(out_feats, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class MLP(nn.Module):
    def __init__(self, ch_in, ch_hid, ch_out=1):
        super().__init__()
        self.linear1 = nn.Conv3d(ch_in, ch_hid, (1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv3d(ch_hid, ch_out, (1, 1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.out_act(x)


class Compress3D(nn.Module):
    def __init__(self, ch_in, ch_mid=32, ch_out=16):
        super().__init__()
        self.conv1 = nn.Conv3d(ch_in, ch_mid, kernel_size=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm3d(ch_mid)
        self.conv2 = nn.Conv3d(ch_mid, ch_out, kernel_size=1, padding=0, bias=False)
        self.bn2   = nn.BatchNorm3d(ch_out)

    def forward(self, x5d):
        x = x5d.permute(0,1,4,2,3).contiguous()
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class Generator(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        ch_in = opts.base_channel
        self.use_geo_acv = getattr(opts, "use_geo_acv", False)
        self.num_groups = getattr(opts, "gwc_groups", getattr(opts, "num_groups", 8))
        self.acv_concat_ch = getattr(opts, "acv_concat_ch", 16)
        self.geo_ch = getattr(opts, "geo_ch", 8)
        self.initcorr_ch = 1

        self.reference_mapping = MLP(2*ch_in+4, ch_in)
        self.target_mapping    = MLP(2*ch_in+4, ch_in)

        self.att_branch = ACVAttention(num_groups=self.num_groups, base_ch=32, use_geo_acv=self.use_geo_acv)
        self.ref_compress = Compress3D(ch_in, ch_mid=max(16, self.acv_concat_ch*2), ch_out=self.acv_concat_ch)
        self.tgt_compress = Compress3D(ch_in, ch_mid=max(16, self.acv_concat_ch*2), ch_out=self.acv_concat_ch)

        acv_in_ch = 2 * self.acv_concat_ch
        self.cost_agg = Hourglass3D(in_ch=acv_in_ch, base=8, out_ch=self.geo_ch)
        self.classifier = nn.Conv3d(self.geo_ch, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.ot_iter = getattr(opts, "ot_iter", 3)

    @staticmethod
    def _concat_every_other(feats):
        return torch.cat(feats[0::2], dim=1), torch.cat(feats[1::2], dim=1)

    @staticmethod
    def _logsumexp_stable(x: torch.Tensor, dim: int, keepdim: bool = False, eps: float = 1e-30) -> torch.Tensor:
        m, _ = x.max(dim=dim, keepdim=True)
        y = (x - m).exp().sum(dim=dim, keepdim=True)
        y = m + torch.log(torch.clamp(y, min=eps))
        return y if keepdim else y.squeeze(dim)

    def _sinkhorn(self, logits: torch.Tensor, iters: int) -> torch.Tensor:
        B, D, H, W = logits.shape
        N = H * W

        log_q = logits.reshape(B, D, N)
        log_mu = torch.full((B, D, 1), -math.log(max(D, 1)), device=logits.device, dtype=logits.dtype)
        log_nu = torch.full((B, 1, N), -math.log(max(N, 1)), device=logits.device, dtype=logits.dtype)

        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(max(int(iters), 1)):
            u = log_mu - self._logsumexp_stable(log_q + v, dim=2, keepdim=True)
            v = log_nu - self._logsumexp_stable(log_q + u, dim=1, keepdim=True)

        return (log_q + u + v).reshape(B, D, H, W)

    def forward(self, spherical_feats):
        front_cat = torch.cat(spherical_feats[0::2], dim=1)
        right_cat = torch.cat(spherical_feats[1::2], dim=1)  # 1,3,5,...

        front_weight = self.reference_mapping(front_cat)
        reference_feat = front_weight * spherical_feats[0] + (1.0 - front_weight) * spherical_feats[2]

        right_weight  = self.target_mapping(right_cat)
        target_feat   = right_weight * spherical_feats[1] + (1.0 - right_weight) * spherical_feats[3]

        context_feat_volume = reference_feat

        gwc = groupwise_corr_5d(reference_feat, target_feat, num_groups=self.num_groups)
        if self.use_geo_acv:
            _, _, D, H, W = gwc.shape
            phi = torch.linspace(-math.pi / 2, math.pi / 2, H, device=gwc.device, dtype=gwc.dtype)
            lam = torch.linspace(-math.pi, math.pi, W, device=gwc.device, dtype=gwc.dtype)
            grid_phi, grid_lam = torch.meshgrid(phi, lam, indexing="ij")
            coord = torch.stack([grid_phi, grid_lam], dim=0).unsqueeze(0).unsqueeze(2)
            coord = coord.expand(gwc.shape[0], -1, D, -1, -1)
            gwc_att = torch.cat([gwc, coord], dim=1)
        else:
            gwc_att = gwc

        att_logits = self.att_branch(gwc_att)
        att_prob   = F.softmax(att_logits, dim=2)

        ref_c = self.ref_compress(reference_feat)
        tgt_c = self.tgt_compress(target_feat)
        concat_vol = torch.cat([ref_c, tgt_c], dim=1)

        acv = att_prob * concat_vol

        geo_encoding_volume = self.cost_agg(acv)

        logits = self.classifier(geo_encoding_volume).squeeze(1)
        ot_log_prob = self._sinkhorn(logits, self.ot_iter)
        init_prob = torch.exp(ot_log_prob)
        init_prob = init_prob / (init_prob.sum(dim=1, keepdim=True) + 1e-9)

        Nd = init_prob.shape[1]
        disp_values = torch.arange(0, Nd, device=init_prob.device, dtype=init_prob.dtype).view(1, Nd, 1, 1)
        init_disp = (init_prob * disp_values).sum(dim=1, keepdim=True)

        init_corr = channel_corr_5d(reference_feat, target_feat)

        geo_sampler = CombinedGeoEncodingSampler(
            geo_encoding_volume=geo_encoding_volume,
            init_corr_volume=init_corr,
            num_levels=getattr(self.opts, "corr_levels", 2),
            radius=getattr(self.opts, "corr_radius", 4),
        )

        return geo_sampler, init_disp, context_feat_volume, init_prob
