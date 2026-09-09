# The selective recurrent unit (small/large-kernel GRU blending) and the
# channel/spatial attention modules are adapted from Selective-Stereo (MIT):
#   https://github.com/Windsrain/Selective-Stereo
# The recurrent update structure follows RomniStereo / RAFT-Stereo (MIT).

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)


class DistortionAwareAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.samconv = nn.Conv2d(3, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, distortion_map):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        if distortion_map is None:
            distortion_map = torch.zeros_like(avg_out)
        elif distortion_map.shape[-2:] != x.shape[-2:]:
            distortion_map = F.interpolate(distortion_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        distortion_map = distortion_map.clamp(0.0, 1.0)

        combined = torch.cat([avg_out, max_out, distortion_map], dim=1)
        return self.sigmoid(self.samconv(combined))


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) == 1:
                kernel_size = int(kernel_size[0])
            else:
                kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        if isinstance(kernel_size, (list, tuple)):
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            pad = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        return (1 - z) * h + z * q


class RaftSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=5):
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, k), padding=(0, pad))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, k), padding=(0, pad))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, k), padding=(0, pad))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (k, 1), padding=(pad, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (k, 1), padding=(pad, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (k, 1), padding=(pad, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256,
                 small_kernel_size=1, large_kernel_size=5, large_mode="sep"):
        super().__init__()
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        if large_mode == "sep":
            self.large_gru = RaftSepConvGRU(hidden_dim, input_dim, large_kernel_size)
        else:
            self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, x):
        h_small = self.small_gru(h, x)
        h_large = self.large_gru(h, x)
        return h_small * (1 - att) + h_large * att


class MotionEncoder(nn.Module):
    def __init__(self, cor_planes, c1_planes=64, c2_planes=64, d1_planes=64, d2_planes=64, out_planes=128):
        super(MotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_planes, c1_planes, kernel_size=1, padding=0, bias=True)
        self.convc2 = nn.Conv2d(c1_planes, c2_planes, kernel_size=3, padding=1, bias=True)

        self.convd1 = nn.Conv2d(1, d1_planes, kernel_size=7, padding=3, bias=True)
        self.convd2 = nn.Conv2d(d1_planes, d2_planes, kernel_size=3, padding=1, bias=True)

        self.conv = nn.Conv2d(c2_planes + d2_planes, out_planes - 1, kernel_size=3, padding=1, bias=True)

    def forward(self, geo_feat, invdepth):
        cor = F.relu(self.convc1(geo_feat))
        cor = F.relu(self.convc2(cor))

        dep = F.relu(self.convd1(invdepth))
        dep = F.relu(self.convd2(dep))

        cor_dep = torch.cat([cor, dep], dim=1)
        out = F.relu(self.conv(cor_dep))
        return torch.cat([out, invdepth], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self, opts, hidden_dim, input_dim):
        super(UpdateBlock, self).__init__()
        self.opts = opts

        L = opts.corr_levels
        r = opts.corr_radius
        G = getattr(opts, 'gwc_groups', 8)
        cor_planes = L * (2 * r + 1) * (G + 1)

        self.encoder = MotionEncoder(cor_planes)
        encoder_output_dim = 128

        self.cam = ChannelAttentionEnhancement(input_dim)
        self.sam = SpatialAttentionExtractor(kernel_size=7)

        small_k = getattr(opts, "gru_small_kernel", 1)
        large_k = getattr(opts, "gru_large_kernel", 5)
        large_mode = getattr(opts, "gru_large_mode", "sep")
        self.gru = SelectiveConvGRU(
            hidden_dim,
            encoder_output_dim + input_dim,
            small_kernel_size=small_k,
            large_kernel_size=large_k,
            large_mode=large_mode
        )

        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)
        factor = 2 ** self.opts.num_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor ** 2) * 9, 1, padding=0)
        )

    def forward(self, net, inp, corr=None, inv_depth=None, no_upsample=False):
        geo_feat = corr

        motion_feat = self.encoder(geo_feat, inv_depth)

        cam_w = self.cam(inp)
        inp_cam = inp * cam_w

        att = self.sam(inp_cam)

        gru_inp = torch.cat([inp_cam, motion_feat], dim=1)
        net = self.gru(att, net, gru_inp)

        delta_inv_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_inv_depth, None

        mask = 0.25 * self.mask(net)
        return net, delta_inv_depth, mask


class DiGRU(nn.Module):
    def __init__(self, opts, hidden_dim, input_dim):
        super(DiGRU, self).__init__()
        self.opts = opts

        L = opts.corr_levels
        r = opts.corr_radius
        G = getattr(opts, "gwc_groups", 8)
        cor_planes = L * (2 * r + 1) * (G + 1)

        self.encoder = MotionEncoder(cor_planes)
        encoder_output_dim = 128

        self.cam = ChannelAttentionEnhancement(input_dim)
        self.da_att = DistortionAwareAttention(kernel_size=7)

        small_k = getattr(opts, "gru_small_kernel", 1)
        large_k = getattr(opts, "gru_large_kernel", 5)
        large_mode = getattr(opts, "gru_large_mode", "sep")
        self.gru = SelectiveConvGRU(
            hidden_dim,
            encoder_output_dim + input_dim,
            small_kernel_size=small_k,
            large_kernel_size=large_k,
            large_mode=large_mode,
        )

        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)
        factor = 2 ** self.opts.num_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor ** 2) * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr=None, inv_depth=None, distortion_map=None, no_upsample=False):
        geo_feat = corr

        motion_feat = self.encoder(geo_feat, inv_depth)

        cam_w = self.cam(inp)
        inp_cam = inp * cam_w

        att = self.da_att(inp_cam, distortion_map)

        gru_inp = torch.cat([inp_cam, motion_feat], dim=1)
        net = self.gru(att, net, gru_inp)

        delta_inv_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_inv_depth, None

        mask = 0.25 * self.mask(net)
        return net, delta_inv_depth, mask
