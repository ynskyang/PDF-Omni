# The multi-resolution transformer (MRT) components used by this encoder are
# adapted from the official S2M2 implementation (see module/mrt_*.py):
#   https://github.com/junhong-3dv/s2m2 (CC BY-NC 4.0, non-commercial use only)

import torch
import torch.nn as nn

from .mrt_layers import CNNEncoder
from .mrt_model import StackedMRT, Unet


class PDFOmniEncoder(nn.Module):
    def __init__(
        self,
        base_channel: int = 32,
        dim_expansion: int = 2,
        num_transformer: int = 3,
        downsample_twice: bool = False,
        pe_mode: str = "grid",
    ):
        super().__init__()
        self.downsample_twice = downsample_twice

        self.cnn_backbone = CNNEncoder(output_dim=base_channel)

        self.feat_pyramid = Unet(
            dims=[base_channel, base_channel, 2 * base_channel],
            dim_expansion=dim_expansion,
            use_gate_fusion=True,
            use_pe=True,
            pe_mode=pe_mode,
            n_attn=num_transformer * 2,
        )

        self.transformer = StackedMRT(
            num_transformer=num_transformer,
            dims=[base_channel, base_channel, 2 * base_channel],
            num_heads=1,
            dim_expansion=dim_expansion,
            use_gate_fusion=True,
        )

    def forward(self, imgs):
        is_list = isinstance(imgs, (list, tuple))
        if is_list:
            batch_size = imgs[0].shape[0]
            x = torch.cat(imgs, dim=0)
        else:
            x = imgs

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat_4x, feat_2x = self.cnn_backbone(x)

        feat_in = feat_4x if self.downsample_twice else feat_2x
        z0, z1, z2, z3 = self.feat_pyramid(feat_in)

        feat_out = self.transformer(z0, z1, z2, z3)

        if is_list:
            feats = torch.split(feat_out, batch_size, dim=0)
            return list(feats)

        return feat_out
