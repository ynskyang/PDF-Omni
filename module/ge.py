import math
import torch
import torch.nn as nn


class GE(nn.Module):
    def __init__(self, input_dim=3, mapping_size=256, scale=10.0, out_dim=128, learnable_B=False):
        super().__init__()
        self.out_dim = out_dim

        B_mat = torch.randn((input_dim, mapping_size)) * scale
        if learnable_B:
            self.B = nn.Parameter(B_mat)
        else:
            self.register_buffer("B", B_mat)

        self.mlp = nn.Sequential(
            nn.Linear(mapping_size * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"GE expects x with shape [B, 3, H, W], got {tuple(x.shape)}")

        x = x.permute(0, 2, 3, 1)

        B_mat = self.B
        if B_mat.dtype != x.dtype:
            B_mat = B_mat.to(dtype=x.dtype)
        proj = (2.0 * math.pi) * torch.matmul(x, B_mat)
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        out = self.mlp(fourier)
        return out.permute(0, 3, 1, 2)
