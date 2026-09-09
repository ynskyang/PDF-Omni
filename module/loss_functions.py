from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _ensure_bool_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    assert mask.shape == like.shape, f"mask shape {mask.shape} != target {like.shape}"
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return mask


def _finite_only(x: torch.Tensor, where: torch.Tensor) -> None:
    if torch.isinf(x[where]).any() or torch.isnan(x[where]).any():
        raise ValueError("Non-finite values found inside valid region.")


def sequence_loss(preds, gt, valid, loss_gamma: float = 0.9, use_smooth_l1: bool = False) -> torch.Tensor:
    n_predictions = len(preds)
    assert n_predictions >= 1, "At least one prediction required"

    adjusted = loss_gamma ** (15.0 / n_predictions)

    loss = 0.0
    mask = _ensure_bool_mask(valid, gt)
    _finite_only(gt, mask)

    for i in range(n_predictions):
        pi = preds[i]
        assert pi.shape == gt.shape, f"pred[{i}] shape {pi.shape} != gt {gt.shape}"
        _finite_only(pi, mask)

        w = adjusted ** (n_predictions - i - 1)
        if use_smooth_l1:
            li = F.smooth_l1_loss(pi[mask], gt[mask], reduction="mean")
        else:
            li = (pi - gt).abs()[mask].mean()
        loss = loss + w * li

    return loss


def lec_loss(
    prob_volume: torch.Tensor,
    gt_disp: torch.Tensor,
    img: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
    detach_entropy: bool = True,
    full_num_invdepth=None,
    return_stats: bool = False,
) -> torch.Tensor:

    if gt_disp.ndim == 3:
        gt_disp = gt_disp.unsqueeze(1)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)

    B, Nd, H, W = prob_volume.shape

    prob = torch.clamp(prob_volume, min=eps)
    prob = prob / (prob.sum(dim=1, keepdim=True) + eps)
    entropy = -(prob * torch.log(prob + eps)).sum(dim=1, keepdim=True)
    h_norm = entropy / max(math.log(Nd), eps)
    h_norm = torch.clamp(h_norm, 0.0, 1.0)
    if detach_entropy:
        h_norm = h_norm.detach()

    radius = torch.ceil(h_norm * float(Nd - 1)).to(dtype=torch.long)
    radius = torch.clamp(radius, min=1)

    factor_h = max(1, gt_disp.shape[-2] // H)
    factor_w = max(1, gt_disp.shape[-1] // W)
    gt_down = F.avg_pool2d(gt_disp, kernel_size=(factor_h, factor_w), stride=(factor_h, factor_w))
    mask_down = -F.max_pool2d(
        -valid_mask.float(), kernel_size=(factor_h, factor_w), stride=(factor_h, factor_w)
    )

    if full_num_invdepth is None:
        full_num_invdepth = Nd
    gt_to_volume_scale = float(full_num_invdepth) / float(Nd)
    gt_idx = torch.round(gt_down / gt_to_volume_scale).clamp(0, Nd - 1).to(dtype=torch.long)
    d_indices = torch.arange(Nd, device=prob_volume.device, dtype=torch.long).view(1, Nd, 1, 1)
    dist = (d_indices - gt_idx).abs()
    neigh = (dist <= radius).to(dtype=prob_volume.dtype)

    mass = (prob * neigh).sum(dim=1, keepdim=True)
    loss_map = F.relu(1.0 - mass)

    denom = mask_down.sum() + 1e-6
    loss = (loss_map * mask_down).sum() / denom
    if not return_stats:
        return loss

    with torch.no_grad():
        valid_down = mask_down > 0.5
        if valid_down.any():
            entropy_mean = entropy[valid_down].mean()
            radius_mean = radius.float()[valid_down].mean()
            mass_mean = mass[valid_down].mean()
        else:
            entropy_mean = entropy.new_tensor(0.0)
            radius_mean = entropy.new_tensor(0.0)
            mass_mean = entropy.new_tensor(0.0)
        stats = {
            "entropy": entropy_mean.detach(),
            "radius": radius_mean.detach(),
            "mass": mass_mean.detach(),
            "gt_to_volume_scale": entropy.new_tensor(gt_to_volume_scale),
        }
    return loss, stats
