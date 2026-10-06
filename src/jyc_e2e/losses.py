"""Training objectives; final model selection still uses competition metrics."""

import torch
from torch.nn import functional as F


def pairwise_rank_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    times: torch.Tensor,
    *,
    min_target_gap: float = 0.0,
    max_pairs_per_time: int = 8192,
) -> torch.Tensor:
    """Compare stocks only within the same decision timestamp."""

    if scores.ndim != 1 or scores.shape != target.shape or scores.shape != times.shape:
        raise ValueError("scores, target, and times must be equal-length vectors")
    if scores.device != target.device or scores.device != times.device:
        raise ValueError("scores, target, and times must be on the same device")
    if not torch.isfinite(scores).all() or not torch.isfinite(target).all():
        raise ValueError("scores and target must be finite")
    if max_pairs_per_time < 1 or min_target_gap < 0:
        raise ValueError("pair limit must be positive and target gap nonnegative")
    losses = []
    for time in torch.unique(times):
        indices = torch.nonzero(times == time, as_tuple=True)[0]
        if indices.numel() < 2:
            continue
        left, right = torch.triu_indices(
            indices.numel(), indices.numel(), offset=1, device=scores.device
        )
        gap = target[indices[left]] - target[indices[right]]
        usable = gap.abs() > min_target_gap
        left, right, gap = left[usable], right[usable], gap[usable]
        if not gap.numel():
            continue
        if gap.numel() > max_pairs_per_time:
            chosen = torch.randperm(gap.numel(), device=scores.device)[:max_pairs_per_time]
            left, right, gap = left[chosen], right[chosen], gap[chosen]
        difference = scores[indices[left]] - scores[indices[right]]
        losses.append(F.softplus(-gap.sign() * difference).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def score_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    times: torch.Tensor,
    *,
    rank_weight: float = 0.0,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    if scores.ndim != 1 or scores.shape != target.shape or scores.shape != times.shape:
        raise ValueError("scores, target, and times must be equal-length vectors")
    if rank_weight < 0:
        raise ValueError("rank_weight must be nonnegative")
    loss = F.huber_loss(scores, target, delta=huber_delta)
    if rank_weight:
        loss = loss + rank_weight * pairwise_rank_loss(scores, target, times)
    return loss
