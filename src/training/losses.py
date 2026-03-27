"""Focal loss for multi-label classification."""

import torch
import torch.nn.functional as F


def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, reduction="mean"):
    """Focal loss for multi-label classification.
    Reduces contribution of easy examples, focuses on hard ones.
    Critical for rare species where BCE is dominated by easy negatives."""
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    return loss
