"""Species-frequency aware focal loss."""

import numpy as np
import torch
import torch.nn.functional as F


def build_class_freq_weights(Y_FULL, cap=10.0):
    """Build per-class weights inversely proportional to sqrt(frequency)."""
    pos_count = Y_FULL.sum(axis=0).astype(np.float32) + 1.0
    total = Y_FULL.shape[0]
    freq = pos_count / total
    weights = 1.0 / (freq ** 0.5)
    weights = np.clip(weights, 1.0, cap)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def species_focal_loss(logits, targets, class_weights,
                       gamma=2.5, label_smoothing=0.03):
    """Focal loss weighted by species frequency."""
    targets_smooth = targets * (1 - label_smoothing) + label_smoothing / 2.0
    bce = F.binary_cross_entropy_with_logits(
        logits, targets_smooth, reduction="none")
    pt = torch.exp(-bce)
    focal = ((1 - pt) ** gamma) * bce
    w = class_weights.to(logits.device).unsqueeze(0)
    return (focal * w).mean()
