"""Mixup + CutMix hybrid augmentation for file-level training."""

import numpy as np
import torch


def mixup_cutmix(emb, logits, labels, alpha=0.4, cutmix_prob=0.3):
    """Hybrid Mixup/CutMix on the temporal dimension.

    With probability cutmix_prob, swaps a random time segment between files.
    Otherwise does standard mixup.
    """
    B, T, D = emb.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B)

    if np.random.rand() < cutmix_prob:
        # CutMix on time dimension
        cut_len = max(1, int(T * (1 - lam)))
        cut_start = np.random.randint(0, T - cut_len + 1)
        new_emb = emb.clone() if isinstance(emb, torch.Tensor) else torch.tensor(emb)
        new_emb[:, cut_start:cut_start + cut_len, :] = new_emb[idx, cut_start:cut_start + cut_len, :]
        new_logits = logits.clone() if isinstance(logits, torch.Tensor) else torch.tensor(logits)
        new_logits[:, cut_start:cut_start + cut_len, :] = new_logits[idx, cut_start:cut_start + cut_len, :]
        lam_actual = 1.0 - cut_len / T
        new_labels = lam_actual * labels + (1 - lam_actual) * labels[idx]
    else:
        # Standard Mixup
        new_emb = lam * emb + (1 - lam) * emb[idx]
        new_logits = lam * logits + (1 - lam) * logits[idx]
        new_labels = lam * labels + (1 - lam) * labels[idx]

    return new_emb, new_logits, new_labels
