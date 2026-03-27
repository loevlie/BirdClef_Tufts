"""File-level mixup augmentation for ProtoSSM training."""

import numpy as np


def mixup_files(emb, logits, labels, site_ids, hours, families, alpha=0.3):
    """File-level mixup augmentation for ProtoSSM training.
    Mixes pairs of files with random lambda from Beta(alpha, alpha).
    Returns augmented versions of all inputs."""
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels, site_ids, hours, families

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 (dominant sample stays dominant)

    perm = np.random.permutation(n)

    emb_mix = lam * emb + (1 - lam) * emb[perm]
    logits_mix = lam * logits + (1 - lam) * logits[perm]
    labels_mix = lam * labels + (1 - lam) * labels[perm]

    # For discrete features (site, hour), keep the dominant sample's values
    families_mix = lam * families + (1 - lam) * families[perm] if families is not None else None

    return emb_mix, logits_mix, labels_mix, site_ids, hours, families_mix
