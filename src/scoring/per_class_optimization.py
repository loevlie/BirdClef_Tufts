"""Per-class ensemble weight and threshold optimization on OOF predictions."""

import numpy as np
from sklearn.metrics import roc_auc_score


def optimize_per_class_weights(oof_proto, oof_mlp, y_true, n_steps=21):
    """Find optimal per-class blend weight between ProtoSSM and MLP on OOF.

    For each class, searches w in [0, 1] that maximizes AUC:
        blended = w * proto + (1-w) * mlp

    Returns array of shape (n_classes,) with optimal weights.
    """
    n_classes = y_true.shape[1]
    weights = np.full(n_classes, 0.5, dtype=np.float32)
    grid = np.linspace(0, 1, n_steps)

    for ci in range(n_classes):
        n_pos = y_true[:, ci].sum()
        if n_pos == 0 or n_pos == len(y_true):
            continue
        best_auc = -1
        best_w = 0.5
        for w in grid:
            blended = w * oof_proto[:, ci] + (1 - w) * oof_mlp[:, ci]
            try:
                auc = roc_auc_score(y_true[:, ci], blended)
                if auc > best_auc:
                    best_auc = auc
                    best_w = w
            except ValueError:
                continue
        weights[ci] = best_w

    return weights


def optimize_per_class_thresholds(probs, y_true, grid=None):
    """Find per-class probability thresholds that maximize macro F1 on OOF.

    Returns array of shape (n_classes,) with optimal thresholds.
    """
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.05)

    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)

    for ci in range(n_classes):
        n_pos = y_true[:, ci].sum()
        if n_pos == 0:
            thresholds[ci] = 0.99
            continue
        if n_pos == len(y_true):
            thresholds[ci] = 0.01
            continue

        best_f1 = -1
        for t in grid:
            preds = (probs[:, ci] >= t).astype(float)
            tp = (preds * y_true[:, ci]).sum()
            fp = (preds * (1 - y_true[:, ci])).sum()
            fn = ((1 - preds) * y_true[:, ci]).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                thresholds[ci] = t

    return thresholds


def apply_per_class_blend(proto_scores, mlp_scores, per_class_weights):
    """Blend ProtoSSM and MLP with per-class optimized weights."""
    return per_class_weights[None, :] * proto_scores + (1 - per_class_weights[None, :]) * mlp_scores


def apply_per_class_thresholds(probs, thresholds):
    """Zero out predictions below per-class thresholds.
    Soft version: scale down sub-threshold predictions rather than hard cutoff."""
    mask = probs >= thresholds[None, :]
    # Soft: sub-threshold gets multiplied by 0.1 instead of zeroed
    result = np.where(mask, probs, probs * 0.1)
    return result
