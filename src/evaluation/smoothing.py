"""Temporal smoothing for texture classes and event birds."""

import numpy as np

from src.constants import N_WINDOWS


def smooth_cols_fixed12(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()

    s = scores.copy()
    assert len(s) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    view = s.reshape(-1, N_WINDOWS, s.shape[1])

    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)

    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s


def smooth_events_fixed12(scores, cols, alpha=0.15):
    """Soft max-pool context for event birds (Aves).
    Uses local_max instead of average neighbor, preserving transient call detection."""
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
    return s
