"""Temperature scaling and file-level confidence calibration."""

import numpy as np


def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    """Rank 1/2 technique: scale each window's predictions by the file's
    top-K mean confidence. This suppresses noise in files where the model
    is uncertain and boosts predictions where it's confident."""
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    # Top-K mean per file per class
    sorted_view = np.sort(view, axis=1)  # sort along windows
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)  # (F, 1, C)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)


def apply_temperature_and_scale(scores, class_temperatures, n_windows=12, top_k=2):
    """Apply per-class temperature scaling, sigmoid, and optional file-level confidence scaling.

    Parameters
    ----------
    scores : np.ndarray, shape (n_rows, n_classes)
        Raw logits (pre-sigmoid).
    class_temperatures : np.ndarray, shape (n_classes,)
        Per-class temperature values.
    n_windows : int
        Windows per file (for file-level scaling reshape).
    top_k : int
        Number of top windows for confidence scaling. Set 0 to disable.

    Returns
    -------
    np.ndarray, shape (n_rows, n_classes)
        Calibrated probabilities.
    """
    scaled = scores / class_temperatures[None, :]
    probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
    if top_k > 0:
        probs = file_level_confidence_scale(probs, n_windows=n_windows, top_k=top_k)
        probs = np.clip(probs, 0.0, 1.0)
    return probs
