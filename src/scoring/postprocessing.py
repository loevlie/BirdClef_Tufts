"""V18 post-processing: rank-aware scaling and adaptive delta-shift smoothing."""

import numpy as np


def rank_aware_scaling(scores, n_windows=12, power=0.5):
    """Scale each window by file_max^power.
    Suppresses predictions in uncertain files, boosts confident files."""
    N, C = scores.shape
    assert N % n_windows == 0
    view = scores.reshape(-1, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(np.clip(file_max, 1e-8, None), power)
    return (view * scale).reshape(N, C)


def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """Confidence-dependent temporal smoothing.
    Low-confidence windows get more smoothing from neighbors."""
    n_files = probs.shape[0] // n_windows
    result = probs.copy()
    view = result.reshape(n_files, n_windows, -1)
    p_view = probs.reshape(n_files, n_windows, -1)
    for i in range(1, n_windows - 1):
        conf = p_view[:, i, :].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (p_view[:, i - 1, :] + p_view[:, i + 1, :]) / 2.0
        view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
    return result
