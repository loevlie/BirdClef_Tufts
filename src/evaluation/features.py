"""Classwise feature engineering for embedding probes."""

import numpy as np

from src.constants import N_WINDOWS


def seq_features_1d(v):
    """
    v: shape (n_rows,), ordered as full-file blocks of 12 windows
    Extended: tambah std_v untuk capture variance temporal dalam file
    """
    assert len(v) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    x = v.reshape(-1, N_WINDOWS)

    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v  = np.repeat(x.max(axis=1),  N_WINDOWS)
    std_v  = np.repeat(x.std(axis=1),  N_WINDOWS)

    return prev_v, next_v, mean_v, max_v, std_v


def build_class_features(emb_proj, raw_col, prior_col, base_col):
    """
    emb_proj: (n, d)
    raw_col, prior_col, base_col: (n,)
    returns: (n, d + 13)

    Fitur: embedding + 7 sequential + 3 interaction + std + 3 diff
    """
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)

    # Diff features: posisi window relatif terhadap konteks file
    diff_mean = base_col - mean_base   # apakah window ini lebih tinggi dari rata2 file?
    diff_prev = base_col - prev_base   # onset: naik dari window sebelumnya?
    diff_next = base_col - next_base   # offset: turun ke window berikutnya?

    feats = np.concatenate([
        emb_proj,
        raw_col[:, None],
        prior_col[:, None],
        base_col[:, None],
        prev_base[:, None],
        next_base[:, None],
        mean_base[:, None],
        max_base[:, None],
        std_base[:, None],             # variance temporal dalam file
        diff_mean[:, None],            # deviasi dari mean file
        diff_prev[:, None],            # deteksi onset
        diff_next[:, None],            # deteksi offset
        # interaction terms
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1)

    return feats.astype(np.float32, copy=False)
