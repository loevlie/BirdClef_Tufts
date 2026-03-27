"""Score fusion: prior-weighted base scores and OOF stacking."""

import numpy as np
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.evaluation.smoothing import smooth_cols_fixed12, smooth_events_fixed12
from src.scoring.priors import fit_prior_tables, prior_logits_from_tables


def fuse_scores_with_tables(base_scores, sites, hours, tables,
                            idx_mapped_active_event,
                            idx_mapped_active_texture,
                            idx_selected_proxy_active_texture,
                            idx_selected_prioronly_active_event,
                            idx_selected_prioronly_active_texture,
                            idx_unmapped_inactive,
                            idx_active_texture,
                            idx_active_event,
                            lambda_event=0.3,
                            lambda_texture=0.3,
                            lambda_proxy_texture=0.3,
                            smooth_texture=0.35,
                            smooth_event=0.15):
    """Fuse raw Perch scores with metadata prior tables.

    Parameters
    ----------
    base_scores : np.ndarray, shape (n_rows, n_classes)
    sites, hours : array-like
        Site identifiers and UTC hours for each row.
    tables : dict
        Output of :func:`fit_prior_tables`.
    idx_mapped_active_event, idx_mapped_active_texture,
    idx_selected_proxy_active_texture,
    idx_selected_prioronly_active_event,
    idx_selected_prioronly_active_texture,
    idx_unmapped_inactive : np.ndarray[int32]
        Class-index arrays computed from the taxonomy mapping.
    idx_active_texture, idx_active_event : np.ndarray[int32]
        Class-index arrays for smoothing.
    lambda_event, lambda_texture, lambda_proxy_texture : float
        Prior blending weights.
    smooth_texture, smooth_event : float
        Temporal smoothing alphas.

    Returns
    -------
    scores : np.ndarray, shape (n_rows, n_classes)
    prior : np.ndarray, shape (n_rows, n_classes)
    """
    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, tables)

    # mapped active
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]

    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]

    # selected frog proxies
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]

    # prior-only active unmapped
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[:, idx_selected_prioronly_active_event]

    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[:, idx_selected_prioronly_active_texture]

    # inactive unmapped
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior


def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC,
                         fuse_kwargs, n_splits=5, verbose=True):
    """Build honest out-of-fold base and prior meta-features.

    Parameters
    ----------
    scores_full_raw : np.ndarray, shape (n_rows, n_classes)
    meta_full : pd.DataFrame
        Must contain ``filename``, ``site``, ``hour_utc``.
    sc_clean : pd.DataFrame
        Soundscape-level metadata with ``filename`` column for fold splitting.
    Y_SC : np.ndarray
        Binary truth matrix aligned with *sc_clean*.
    fuse_kwargs : dict
        Keyword arguments forwarded to :func:`fuse_scores_with_tables`
        (all idx_* arrays plus lambda/smooth params).
    n_splits : int
    verbose : bool

    Returns
    -------
    oof_base : np.ndarray
    oof_prior : np.ndarray
    fold_id : np.ndarray[int16]
    """
    groups_full = meta_full["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_id = np.full(len(meta_full), -1, dtype=np.int16)

    splits = list(gkf.split(scores_full_raw, groups=groups_full))
    iterator = tqdm(splits, desc="OOF base/prior folds", disable=not verbose)

    for fold, (tr_idx, va_idx) in enumerate(iterator, 1):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())

        # Fold-safe prior tables: exclude all validation files
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]

        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        va_base, va_prior = fuse_scores_with_tables(
            scores_full_raw[va_idx],
            sites=meta_full.iloc[va_idx]["site"].to_numpy(),
            hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
            **fuse_kwargs,
        )

        oof_base[va_idx] = va_base
        oof_prior[va_idx] = va_prior
        fold_id[va_idx] = fold

    assert (fold_id >= 0).all()
    return oof_base, oof_prior, fold_id
