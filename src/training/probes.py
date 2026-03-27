"""Classwise embedding-probe OOF helpers for per-species fine-tuning."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from src.evaluation.features import build_class_features
from src.evaluation.metrics import macro_auc_skip_empty
from src.scoring.priors import fit_prior_tables
from src.scoring.fusion import fuse_scores_with_tables


def run_oof_embedding_probe(
    scores_raw,
    emb,
    meta_df,
    y_true,
    sc_clean,
    Y_SC,
    fuse_kwargs,
    pca_dim=64,
    min_pos=8,
    C=0.25,
    alpha=0.5,
    probe_backend="mlp",
    mlp_params=None,
    lgbm_params=None,
    verbose=True,
):
    """Train per-class LogisticRegression or MLP probes with fold-safe priors.

    Parameters
    ----------
    scores_raw : ndarray, shape (n_windows, n_classes)
        Raw Perch logits for all soundscape windows.
    emb : ndarray, shape (n_windows, d_emb)
        Perch embeddings for all soundscape windows.
    meta_df : pd.DataFrame
        Per-window metadata with ``filename``, ``site``, ``hour_utc`` columns.
    y_true : ndarray, shape (n_windows, n_classes)
        Binary ground-truth labels.
    sc_clean : pd.DataFrame
        Soundscape-level metadata for prior tables (with ``filename``).
    Y_SC : ndarray, shape (n_soundscapes, n_classes)
        Binary truth matrix aligned with *sc_clean*.
    fuse_kwargs : dict
        Keyword arguments forwarded to :func:`fuse_scores_with_tables`
        (all idx_* arrays plus lambda/smooth params).
    pca_dim : int
        Number of PCA components for embedding projection.
    min_pos : int
        Minimum positive samples to attempt probe for a class.
    C : float
        Regularization strength for LogisticRegression backend.
    alpha : float
        Blend weight: ``(1-alpha)*base + alpha*probe``.
    probe_backend : str
        One of ``"mlp"``, ``"lgbm"``, ``"logreg"``.
    mlp_params : dict or None
        Kwargs for ``MLPClassifier`` when ``probe_backend="mlp"``.
    lgbm_params : dict or None
        Kwargs for ``LGBMClassifier`` when ``probe_backend="lgbm"``.
    verbose : bool
        Show tqdm progress bars.

    Returns
    -------
    dict
        Keys: oof_base, oof_final, modeled_counts, score_base, score_final.
    """
    if mlp_params is None:
        mlp_params = {}
    if lgbm_params is None:
        lgbm_params = {}

    groups = meta_df["filename"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    oof_base_local = np.zeros_like(scores_raw, dtype=np.float32)
    oof_final = np.zeros_like(scores_raw, dtype=np.float32)

    modeled_counts = np.zeros(scores_raw.shape[1], dtype=np.int32)

    split_list = list(gkf.split(scores_raw, groups=groups))

    for fold, (tr_idx, va_idx) in enumerate(
        tqdm(split_list, desc="Embedding-probe folds", disable=not verbose), 1
    ):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_df.iloc[va_idx]["filename"].tolist())

        # Fold-safe priors
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]
        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        base_tr, prior_tr = fuse_scores_with_tables(
            scores_raw[tr_idx],
            sites=meta_df.iloc[tr_idx]["site"].to_numpy(),
            hours=meta_df.iloc[tr_idx]["hour_utc"].to_numpy(),
            tables=tables,
            **fuse_kwargs,
        )
        base_va, prior_va = fuse_scores_with_tables(
            scores_raw[va_idx],
            sites=meta_df.iloc[va_idx]["site"].to_numpy(),
            hours=meta_df.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
            **fuse_kwargs,
        )

        oof_base_local[va_idx] = base_va
        oof_final[va_idx] = base_va

        # Embedding preprocessing on train fold only
        scaler = StandardScaler()
        emb_tr_s = scaler.fit_transform(emb[tr_idx])
        emb_va_s = scaler.transform(emb[va_idx])

        n_comp = min(pca_dim, emb_tr_s.shape[0] - 1, emb_tr_s.shape[1])
        pca = PCA(n_components=n_comp)
        Z_tr = pca.fit_transform(emb_tr_s).astype(np.float32)
        Z_va = pca.transform(emb_va_s).astype(np.float32)

        class_iterator = np.where(y_true[tr_idx].sum(axis=0) >= min_pos)[0].tolist()

        for cls_idx in tqdm(
            class_iterator, desc=f"Fold {fold} classes", leave=False, disable=not verbose
        ):
            y_tr = y_true[tr_idx, cls_idx]

            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue

            X_tr_cls = build_class_features(
                Z_tr,
                raw_col=scores_raw[tr_idx, cls_idx],
                prior_col=prior_tr[:, cls_idx],
                base_col=base_tr[:, cls_idx],
            )
            X_va_cls = build_class_features(
                Z_va,
                raw_col=scores_raw[va_idx, cls_idx],
                prior_col=prior_va[:, cls_idx],
                base_col=base_va[:, cls_idx],
            )

            # Choose probe backend: mlp | lgbm | logreg
            backend = probe_backend
            n_pos = int(y_tr.sum())
            n_neg = len(y_tr) - n_pos

            if backend == "mlp":
                # MLPClassifier does not support sample_weight
                # Use oversampling: duplicate positives to balance
                if n_pos > 0 and n_neg > n_pos:
                    repeat = max(1, n_neg // n_pos)
                    pos_idx = np.where(y_tr == 1)[0]
                    X_bal = np.vstack([X_tr_cls, np.tile(X_tr_cls[pos_idx], (repeat, 1))])
                    y_bal = np.concatenate([y_tr, np.ones(len(pos_idx) * repeat, dtype=y_tr.dtype)])
                else:
                    X_bal, y_bal = X_tr_cls, y_tr
                clf = MLPClassifier(**mlp_params)
                clf.fit(X_bal, y_bal)
                pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
            elif backend == "lgbm" and _LGBM_AVAILABLE:
                scale_pos = max(1.0, n_neg / max(n_pos, 1))
                clf = LGBMClassifier(
                    **lgbm_params,
                    scale_pos_weight=scale_pos,
                )
                clf.fit(X_tr_cls, y_tr)
                pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
            else:
                clf = LogisticRegression(
                    C=C, max_iter=400, solver="liblinear",
                    class_weight="balanced",
                )
                clf.fit(X_tr_cls, y_tr)
                pred_va = clf.decision_function(X_va_cls).astype(np.float32)

            oof_final[va_idx, cls_idx] = (
                (1.0 - alpha) * base_va[:, cls_idx]
                + alpha * pred_va
            )

            modeled_counts[cls_idx] += 1

    score_base = macro_auc_skip_empty(y_true, oof_base_local)
    score_final = macro_auc_skip_empty(y_true, oof_final)

    return {
        "oof_base": oof_base_local,
        "oof_final": oof_final,
        "modeled_counts": modeled_counts,
        "score_base": score_base,
        "score_final": score_final,
    }
