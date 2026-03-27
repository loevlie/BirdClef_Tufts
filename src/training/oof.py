"""OOF cross-validation and ensemble weight optimization for ProtoSSM."""

from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.constants import N_WINDOWS
from src.models.proto_ssm import ProtoSSMv2
from src.evaluation.metrics import macro_auc_skip_empty
from .trainer import train_proto_ssm_single


def site_stratified_kfold(n_files, file_groups, n_splits=5, seed=42):
    """Split files into folds, distributing each site's files evenly across folds.

    Unlike GroupKFold (which puts entire sites in one fold), this splits
    files *within* each site across folds. This avoids the problem where
    one dominant site (e.g. S22 with 39/59 files) creates wildly imbalanced folds.

    Returns list of (train_idx, val_idx) tuples.
    """
    rng = np.random.default_rng(seed)

    # Group file indices by site
    site_to_files = defaultdict(list)
    for i, g in enumerate(file_groups):
        site_to_files[g].append(i)

    # Assign each file to a fold, distributing each site evenly
    fold_assignments = np.zeros(n_files, dtype=int)
    for site, file_indices in site_to_files.items():
        indices = np.array(file_indices)
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            fold_assignments[idx] = i % n_splits

    # Build train/val splits
    splits = []
    for fold in range(n_splits):
        val_idx = np.where(fold_assignments == fold)[0]
        train_idx = np.where(fold_assignments != fold)[0]
        splits.append((train_idx, val_idx))

    return splits


def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=(0, 1, -1)):
    """TTA by circular-shifting the 12-window embedding sequence.
    Averages predictions from shifted versions for more robust output."""
    all_preds = []
    model.eval()

    for shift in shifts:
        if shift == 0:
            e = emb_files
            l = logits_files
        else:
            e = np.roll(emb_files, shift, axis=1)
            l = np.roll(logits_files, shift, axis=1)

        with torch.no_grad():
            out, _, _ = model(
                torch.tensor(e, dtype=torch.float32),
                torch.tensor(l, dtype=torch.float32),
                site_ids=torch.tensor(site_ids, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
            pred = out.numpy()

        # Reverse the shift on predictions
        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


def run_proto_ssm_oof(emb_files, logits_files, labels_files,
                      site_ids_all, hours_all,
                      file_families, file_groups,
                      n_families, class_to_family,
                      n_classes, ssm_cfg, train_cfg,
                      tta_shifts=(0,), device=None, verbose=True):
    """Run GroupKFold OOF cross-validation for ProtoSSM v4.

    Parameters
    ----------
    emb_files : ndarray, shape (n_files, n_windows, d_emb)
    logits_files : ndarray, shape (n_files, n_windows, n_classes)
    labels_files : ndarray, shape (n_files, n_windows, n_classes)
    site_ids_all : ndarray, shape (n_files,)
    hours_all : ndarray, shape (n_files,)
    file_families : ndarray, shape (n_files, n_families)
        Taxonomic family soft-labels per file.
    file_groups : array-like, shape (n_files,)
        Group labels for GroupKFold (e.g. taxonomy family per file).
    n_families : int
        Number of unique families.
    class_to_family : list[int]
        Mapping from species index to family index.
    n_classes : int
        Number of species classes.
    ssm_cfg : dict
        Architecture hyperparameters (proto_ssm section of CFG).
    train_cfg : dict
        Training hyperparameters (proto_ssm_train section of CFG).
    tta_shifts : tuple[int]
        Window shifts for test-time augmentation.
    device : torch.device or None
        Device for model (defaults to CPU).
    verbose : bool
        Print fold-level progress.

    Returns
    -------
    oof_preds : ndarray, shape (n_files, n_windows, n_classes)
    fold_histories : list[dict]
    fold_alphas : list[ndarray]
    """
    if device is None:
        device = torch.device("cpu")

    n_splits = train_cfg.get("oof_n_splits", 5)
    n_files = len(emb_files)

    oof_preds = np.zeros((n_files, N_WINDOWS, n_classes), dtype=np.float32)
    fold_histories = []
    fold_alphas = []

    splits = site_stratified_kfold(n_files, file_groups, n_splits=n_splits)

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            print(f"\n--- Fold {fold_i+1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) ---")

        fold_model = ProtoSSMv2(
            d_input=emb_files.shape[2],
            d_model=ssm_cfg["d_model"],
            d_state=ssm_cfg["d_state"],
            n_ssm_layers=ssm_cfg["n_ssm_layers"],
            n_classes=n_classes,
            n_windows=N_WINDOWS,
            dropout=ssm_cfg["dropout"],
            n_sites=ssm_cfg["n_sites"],
            meta_dim=ssm_cfg["meta_dim"],
            use_cross_attn=ssm_cfg.get("use_cross_attn", True),
            cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
        ).to(device)

        # Initialize prototypes
        emb_flat_fold = emb_files[train_idx].reshape(-1, emb_files.shape[2])
        labels_flat_fold = labels_files[train_idx].reshape(-1, n_classes)
        fold_model.init_prototypes_from_data(
            torch.tensor(emb_flat_fold, dtype=torch.float32),
            torch.tensor(labels_flat_fold, dtype=torch.float32)
        )
        fold_model.init_family_head(n_families, class_to_family)

        # Train on fold
        fold_model, fold_hist = train_proto_ssm_single(
            fold_model,
            emb_files[train_idx], logits_files[train_idx], labels_files[train_idx].astype(np.float32),
            site_ids_train=site_ids_all[train_idx], hours_train=hours_all[train_idx],
            emb_val=emb_files[val_idx], logits_val=logits_files[val_idx],
            labels_val=labels_files[val_idx].astype(np.float32),
            site_ids_val=site_ids_all[val_idx], hours_val=hours_all[val_idx],
            file_families_train=file_families[train_idx],
            file_families_val=file_families[val_idx],
            cfg=train_cfg, verbose=verbose,
        )

        # OOF predictions with TTA
        fold_model.eval()
        if len(tta_shifts) > 1:
            oof_preds[val_idx] = temporal_shift_tta(
                emb_files[val_idx], logits_files[val_idx], fold_model,
                site_ids_all[val_idx], hours_all[val_idx], shifts=tta_shifts
            )
        else:
            with torch.no_grad():
                val_emb = torch.tensor(emb_files[val_idx], dtype=torch.float32)
                val_logits = torch.tensor(logits_files[val_idx], dtype=torch.float32)
                val_sites = torch.tensor(site_ids_all[val_idx], dtype=torch.long)
                val_hours = torch.tensor(hours_all[val_idx], dtype=torch.long)
                val_out, _, _ = fold_model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
                oof_preds[val_idx] = val_out.numpy()

        fold_alphas.append(torch.sigmoid(fold_model.fusion_alpha).detach().numpy().copy())
        fold_histories.append(fold_hist)

    return oof_preds, fold_histories, fold_alphas


def optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_true_flat):
    """Grid search over blend weights to find optimal ProtoSSM ensemble weight.

    Parameters
    ----------
    oof_proto_flat : ndarray, shape (n_samples, n_classes)
        OOF predictions from ProtoSSM (flattened across windows).
    oof_mlp_flat : ndarray, shape (n_samples, n_classes)
        OOF predictions from MLP probe (flattened across windows).
    y_true_flat : ndarray, shape (n_samples, n_classes)
        Ground truth labels (flattened across windows).

    Returns
    -------
    best_w : float
        Optimal weight for ProtoSSM (1 - best_w goes to MLP).
    best_auc : float
        Best macro AUC achieved at the optimal weight.
    results : list[tuple[float, float]]
        All (weight, auc) pairs evaluated.
    """
    weights = np.arange(0.0, 1.05, 0.05)
    results = []

    for w in weights:
        blended = w * oof_proto_flat + (1.0 - w) * oof_mlp_flat
        try:
            auc = macro_auc_skip_empty(y_true_flat, blended)
        except Exception:
            auc = 0.0
        results.append((w, auc))

    best_w, best_auc = max(results, key=lambda x: x[1])
    return best_w, best_auc, results
