#!/usr/bin/env python3
"""OOF evaluation: ProtoSSM cross-validation, MLP probe OOF, ensemble weight search."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.constants import N_WINDOWS, DEVICE
from src.data.reshape import reshape_to_files
from src.data.sites import build_site_mapping, get_file_metadata
from src.data.taxonomy import build_taxonomy_groups
from src.evaluation.metrics import macro_auc_skip_empty
from src.pipeline import (
    load_competition_data,
    prepare_labels,
    build_perch_mapping,
    load_or_compute_cache,
    align_truth_to_cache,
    load_or_compute_oof_meta,
)
from src.timer.wallclock import WallTimer
from src.training.oof import run_proto_ssm_oof, optimize_ensemble_weight
from src.training.probes import run_oof_embedding_probe


def parse_args():
    p = argparse.ArgumentParser(description="BirdCLEF 2026 — OOF evaluation pipeline")
    p.add_argument("--config", default="configs/base.yaml", help="Path to YAML config file")
    p.add_argument("--data-dir", default=None, help="Override paths.data_dir")
    p.add_argument("--model-dir", default=None, help="Override paths.model_dir (Perch)")
    p.add_argument("--cache-dir", default=None, help="Override paths.cache_dir")
    p.add_argument("--n-splits", type=int, default=None, help="Override oof_n_splits")
    p.add_argument("--output-dir", default="outputs/eval", help="Directory for results")
    p.add_argument("--name", default=None, help="wandb run name")
    p.add_argument("--notes", default=None, help="wandb experiment notes")
    p.add_argument("--tags", nargs="*", default=None, help="wandb tags")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return p.parse_args()


def main():
    args = parse_args()
    timer = WallTimer(budget_seconds=7200.0)
    timer.stage_start("setup")

    # ── Load config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg_dict = cfg.to_dict()

    data_dir = Path(args.data_dir or cfg.paths.data_dir)
    model_dir = Path(args.model_dir or cfg.paths.model_dir)
    cache_dir = Path(args.cache_dir or cfg.paths.cache_dir)
    cache_input_dir = Path(cfg.paths.cache_input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb tracking ────────────────────────────────────────────────
    import os
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    from src import tracking
    tracking.init(
        name=args.name or f"eval-{Path(args.config).stem}",
        config=cfg_dict,
        tags=args.tags or ["evaluate"],
        notes=args.notes,
    )

    # ── Validate data exists ─────────────────────────────────────────────
    for required in ["taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
        if not (data_dir / required).exists():
            print(f"ERROR: Required file not found: {data_dir / required}")
            sys.exit(1)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("[eval] Loading competition data...")
    taxonomy, sample_sub, soundscape_labels, PRIMARY_LABELS, N_CLASSES = (
        load_competition_data(data_dir)
    )

    print("[eval] Preparing labels...")
    sc_clean, Y_SC, full_files, full_truth, Y_FULL_TRUTH, label_to_idx = (
        prepare_labels(soundscape_labels, PRIMARY_LABELS)
    )
    print(f"  Full files: {len(full_files)}, Classes: {N_CLASSES}")

    # ── 2. Perch mapping ─────────────────────────────────────────────────
    print("[eval] Building Perch mapping...")
    mapping = build_perch_mapping(taxonomy, model_dir, PRIMARY_LABELS, Y_SC, label_to_idx)

    # ── 3. Load Perch cache ──────────────────────────────────────────────
    print("[eval] Resolving Perch cache...")
    pipeline_cfg = {"mode": "train", "verbose": True}
    meta_full, scores_full_raw, emb_full = load_or_compute_cache(
        full_files, data_dir, cache_dir, cache_input_dir, mapping, pipeline_cfg,
    )
    Y_FULL = align_truth_to_cache(full_truth, Y_SC, meta_full)
    print(f"  Samples: {scores_full_raw.shape[0]}, Active classes: {int((Y_FULL.sum(axis=0) > 0).sum())}")

    # ── 4. OOF meta-features ─────────────────────────────────────────────
    fusion_cfg = cfg_dict.get("best_fusion", cfg_dict.get("fusion", {}))
    fuse_kwargs = mapping["fuse_kwargs"]

    oof_base, oof_prior, oof_fold_id = load_or_compute_oof_meta(
        scores_full_raw, meta_full, sc_clean, Y_SC,
        cache_dir, fuse_kwargs, fusion_cfg,
        n_splits=5, verbose=True,
    )

    baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)
    raw_auc = macro_auc_skip_empty(Y_FULL, scores_full_raw)
    print(f"  Raw AUC (no priors): {raw_auc:.6f}")
    print(f"  OOF baseline AUC (with priors): {baseline_oof_auc:.6f}")
    timer.stage_end()

    # ── 5. Reshape to file-level ─────────────────────────────────────────
    timer.stage_start("reshape")
    emb_files, file_list = reshape_to_files(emb_full, meta_full)
    logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
    labels_files, _ = reshape_to_files(Y_FULL, meta_full)

    n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
    site_to_idx, n_sites_mapped = build_site_mapping(meta_full)

    ssm_cfg = cfg_dict.get("proto_ssm", {})
    n_sites_cfg = ssm_cfg.get("n_sites", 20)
    site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)

    # Per-file family labels
    file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
    for fi in range(len(file_list)):
        active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
        for ci in active_classes:
            file_families[fi, class_to_family[ci]] = 1.0

    # File groups for GroupKFold
    file_groups = np.array([
        f.split("_")[3] if len(f.split("_")) > 3 else f for f in file_list
    ])
    print(f"  Files: {len(file_list)}, Unique groups: {len(set(file_groups))}")
    timer.stage_end()

    # ── 6. ProtoSSM OOF cross-validation ─────────────────────────────────
    timer.stage_start("proto_ssm_oof")
    train_cfg = cfg_dict.get("proto_ssm_train", {})
    if args.n_splits is not None:
        train_cfg["oof_n_splits"] = args.n_splits

    tta_shifts = tuple(cfg_dict.get("tta_shifts", [0]))

    print(f"\n[eval] Running ProtoSSM OOF ({train_cfg.get('oof_n_splits', 3)}-fold)...")
    oof_proto_preds, fold_histories, fold_alphas = run_proto_ssm_oof(
        emb_files, logits_files, labels_files,
        site_ids_all, hours_all,
        file_families, file_groups,
        n_families, class_to_family,
        n_classes=N_CLASSES,
        ssm_cfg=ssm_cfg,
        train_cfg=train_cfg,
        tta_shifts=tta_shifts,
        device=DEVICE,
        verbose=True,
    )

    oof_proto_flat = oof_proto_preds.reshape(-1, N_CLASSES)
    y_flat = labels_files.reshape(-1, N_CLASSES).astype(np.float32)
    overall_oof_auc_proto = macro_auc_skip_empty(y_flat, oof_proto_flat)
    print(f"\n  ProtoSSM OOF macro AUC: {overall_oof_auc_proto:.4f}")

    # Per-fold AUC
    n_splits_actual = train_cfg.get("oof_n_splits", 3)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits_actual)
    n_unique = len(set(file_groups))
    if n_unique < n_splits_actual:
        n_splits_actual = n_unique
        gkf = GroupKFold(n_splits=n_splits_actual)

    fold_aucs_proto = []
    for fold_i, (_, val_idx) in enumerate(gkf.split(np.zeros(len(file_list)), groups=file_groups)):
        val_preds = oof_proto_preds[val_idx].reshape(-1, N_CLASSES)
        val_truth = labels_files[val_idx].reshape(-1, N_CLASSES).astype(np.float32)
        try:
            fold_auc = macro_auc_skip_empty(val_truth, val_preds)
            fold_aucs_proto.append(fold_auc)
            print(f"    Fold {fold_i+1}: AUC={fold_auc:.4f} (n_files={len(val_idx)})")
        except Exception:
            fold_aucs_proto.append(0.0)
    timer.stage_end()

    # ── 7. MLP probe OOF ─────────────────────────────────────────────────
    timer.stage_start("mlp_probe_oof")
    probe_cfg = cfg_dict.get("probe", {})
    mlp_sub = probe_cfg.get("mlp", {})

    # Build fuse_kwargs with fusion params for probe OOF
    fuse_kw_probes = dict(fuse_kwargs)
    fuse_kw_probes["lambda_event"] = fusion_cfg.get("lambda_event", 0.4)
    fuse_kw_probes["lambda_texture"] = fusion_cfg.get("lambda_texture", 1.0)
    fuse_kw_probes["lambda_proxy_texture"] = fusion_cfg.get("lambda_proxy_texture", 0.8)
    fuse_kw_probes["smooth_texture"] = fusion_cfg.get("smooth_texture", 0.35)
    fuse_kw_probes["smooth_event"] = fusion_cfg.get("smooth_event", 0.15)

    mlp_params = {
        "hidden_layer_sizes": tuple(mlp_sub.get("hidden_layer_sizes", [128])),
        "activation": mlp_sub.get("activation", "relu"),
        "max_iter": mlp_sub.get("max_iter", 100),
        "early_stopping": mlp_sub.get("early_stopping", True),
        "validation_fraction": mlp_sub.get("validation_fraction", 0.15),
        "n_iter_no_change": mlp_sub.get("n_iter_no_change", 10),
        "random_state": 42,
        "learning_rate_init": mlp_sub.get("learning_rate_init", 0.001),
        "alpha": mlp_sub.get("l2_alpha", 0.01),
    }

    probe_backend = probe_cfg.get("backend", "mlp")

    print(f"\n[eval] Running MLP probe OOF (backend={probe_backend})...")
    probe_result = run_oof_embedding_probe(
        scores_raw=scores_full_raw,
        emb=emb_full,
        meta_df=meta_full,
        y_true=Y_FULL,
        sc_clean=sc_clean,
        Y_SC=Y_SC,
        fuse_kwargs=fuse_kw_probes,
        pca_dim=int(probe_cfg.get("pca_dim", 64)),
        min_pos=int(probe_cfg.get("min_pos", 8)),
        C=float(probe_cfg.get("C", 0.50)),
        alpha=float(probe_cfg.get("alpha", 0.40)),
        probe_backend=probe_backend,
        mlp_params=mlp_params,
        verbose=True,
    )

    print(f"  Probe OOF base AUC:  {probe_result['score_base']:.6f}")
    print(f"  Probe OOF final AUC: {probe_result['score_final']:.6f}")
    print(f"  Delta: {probe_result['score_final'] - probe_result['score_base']:+.6f}")
    print(f"  Modeled classes: {int((probe_result['modeled_counts'] > 0).sum())}")
    timer.stage_end()

    # ── 8. Ensemble weight optimization ──────────────────────────────────
    timer.stage_start("ensemble_opt")
    oof_mlp_flat = probe_result["oof_final"]

    print("\n[eval] Optimizing ensemble weight (ProtoSSM vs MLP)...")
    best_w, best_auc, weight_results = optimize_ensemble_weight(
        oof_proto_flat, oof_mlp_flat, y_flat,
    )

    mlp_only_auc = macro_auc_skip_empty(y_flat, oof_mlp_flat)
    proto_only_auc = overall_oof_auc_proto

    print(f"\n  === Ensemble Results ===")
    print(f"  ProtoSSM-only AUC:  {proto_only_auc:.4f}")
    print(f"  MLP-only AUC:       {mlp_only_auc:.4f}")
    print(f"  Best ensemble AUC:  {best_auc:.4f}")
    print(f"  Best ProtoSSM weight: {best_w:.2f}")
    print()

    for w, auc in weight_results:
        marker = " <-- best" if abs(w - best_w) < 0.01 else ""
        print(f"    w={w:.2f}: AUC={auc:.4f}{marker}")
    timer.stage_end()

    # ── 9. Save results ──────────────────────────────────────────────────
    results = {
        "baseline": {
            "raw_auc": float(raw_auc),
            "oof_prior_auc": float(baseline_oof_auc),
        },
        "proto_ssm": {
            "overall_auc": float(overall_oof_auc_proto),
            "fold_aucs": [float(a) for a in fold_aucs_proto],
        },
        "mlp_probe": {
            "base_auc": float(probe_result["score_base"]),
            "final_auc": float(probe_result["score_final"]),
            "n_modeled_classes": int((probe_result["modeled_counts"] > 0).sum()),
        },
        "ensemble": {
            "best_weight_proto": float(best_w),
            "best_auc": float(best_auc),
            "proto_only_auc": float(proto_only_auc),
            "mlp_only_auc": float(mlp_only_auc),
            "weight_curve": [(float(w), float(a)) for w, a in weight_results],
        },
        "config": {
            "n_splits": train_cfg.get("oof_n_splits", 3),
            "n_files": len(file_list),
            "n_classes": N_CLASSES,
            "tta_shifts": list(tta_shifts),
        },
        "timing": timer.report(),
    }

    results_path = output_dir / "oof_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[eval] Results saved to {results_path}")

    # ── Baseline comparison ─────────────────────────────────────────────
    baselines_path = Path(__file__).parent.parent / "configs" / "baselines.yaml"
    best_lb = None
    if baselines_path.exists():
        import yaml
        with open(baselines_path) as f:
            baselines = yaml.safe_load(f)
        best_lb = baselines.get("best_public_lb")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Raw AUC (no priors):        {raw_auc:.6f}")
    print(f"  OOF baseline (priors only): {baseline_oof_auc:.6f}")
    print(f"  ProtoSSM OOF:               {overall_oof_auc_proto:.4f}")
    print(f"  MLP probe OOF:              {mlp_only_auc:.4f}")
    print(f"  Best ensemble OOF:          {best_auc:.4f}  (w={best_w:.2f})")
    if best_lb is not None:
        delta = best_auc - best_lb
        arrow = "+" if delta >= 0 else ""
        print(f"  vs best public LB ({best_lb:.3f}): {arrow}{delta:.4f}")
    print("=" * 60)

    # Log to wandb
    tracking.log_summary({
        "raw_auc": raw_auc,
        "oof_baseline_auc": baseline_oof_auc,
        "proto_ssm_oof_auc": overall_oof_auc_proto,
        "mlp_probe_oof_auc": mlp_only_auc,
        "best_ensemble_auc": best_auc,
        "best_ensemble_weight": best_w,
        "n_splits": train_cfg.get("oof_n_splits", 3),
        "n_files": len(file_list),
    })
    tracking.finish()

    timer.print_report()


if __name__ == "__main__":
    main()
