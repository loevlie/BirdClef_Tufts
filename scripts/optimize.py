#!/usr/bin/env python3
"""neuropt hyperparameter search for BirdCLEF 2026 ProtoSSM training params."""

import argparse
import copy
import gc
import json
import os
import sys
import traceback
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
from src.models.proto_ssm import ProtoSSMv2
from src.pipeline import (
    load_competition_data,
    prepare_labels,
    build_perch_mapping,
    load_or_compute_cache,
    align_truth_to_cache,
    load_or_compute_oof_meta,
)
from src.timer.wallclock import WallTimer
from src.training.trainer import train_proto_ssm_single


def parse_args():
    p = argparse.ArgumentParser(description="BirdCLEF 2026 — neuropt hyperparameter search")
    p.add_argument("--config", default="configs/base.yaml", help="Path to YAML config file")
    p.add_argument("--data-dir", default=None, help="Override paths.data_dir")
    p.add_argument("--model-dir", default=None, help="Override paths.model_dir (Perch)")
    p.add_argument("--cache-dir", default=None, help="Override paths.cache_dir")
    p.add_argument("--max-evals", type=int, default=None, help="Override neuropt max_evals")
    p.add_argument("--output-dir", default="outputs/neuropt", help="Directory for search results")
    p.add_argument("--resume", action="store_true", help="Resume from previous search state")
    return p.parse_args()


def main():
    args = parse_args()
    timer = WallTimer(budget_seconds=36000.0)  # 10h budget for search
    timer.stage_start("setup")

    # ── Require API key ──────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is required for neuropt search.")
        sys.exit(1)

    # ── Load config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg_dict = cfg.to_dict()

    data_dir = Path(args.data_dir or cfg.paths.data_dir)
    model_dir = Path(args.model_dir or cfg.paths.model_dir)
    cache_dir = Path(args.cache_dir or cfg.paths.cache_dir)
    cache_input_dir = Path(cfg.paths.cache_input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate data exists ─────────────────────────────────────────────
    for required in ["taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
        if not (data_dir / required).exists():
            print(f"ERROR: Required file not found: {data_dir / required}")
            sys.exit(1)

    # ── 1. Load and prepare data ─────────────────────────────────────────
    print("[optimize] Loading competition data...")
    taxonomy, sample_sub, soundscape_labels, PRIMARY_LABELS, N_CLASSES = (
        load_competition_data(data_dir)
    )

    print("[optimize] Preparing labels...")
    sc_clean, Y_SC, full_files, full_truth, Y_FULL_TRUTH, label_to_idx = (
        prepare_labels(soundscape_labels, PRIMARY_LABELS)
    )

    print("[optimize] Building Perch mapping...")
    mapping = build_perch_mapping(taxonomy, model_dir, PRIMARY_LABELS, Y_SC, label_to_idx)

    print("[optimize] Resolving Perch cache...")
    pipeline_cfg = {"mode": "train", "verbose": True}
    meta_full, scores_full_raw, emb_full = load_or_compute_cache(
        full_files, data_dir, cache_dir, cache_input_dir, mapping, pipeline_cfg,
    )
    Y_FULL = align_truth_to_cache(full_truth, Y_SC, meta_full)

    # ── 2. Reshape and prepare structures ────────────────────────────────
    print("[optimize] Reshaping to file-level...")
    emb_files, file_list = reshape_to_files(emb_full, meta_full)
    logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
    labels_files, _ = reshape_to_files(Y_FULL, meta_full)

    n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
    site_to_idx, n_sites_mapped = build_site_mapping(meta_full)

    ssm_cfg = cfg_dict.get("proto_ssm", {})
    n_sites_cfg = ssm_cfg.get("n_sites", 20)
    site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)

    # File groups for GroupKFold
    file_groups = np.array([
        f.split("_")[3] if len(f.split("_")) > 3 else f for f in file_list
    ])
    print(f"  Files: {len(file_list)}, Groups: {len(set(file_groups))}, Classes: {N_CLASSES}")
    timer.stage_end()

    # ── 3. Define neuropt training function ──────────────────────────────
    import torch
    from src.training.oof import site_stratified_kfold

    n_splits_cv = 3

    def neuropt_train_fn(config):
        """3-fold site-stratified CV for neuropt evaluation."""
        try:
            _cfg = copy.deepcopy(cfg_dict)
            t = _cfg.get("proto_ssm_train", {})
            for k in ["lr", "weight_decay", "distill_weight", "label_smoothing",
                       "mixup_alpha", "focal_gamma", "swa_start_frac", "pos_weight_cap"]:
                if k in config:
                    t[k] = float(config[k])

            _a = ssm_cfg  # architecture LOCKED

            splits = site_stratified_kfold(len(file_list), file_groups, n_splits=n_splits_cv)
            fold_aucs = []
            all_train_losses = []
            all_val_losses = []

            for fold, (ti, vi) in enumerate(splits):
                _m = ProtoSSMv2(
                    d_input=1536,
                    d_model=_a.get("d_model", 128),
                    d_state=_a.get("d_state", 16),
                    n_ssm_layers=_a.get("n_ssm_layers", 2),
                    n_classes=N_CLASSES,
                    n_windows=N_WINDOWS,
                    dropout=_a.get("dropout", 0.15),
                    n_sites=_a.get("n_sites", 20),
                    meta_dim=_a.get("meta_dim", 16),
                    use_cross_attn=_a.get("use_cross_attn", True),
                    cross_attn_heads=_a.get("cross_attn_heads", 4),
                )

                _m.init_prototypes_from_data(
                    torch.tensor(emb_files[ti], dtype=torch.float32).reshape(-1, 1536),
                    torch.tensor(labels_files[ti], dtype=torch.float32).reshape(-1, N_CLASSES),
                )
                _m.init_family_head(n_families, class_to_family)

                _m, _h = train_proto_ssm_single(
                    _m,
                    emb_files[ti], logits_files[ti], labels_files[ti],
                    site_ids_train=site_ids_all[ti], hours_train=hours_all[ti],
                    cfg=t, verbose=False,
                )

                _m.eval()
                with torch.no_grad():
                    _o, _, _ = _m(
                        torch.tensor(emb_files[vi], dtype=torch.float32),
                        torch.tensor(logits_files[vi], dtype=torch.float32),
                        site_ids=torch.tensor(site_ids_all[vi], dtype=torch.long),
                        hours=torch.tensor(hours_all[vi], dtype=torch.long),
                    )

                try:
                    _auc = macro_auc_skip_empty(
                        labels_files[vi].reshape(-1, N_CLASSES),
                        _o.reshape(-1, N_CLASSES).numpy(),
                    )
                    fold_aucs.append(_auc)
                except Exception:
                    pass

                if fold == 0:
                    all_train_losses = _h.get("train_loss", [])
                    all_val_losses = _h.get("val_loss", [])

                del _m
                gc.collect()

            mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
            return {
                "score": mean_auc,
                "train_losses": all_train_losses,
                "val_losses": all_val_losses,
                "val_auc": mean_auc,
                "fold_aucs": [float(a) for a in fold_aucs],
            }
        except Exception as e:
            print(f"ERROR in neuropt_train_fn: {e}")
            traceback.print_exc()
            raise

    # ── 4. Run baseline ──────────────────────────────────────────────────
    timer.stage_start("baseline")
    train_cfg = cfg_dict.get("proto_ssm_train", {})
    baseline_config = {
        "lr": train_cfg.get("lr", 5.5e-4),
        "weight_decay": train_cfg.get("weight_decay", 1e-3),
        "distill_weight": train_cfg.get("distill_weight", 0.23),
        "label_smoothing": train_cfg.get("label_smoothing", 0.019),
        "mixup_alpha": train_cfg.get("mixup_alpha", 0.25),
        "focal_gamma": train_cfg.get("focal_gamma", 1.15),
        "swa_start_frac": train_cfg.get("swa_start_frac", 0.68),
        "pos_weight_cap": train_cfg.get("pos_weight_cap", 41.0),
    }

    print(f"\n[optimize] Running {n_splits_cv}-fold baseline with current config...")
    baseline = neuropt_train_fn(baseline_config)
    baseline_score = baseline["score"]
    print(f"  BASELINE {n_splits_cv}-fold AUC: {baseline_score:.4f} (folds: {baseline.get('fold_aucs', [])})")
    print(f"  neuropt must beat {baseline_score:.4f}\n")
    timer.stage_end()

    # ── 5. Run neuropt search ────────────────────────────────────────────
    timer.stage_start("neuropt_search")
    from src.neuropt_integration.search import run_neuropt_search

    neuropt_cfg = cfg_dict.get("neuropt", {})
    if args.max_evals is not None:
        neuropt_cfg["max_evals"] = args.max_evals

    search_space_cfg = neuropt_cfg.get("search_space", {})
    if not search_space_cfg:
        # Fallback default search space
        search_space_cfg = {
            "lr": {"type": "log_uniform", "low": 5e-4, "high": 3e-3},
            "weight_decay": {"type": "log_uniform", "low": 8e-4, "high": 5e-3},
            "distill_weight": {"type": "uniform", "low": 0.05, "high": 0.25},
            "label_smoothing": {"type": "uniform", "low": 0.01, "high": 0.05},
            "mixup_alpha": {"type": "uniform", "low": 0.1, "high": 0.45},
            "focal_gamma": {"type": "uniform", "low": 1.0, "high": 3.0},
            "swa_start_frac": {"type": "uniform", "low": 0.6, "high": 0.85},
            "pos_weight_cap": {"type": "uniform", "low": 20.0, "high": 45.0},
        }

    ml_context = (
        f"BirdCLEF 2026 ProtoSSM v4. Arch LOCKED (d_model={ssm_cfg.get('d_model', 128)}, "
        f"{ssm_cfg.get('n_ssm_layers', 2)} layers). "
        f"{n_splits_cv}-FOLD CV validation (honest, no leakage). "
        f"BASELINE {n_splits_cv}-fold AUC = {baseline_score:.4f}. "
        f"Only searching 8 TRAINING params. Post-processing is FIXED. "
        f"~{len(file_list)} files, {len(file_list)*N_WINDOWS} windows, {N_CLASSES} species. "
        f"{n_splits_cv}-fold CV takes ~45s/eval. "
        f"focal_gamma: higher = more focus on hard examples. "
        f"mixup_alpha: file-level augmentation. distill_weight: Perch logit distillation. "
        f"swa_start_frac: when to start weight averaging (fraction of training). "
        f"pos_weight_cap: class imbalance cap (higher = more weight on rare species)."
    )

    results = run_neuropt_search(
        train_fn=neuropt_train_fn,
        search_space_cfg=search_space_cfg,
        neuropt_cfg=neuropt_cfg,
        output_dir=output_dir,
        ml_context=ml_context,
    )
    timer.stage_end()

    # ── 6. Report results ────────────────────────────────────────────────
    best_score = results["best_score"]
    best_config = results["best_config"]
    improvement = best_score - baseline_score

    print("\n" + "=" * 60)
    print("  NEUROPT SEARCH COMPLETE")
    print("=" * 60)
    print(f"  Baseline {n_splits_cv}-fold AUC: {baseline_score:.4f}")
    print(f"  Best {n_splits_cv}-fold AUC:     {best_score:.4f}")
    print(f"  Improvement:         {improvement:+.4f}")
    print(f"  Total evaluations:   {results['total']}")
    print(f"  Best config:")
    print(json.dumps(best_config, indent=4, default=str))
    print("=" * 60)

    # Save final summary
    summary = {
        "baseline_score": float(baseline_score),
        "baseline_config": baseline_config,
        "best_score": float(best_score),
        "best_config": best_config,
        "improvement": float(improvement),
        "total_evals": results["total"],
        "state_path": results["state_path"],
        "log_path": results["log_path"],
        "timing": timer.report(),
    }
    summary_path = output_dir / "search_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[optimize] Summary saved to {summary_path}")
    print(f"[optimize] State saved to {results['state_path']}")
    print(f"[optimize] Search log at {results['log_path']}")

    # Compare against known best public LB
    baselines_path = Path(__file__).parent.parent / "configs" / "baselines.yaml"
    if baselines_path.exists():
        import yaml
        with open(baselines_path) as f:
            baselines = yaml.safe_load(f)
        best_lb = baselines.get("best_public_lb")
        if best_lb is not None:
            delta = best_score - best_lb
            arrow = "+" if delta >= 0 else ""
            print(f"\n[optimize] vs best public LB ({best_lb:.3f}): {arrow}{delta:.4f}")

    # Apply if it beat baseline
    if improvement > 0.001:
        from src.neuropt_integration.config_apply import apply_neuropt_config
        applied_cfg = copy.deepcopy(cfg_dict)
        apply_neuropt_config(applied_cfg, best_config)
        with open(output_dir / "best_config_applied.json", "w") as f:
            json.dump(applied_cfg, f, indent=2, default=str)
        print(f"[optimize] Applied config saved to {output_dir / 'best_config_applied.json'}")
    else:
        print("[optimize] neuropt did not reliably beat baseline. Keeping original config.")

    timer.print_report()


if __name__ == "__main__":
    main()
