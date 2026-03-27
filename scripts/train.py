#!/usr/bin/env python3
"""Full training pipeline: ProtoSSM + MLP probes + optional ResidualSSM.

If test soundscapes exist, also runs inference and writes submission.csv.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on the import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.constants import N_WINDOWS, DEVICE
from src.data.reshape import reshape_to_files
from src.data.sites import build_site_mapping, get_file_metadata
from src.data.taxonomy import build_taxonomy_groups
from src.evaluation.features import build_class_features
from src.evaluation.metrics import macro_auc_skip_empty
from src.models.proto_ssm import ProtoSSMv2
from src.pipeline import (
    load_competition_data,
    prepare_labels,
    build_perch_mapping,
    load_or_compute_cache,
    align_truth_to_cache,
    load_or_compute_oof_meta,
    build_class_temperatures,
)
from src.scoring.calibration import apply_temperature_and_scale
from src.scoring.fusion import fuse_scores_with_tables
from src.scoring.priors import fit_prior_tables
from src.submission.generate import build_submission
from src.timer.wallclock import WallTimer
from src.training.trainer import train_proto_ssm_single


def parse_args():
    p = argparse.ArgumentParser(description="BirdCLEF 2026 — full training pipeline")
    p.add_argument("--config", default="configs/base.yaml", help="Path to YAML config file")
    p.add_argument("--data-dir", default=None, help="Override paths.data_dir")
    p.add_argument("--model-dir", default=None, help="Override paths.model_dir (Perch)")
    p.add_argument("--cache-dir", default=None, help="Override paths.cache_dir")
    p.add_argument("--output-dir", default="outputs", help="Directory for saved models and logs")
    p.add_argument("--run-name", default=None, help="Run name (defaults to timestamp)")
    p.add_argument("--notes", default=None, help="wandb experiment notes")
    p.add_argument("--tags", nargs="*", default=None, help="wandb tags")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return p.parse_args()


def main():
    args = parse_args()
    timer = WallTimer(budget_seconds=3600.0)  # generous budget for local runs
    timer.stage_start("setup")

    # ── Load config ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg_dict = cfg.to_dict()

    data_dir = Path(args.data_dir or cfg.paths.data_dir)
    model_dir = Path(args.model_dir or cfg.paths.model_dir)
    cache_dir = Path(args.cache_dir or cfg.paths.cache_dir)
    cache_input_dir = Path(cfg.paths.cache_input_dir)
    output_dir = Path(args.output_dir)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)
    print(f"[train] Run directory: {run_dir}")

    # ── wandb tracking (optional) ─────────────────────────────────────
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    from src import tracking
    tracking.init(
        name=run_name,
        config=cfg_dict,
        tags=args.tags or ["train"],
        notes=args.notes,
    )

    # ── Validate data exists ─────────────────────────────────────────────
    for required in ["taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"]:
        if not (data_dir / required).exists():
            print(f"ERROR: Required file not found: {data_dir / required}")
            sys.exit(1)

    # ── 1. Load competition data ─────────────────────────────────────────
    print("[train] Loading competition data...")
    taxonomy, sample_sub, soundscape_labels, PRIMARY_LABELS, N_CLASSES = (
        load_competition_data(data_dir)
    )
    print(f"  Classes: {N_CLASSES}")

    # ── 2. Prepare labels ────────────────────────────────────────────────
    print("[train] Preparing labels...")
    sc_clean, Y_SC, full_files, full_truth, Y_FULL_TRUTH, label_to_idx = (
        prepare_labels(soundscape_labels, PRIMARY_LABELS)
    )
    print(f"  Full files: {len(full_files)}, Trusted windows: {len(full_truth)}")

    # ── 3. Build Perch mapping ───────────────────────────────────────────
    print("[train] Building Perch mapping...")
    mapping = build_perch_mapping(taxonomy, model_dir, PRIMARY_LABELS, Y_SC, label_to_idx)
    print(f"  Mapped: {mapping['MAPPED_MASK'].sum()}, Unmapped: {(~mapping['MAPPED_MASK']).sum()}")
    print(f"  Proxy targets: {len(mapping['SELECTED_PROXY_TARGETS'])}")

    # ── 4. Load TF model (if needed) ────────────────────────────────────
    infer_fn = None
    try:
        import tensorflow as tf
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        birdclassifier = tf.saved_model.load(str(model_dir))
        infer_fn = birdclassifier.signatures["serving_default"]
        print("[train] Loaded TF Perch model.")
    except Exception as e:
        print(f"[train] TF not available ({e}); will require cache.")

    # ── 5. Load or compute Perch cache ───────────────────────────────────
    print("[train] Resolving Perch cache...")
    pipeline_cfg = {
        "proxy_reduce": cfg_dict.get("pipeline", {}).get("proxy_reduce", "max"),
        "batch_files": cfg_dict.get("pipeline", {}).get("batch_files", 32),
        "verbose": cfg_dict.get("pipeline", {}).get("verbose", True),
        "mode": "train",
    }
    meta_full, scores_full_raw, emb_full = load_or_compute_cache(
        full_files, data_dir, cache_dir, cache_input_dir, mapping, pipeline_cfg, infer_fn
    )
    print(f"  meta_full: {meta_full.shape}, scores: {scores_full_raw.shape}, emb: {emb_full.shape}")

    # ── 6. Align truth ───────────────────────────────────────────────────
    Y_FULL = align_truth_to_cache(full_truth, Y_SC, meta_full)
    print(f"  Y_FULL: {Y_FULL.shape}, Active classes: {int((Y_FULL.sum(axis=0) > 0).sum())}")
    timer.stage_end()

    # ── 7. OOF base/prior meta-features ──────────────────────────────────
    timer.stage_start("oof_meta")
    fusion_cfg = cfg_dict.get("best_fusion", cfg_dict.get("fusion", {}))
    fuse_kwargs = mapping["fuse_kwargs"]

    oof_base, oof_prior, oof_fold_id = load_or_compute_oof_meta(
        scores_full_raw, meta_full, sc_clean, Y_SC,
        cache_dir, fuse_kwargs, fusion_cfg,
        n_splits=5, verbose=True,
    )

    baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)
    print(f"  OOF baseline AUC: {baseline_oof_auc:.6f}")
    timer.stage_end()

    # ── 8. Fit final prior tables + embedding PCA ────────────────────────
    timer.stage_start("prior_and_pca")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

    probe_cfg = cfg_dict.get("probe", {})
    pca_dim = int(probe_cfg.get("pca_dim", 64))

    emb_scaler = StandardScaler()
    emb_full_scaled = emb_scaler.fit_transform(emb_full)

    n_comp = min(pca_dim, emb_full_scaled.shape[0] - 1, emb_full_scaled.shape[1])
    emb_pca = PCA(n_components=n_comp)
    Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)
    print(f"  PCA: {emb_full.shape[1]} -> {Z_FULL.shape[1]} (var ratio sum: {emb_pca.explained_variance_ratio_.sum():.3f})")
    timer.stage_end()

    # ── 9. Reshape to file-level ─────────────────────────────────────────
    timer.stage_start("reshape")
    emb_files, file_list = reshape_to_files(emb_full, meta_full)
    logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
    labels_files, _ = reshape_to_files(Y_FULL, meta_full)
    print(f"  Files: {len(file_list)}, emb_files: {emb_files.shape}")

    # Build taxonomy groups and site mapping
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
    timer.stage_end()

    # ── 10. Train final ProtoSSM on ALL data ─────────────────────────────
    timer.stage_start("proto_ssm_train")
    import torch

    train_cfg = cfg_dict.get("proto_ssm_train", {})

    model = ProtoSSMv2(
        d_input=emb_full.shape[1],
        d_model=ssm_cfg.get("d_model", 128),
        d_state=ssm_cfg.get("d_state", 16),
        n_ssm_layers=ssm_cfg.get("n_ssm_layers", 2),
        n_classes=N_CLASSES,
        n_windows=N_WINDOWS,
        dropout=ssm_cfg.get("dropout", 0.15),
        n_sites=n_sites_cfg,
        meta_dim=ssm_cfg.get("meta_dim", 16),
        use_cross_attn=ssm_cfg.get("use_cross_attn", True),
        cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
    ).to(DEVICE)

    # Initialize prototypes from data
    emb_flat_tensor = torch.tensor(emb_full, dtype=torch.float32)
    labels_flat_tensor = torch.tensor(Y_FULL, dtype=torch.float32)
    model.init_prototypes_from_data(emb_flat_tensor, labels_flat_tensor)
    model.init_family_head(n_families, class_to_family)
    print(f"  ProtoSSM parameters: {model.count_parameters():,}")

    model, train_history = train_proto_ssm_single(
        model,
        emb_files, logits_files, labels_files.astype(np.float32),
        site_ids_train=site_ids_all, hours_train=hours_all,
        file_families_train=file_families,
        cfg=train_cfg, verbose=True,
    )
    timer.stage_end()

    # Log training curves
    for epoch_i, (tl, vl) in enumerate(zip(
        train_history.get("train_loss", []),
        train_history.get("val_loss", []),
    )):
        tracking.log({"train/loss": tl, "train/val_loss": vl}, step=epoch_i)
    tracking.log({"train/best_val_loss": min(train_history.get("val_loss", [float("inf")]))})

    # ── 11. Train MLP probes ─────────────────────────────────────────────
    timer.stage_start("mlp_probes")
    from sklearn.neural_network import MLPClassifier
    from tqdm.auto import tqdm

    min_pos = int(probe_cfg.get("min_pos", 8))
    alpha_probe = float(probe_cfg.get("alpha", 0.40))

    # Build MLP params from config
    mlp_sub = probe_cfg.get("mlp", {})
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

    PROBE_CLASS_IDX = np.where(Y_FULL.sum(axis=0) >= min_pos)[0].astype(np.int32)
    probe_models = {}

    for cls_idx in tqdm(PROBE_CLASS_IDX, desc="Training MLP probes"):
        y = Y_FULL[:, cls_idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        X_cls = build_class_features(
            Z_FULL,
            raw_col=scores_full_raw[:, cls_idx],
            prior_col=oof_prior[:, cls_idx],
            base_col=oof_base[:, cls_idx],
        )
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > n_pos:
            repeat = max(1, n_neg // n_pos)
            pos_idx = np.where(y == 1)[0]
            X_bal = np.vstack([X_cls, np.tile(X_cls[pos_idx], (repeat, 1))])
            y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat, dtype=y.dtype)])
        else:
            X_bal, y_bal = X_cls, y
        clf = MLPClassifier(**mlp_params)
        clf.fit(X_bal, y_bal)
        probe_models[cls_idx] = clf

    print(f"  MLP probes trained: {len(probe_models)}")
    timer.stage_end()

    # ── 12. (Optional) Train ResidualSSM ─────────────────────────────────
    timer.stage_start("residual_ssm")
    res_cfg = cfg_dict.get("residual_ssm", {})
    timer_cfg = cfg_dict.get("timer", {})
    res_min_remaining = timer_cfg.get("residual_ssm_min_remaining", 240.0)
    res_model = None
    CORRECTION_WEIGHT = 0.0

    if not timer.should_skip("residual_ssm", res_min_remaining):
        try:
            from src.models.ssm import SelectiveSSM
            import torch.nn as nn
            import torch.nn.functional as F

            # Compute first-pass scores
            model.eval()
            with torch.no_grad():
                proto_train_out, _, _ = model(
                    torch.tensor(emb_files, dtype=torch.float32),
                    torch.tensor(logits_files, dtype=torch.float32),
                    site_ids=torch.tensor(site_ids_all, dtype=torch.long),
                    hours=torch.tensor(hours_all, dtype=torch.long),
                )
                proto_train_scores = proto_train_out.numpy()

            # MLP probe scores on training data
            fuse_kw_full = dict(fuse_kwargs)
            fuse_kw_full["lambda_event"] = fusion_cfg.get("lambda_event", 0.4)
            fuse_kw_full["lambda_texture"] = fusion_cfg.get("lambda_texture", 1.0)
            fuse_kw_full["lambda_proxy_texture"] = fusion_cfg.get("lambda_proxy_texture", 0.8)
            fuse_kw_full["smooth_texture"] = fusion_cfg.get("smooth_texture", 0.35)
            fuse_kw_full["smooth_event"] = fusion_cfg.get("smooth_event", 0.15)

            train_base_scores, train_prior_scores = fuse_scores_with_tables(
                scores_full_raw,
                sites=meta_full["site"].to_numpy(),
                hours=meta_full["hour_utc"].to_numpy(),
                tables=final_prior_tables,
                **fuse_kw_full,
            )
            mlp_train_flat = train_base_scores.copy()
            for cls_idx, clf in probe_models.items():
                X_cls = build_class_features(
                    Z_FULL,
                    raw_col=scores_full_raw[:, cls_idx],
                    prior_col=train_prior_scores[:, cls_idx],
                    base_col=train_base_scores[:, cls_idx],
                )
                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
                    pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
                else:
                    pred = clf.decision_function(X_cls).astype(np.float32)
                mlp_train_flat[:, cls_idx] = (1.0 - alpha_probe) * train_base_scores[:, cls_idx] + alpha_probe * pred

            mlp_train_files, _ = reshape_to_files(mlp_train_flat, meta_full)

            # Default ensemble weight (would be OOF-optimized in evaluate.py)
            ENSEMBLE_WEIGHT_PROTO = 0.5
            first_pass_files = (
                ENSEMBLE_WEIGHT_PROTO * proto_train_scores
                + (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_train_files
            ).astype(np.float32)

            labels_float = labels_files.astype(np.float32)
            first_pass_probs = 1.0 / (1.0 + np.exp(-first_pass_files))
            residuals = labels_float - first_pass_probs

            # Import ResidualSSM (defined inline like notebook for now)
            class ResidualSSM(nn.Module):
                def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                             n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
                    super().__init__()
                    self.input_proj = nn.Sequential(
                        nn.Linear(d_input + d_scores, d_model),
                        nn.LayerNorm(d_model),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    self.site_emb = nn.Embedding(n_sites, meta_dim)
                    self.hour_emb = nn.Embedding(24, meta_dim)
                    self.meta_proj = nn.Linear(2 * meta_dim, d_model)
                    self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
                    self.ssm_fwd = SelectiveSSM(d_model, d_state)
                    self.ssm_bwd = SelectiveSSM(d_model, d_state)
                    self.ssm_merge = nn.Linear(2 * d_model, d_model)
                    self.ssm_norm = nn.LayerNorm(d_model)
                    self.ssm_drop = nn.Dropout(dropout)
                    self.output_head = nn.Linear(d_model, n_classes)
                    nn.init.zeros_(self.output_head.weight)
                    nn.init.zeros_(self.output_head.bias)

                def forward(self, emb, first_pass_scores, site_ids=None, hours=None):
                    B, T, _ = emb.shape
                    x = torch.cat([emb, first_pass_scores], dim=-1)
                    h = self.input_proj(x)
                    if site_ids is not None and hours is not None:
                        s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
                        h_e = self.hour_emb(hours.clamp(0, 23))
                        meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
                        h = h + meta.unsqueeze(1)
                    h = h + self.pos_enc[:, :T, :]
                    residual = h
                    h_f = self.ssm_fwd(h)
                    h_b = self.ssm_bwd(h.flip(1)).flip(1)
                    h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
                    h = self.ssm_drop(h)
                    h = self.ssm_norm(h + residual)
                    return self.output_head(h)

                def count_parameters(self):
                    return sum(p.numel() for p in self.parameters() if p.requires_grad)

            res_model = ResidualSSM(
                d_input=emb_full.shape[1], d_scores=N_CLASSES,
                d_model=res_cfg.get("d_model", 64), d_state=res_cfg.get("d_state", 8),
                n_classes=N_CLASSES, n_windows=N_WINDOWS,
                dropout=res_cfg.get("dropout", 0.1), n_sites=n_sites_cfg, meta_dim=8,
            ).to(DEVICE)
            print(f"  ResidualSSM parameters: {res_model.count_parameters():,}")

            # Split train/val
            n_files = len(file_list)
            n_val = max(1, int(n_files * 0.15))
            perm = torch.randperm(n_files, generator=torch.Generator().manual_seed(123))
            val_i = perm[:n_val].numpy()
            train_i = perm[n_val:].numpy()

            opt = torch.optim.AdamW(res_model.parameters(), lr=res_cfg.get("lr", 1e-3), weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=res_cfg.get("lr", 1e-3),
                epochs=res_cfg.get("n_epochs", 20), steps_per_epoch=1,
                pct_start=0.1, anneal_strategy="cos",
            )

            best_val_loss = float("inf")
            best_state = None
            wait = 0

            for epoch in range(res_cfg.get("n_epochs", 20)):
                res_model.train()
                corr = res_model(
                    torch.tensor(emb_files[train_i], dtype=torch.float32),
                    torch.tensor(first_pass_files[train_i], dtype=torch.float32),
                    site_ids=torch.tensor(site_ids_all[train_i], dtype=torch.long),
                    hours=torch.tensor(hours_all[train_i], dtype=torch.long),
                )
                loss = F.mse_loss(corr, torch.tensor(residuals[train_i], dtype=torch.float32))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
                opt.step()
                sched.step()

                res_model.eval()
                with torch.no_grad():
                    val_corr = res_model(
                        torch.tensor(emb_files[val_i], dtype=torch.float32),
                        torch.tensor(first_pass_files[val_i], dtype=torch.float32),
                        site_ids=torch.tensor(site_ids_all[val_i], dtype=torch.long),
                        hours=torch.tensor(hours_all[val_i], dtype=torch.long),
                    )
                    val_loss = F.mse_loss(val_corr, torch.tensor(residuals[val_i], dtype=torch.float32))

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = {k: v.clone() for k, v in res_model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1

                if (epoch + 1) % 10 == 0:
                    print(f"    ResidualSSM epoch {epoch+1}: train={loss.item():.6f} val={val_loss.item():.6f} wait={wait}")

                if wait >= res_cfg.get("patience", 8):
                    print(f"    ResidualSSM early stop at epoch {epoch+1}")
                    break

            if best_state is not None:
                res_model.load_state_dict(best_state)
            CORRECTION_WEIGHT = res_cfg.get("correction_weight", 0.3)
            print(f"  ResidualSSM trained. Best val MSE: {best_val_loss:.6f}")
        except Exception as e:
            print(f"  ResidualSSM skipped due to error: {e}")
            res_model = None
    else:
        print("  ResidualSSM skipped (time budget).")
    timer.stage_end()

    # ── 13. Save model weights ───────────────────────────────────────────
    timer.stage_start("save_models")
    import pickle

    torch.save(model.state_dict(), run_dir / "proto_ssm.pt")
    with open(run_dir / "probe_models.pkl", "wb") as f:
        pickle.dump(probe_models, f)
    with open(run_dir / "emb_scaler.pkl", "wb") as f:
        pickle.dump(emb_scaler, f)
    with open(run_dir / "emb_pca.pkl", "wb") as f:
        pickle.dump(emb_pca, f)
    if res_model is not None:
        torch.save(res_model.state_dict(), run_dir / "residual_ssm.pt")

    print(f"[train] Models saved to {run_dir}")
    timer.stage_end()

    # ── 14. Test inference (if test files exist) ─────────────────────────
    timer.stage_start("test_inference")
    test_dir = data_dir / "test_soundscapes"
    test_paths = sorted(test_dir.glob("*.ogg")) if test_dir.exists() else []

    dryrun_n = cfg_dict.get("pipeline", {}).get("dryrun_n_files", 20)
    if len(test_paths) == 0:
        train_snd = data_dir / "train_soundscapes"
        if train_snd.exists():
            test_paths = sorted(train_snd.glob("*.ogg"))[:dryrun_n]
            print(f"[train] No test files. Dry-run on {len(test_paths)} train soundscapes.")
        else:
            print("[train] No soundscape files available for inference. Skipping.")
            timer.stage_end()
            timer.print_report()
            return
    else:
        print(f"[train] Test files found: {len(test_paths)}")

    if infer_fn is None:
        print("[train] Cannot run test inference without TF. Skipping.")
        timer.stage_end()
        timer.print_report()
        return

    from src.inference.perch import infer_perch_with_embeddings

    meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
        test_paths,
        infer_fn=infer_fn,
        n_classes=N_CLASSES,
        mapped_pos=mapping["MAPPED_POS"],
        mapped_bc_indices=mapping["MAPPED_BC_INDICES"],
        proxy_pos_to_bc=mapping["selected_proxy_pos_to_bc"],
        batch_files=cfg_dict.get("pipeline", {}).get("batch_files", 32),
        verbose=True,
        proxy_reduce=cfg_dict.get("pipeline", {}).get("proxy_reduce", "max"),
    )

    # ProtoSSM inference
    emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
    logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)
    test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx, n_sites_cfg)

    model.eval()
    tta_shifts = cfg_dict.get("tta_shifts", [0])

    if len(tta_shifts) > 1:
        from src.training.oof import temporal_shift_tta
        proto_scores = temporal_shift_tta(
            emb_test_files, logits_test_files, model,
            test_site_ids, test_hours, shifts=tta_shifts,
        )
    else:
        with torch.no_grad():
            proto_out, _, _ = model(
                torch.tensor(emb_test_files, dtype=torch.float32),
                torch.tensor(logits_test_files, dtype=torch.float32),
                site_ids=torch.tensor(test_site_ids, dtype=torch.long),
                hours=torch.tensor(test_hours, dtype=torch.long),
            )
            proto_scores = proto_out.numpy()

    proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)

    # Prior-fused base scores
    fuse_kw_test = dict(fuse_kwargs)
    fuse_kw_test["lambda_event"] = fusion_cfg.get("lambda_event", 0.4)
    fuse_kw_test["lambda_texture"] = fusion_cfg.get("lambda_texture", 1.0)
    fuse_kw_test["lambda_proxy_texture"] = fusion_cfg.get("lambda_proxy_texture", 0.8)
    fuse_kw_test["smooth_texture"] = fusion_cfg.get("smooth_texture", 0.35)
    fuse_kw_test["smooth_event"] = fusion_cfg.get("smooth_event", 0.15)

    test_base_scores, test_prior_scores = fuse_scores_with_tables(
        scores_test_raw,
        sites=meta_test["site"].to_numpy(),
        hours=meta_test["hour_utc"].to_numpy(),
        tables=final_prior_tables,
        **fuse_kw_test,
    )

    # MLP probe scores
    emb_test_scaled = emb_scaler.transform(emb_test)
    Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)

    mlp_scores = test_base_scores.copy()
    for cls_idx, clf in probe_models.items():
        X_cls_test = build_class_features(
            Z_TEST,
            raw_col=scores_test_raw[:, cls_idx],
            prior_col=test_prior_scores[:, cls_idx],
            base_col=test_base_scores[:, cls_idx],
        )
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
            pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        else:
            pred = clf.decision_function(X_cls_test).astype(np.float32)
        mlp_scores[:, cls_idx] = (1.0 - alpha_probe) * test_base_scores[:, cls_idx] + alpha_probe * pred

    # Ensemble
    ENSEMBLE_WEIGHT_PROTO = 0.5
    final_test_scores = (
        ENSEMBLE_WEIGHT_PROTO * proto_scores_flat
        + (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
    ).astype(np.float32)

    # Residual SSM correction
    if res_model is not None and CORRECTION_WEIGHT > 0:
        first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
        res_model.eval()
        with torch.no_grad():
            test_correction = res_model(
                torch.tensor(emb_test_files, dtype=torch.float32),
                torch.tensor(first_pass_test_files, dtype=torch.float32),
                site_ids=torch.tensor(test_site_ids, dtype=torch.long),
                hours=torch.tensor(test_hours, dtype=torch.long),
            ).numpy()
        final_test_scores += CORRECTION_WEIGHT * test_correction.reshape(-1, N_CLASSES).astype(np.float32)

    # Temperature scaling and submission
    temp_cfg = cfg_dict.get("temperature", {})
    class_temperatures = build_class_temperatures(PRIMARY_LABELS, mapping["CLASS_NAME_MAP"], temp_cfg)
    top_k = cfg_dict.get("file_level_top_k", 2)

    probs = apply_temperature_and_scale(final_test_scores, class_temperatures, n_windows=N_WINDOWS, top_k=top_k)

    submission = build_submission(probs, meta_test, PRIMARY_LABELS, test_paths, n_windows=N_WINDOWS)
    sub_path = run_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Submission saved: {sub_path}  ({len(submission)} rows)")
    timer.stage_end()

    # ── Summary ──────────────────────────────────────────────────────────
    baselines_path = Path(__file__).parent.parent / "configs" / "baselines.yaml"
    if baselines_path.exists():
        import yaml
        with open(baselines_path) as f:
            baselines = yaml.safe_load(f)
        best_lb = baselines.get("best_public_lb")
        if best_lb is not None and baseline_oof_auc is not None:
            delta = baseline_oof_auc - best_lb
            arrow = "+" if delta >= 0 else ""
            print(f"\n  OOF AUC: {baseline_oof_auc:.4f}  vs best public LB ({best_lb:.3f}): {arrow}{delta:.4f}")
    timer.print_report()

    logs = {
        "baseline_oof_auc": baseline_oof_auc,
        "n_probe_models": len(probe_models),
        "n_files": len(file_list),
        "n_classes": N_CLASSES,
        "residual_ssm_trained": res_model is not None,
        "timing": timer.report(),
    }
    with open(run_dir / "train_log.json", "w") as f:
        json.dump(logs, f, indent=2, default=str)

    # Log final summary to wandb
    tracking.log_summary({
        "oof_baseline_auc": baseline_oof_auc,
        "n_probe_models": len(probe_models),
        "n_files": len(file_list),
        "residual_ssm_trained": res_model is not None,
        "wall_time_seconds": timer.elapsed(),
    })
    tracking.finish()

    print(f"[train] Done. Logs at {run_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()
