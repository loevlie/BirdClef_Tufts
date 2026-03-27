#!/usr/bin/env python3
"""Self-supervised pretraining: masked window prediction on all soundscapes.

Step 1: Extract Perch embeddings for all 10K+ soundscapes (or load cache)
Step 2: Pretrain SSM encoder via masked window reconstruction
Step 3: Save pretrained weights for transfer to ProtoSSM

Usage:
    uv run python scripts/ssl_pretrain.py --data-dir data/competition
    uv run python scripts/ssl_pretrain.py --data-dir data/competition --n-epochs 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.constants import N_WINDOWS
from src.data.parsing import parse_soundscape_filename
from src.data.sites import build_site_mapping
from src.timer.wallclock import WallTimer
from src.training.ssl_pretrain import ssl_pretrain


def main():
    p = argparse.ArgumentParser(description="SSL pretraining on all soundscapes")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data-dir", default="data/competition")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--output-dir", default="outputs/ssl")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--name", default=None, help="wandb run name")
    p.add_argument("--notes", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    timer = WallTimer(budget_seconds=36000)  # 10 hour budget for SSL
    timer.stage_start("setup")

    cfg = load_config(args.config)
    cfg_dict = cfg.to_dict()
    ssm_cfg = cfg_dict.get("proto_ssm", {})

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # wandb
    import os
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    from src import tracking
    tracking.init(
        name=args.name or "ssl-pretrain",
        config={**cfg_dict, "ssl_epochs": args.n_epochs, "ssl_lr": args.lr,
                "ssl_batch_size": args.batch_size, "ssl_mask_ratio": args.mask_ratio},
        tags=["ssl", "pretrain"],
        notes=args.notes,
    )

    # Load cached embeddings
    emb_path = cache_dir / "all_soundscapes_emb.npz"
    meta_path = cache_dir / "all_soundscapes_meta.parquet"

    if not emb_path.exists():
        print(f"ERROR: {emb_path} not found. Run extract_all_embeddings.py first.")
        sys.exit(1)

    print("[ssl] Loading embeddings...")
    import pandas as pd
    arr = np.load(emb_path)
    embeddings = arr["embeddings"]
    meta_df = pd.read_parquet(meta_path)
    n_windows_total = len(embeddings)
    n_files = n_windows_total // N_WINDOWS
    print(f"  {n_files} files, {n_windows_total} windows, emb dim={embeddings.shape[1]}")

    # Reshape to file-level (n_files, 12, 1536)
    emb_files = embeddings.reshape(n_files, N_WINDOWS, -1)

    # Build site/hour arrays
    file_sites = meta_df.groupby(meta_df.index // N_WINDOWS)["site"].first().values
    file_hours = meta_df.groupby(meta_df.index // N_WINDOWS)["hour_utc"].first().values.astype(int)

    # Map sites to indices
    site_to_idx = {s: i + 1 for i, s in enumerate(sorted(set(file_sites)))}
    site_ids = np.array([site_to_idx.get(s, 0) for s in file_sites], dtype=np.int64)
    site_ids = np.clip(site_ids, 0, ssm_cfg.get("n_sites", 20) - 1)
    hours = file_hours % 24

    timer.stage_end()
    print(f"  Sites: {len(site_to_idx)}, Setup time: {timer.stages.get('setup', 0):.1f}s")

    # Pretrain
    timer.stage_start("ssl_pretrain")
    print(f"\n[ssl] Starting masked window pretraining ({args.n_epochs} epochs, {n_files} files)...")

    model, history = ssl_pretrain(
        emb_files, site_ids, hours,
        ssm_cfg=ssm_cfg,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        verbose=True,
    )
    timer.stage_end()

    # Log training curves to wandb
    for ep, loss in enumerate(history["train_loss"]):
        tracking.log({"ssl/train_loss": loss}, step=ep)

    # Save pretrained weights
    save_path = output_dir / "ssl_encoder.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n[ssl] Saved pretrained encoder to {save_path}")
    print(f"  Final loss: {history['train_loss'][-1]:.6f}")

    # Save config
    ssl_config = {
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "n_files": n_files,
        "final_loss": history["train_loss"][-1],
        "ssm_cfg": ssm_cfg,
        "timing": timer.report(),
    }
    with open(output_dir / "ssl_config.json", "w") as f:
        json.dump(ssl_config, f, indent=2, default=str)

    tracking.log_summary({
        "ssl_final_loss": history["train_loss"][-1],
        "ssl_n_files": n_files,
        "ssl_n_epochs": args.n_epochs,
    })
    tracking.finish()

    timer.print_report()
    print(f"\n[ssl] Done. Use --ssl-weights {save_path} with train.py to transfer.")


if __name__ == "__main__":
    main()
