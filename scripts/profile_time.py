#!/usr/bin/env python3
"""Profile wall time of the pipeline with mock test data."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.timer.wallclock import WallTimer


def main():
    parser = argparse.ArgumentParser(description="Profile pipeline wall time")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mock-test-files", type=int, default=200)
    parser.add_argument("--budget", type=float, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    budget = args.budget or cfg.get("timer", {}).get("budget_seconds", 540.0)
    timer = WallTimer(budget_seconds=budget)

    print(f"Profiling with {args.mock_test_files} mock test files, budget={budget}s\n")

    # Stage 1: Imports
    timer.stage_start("imports")
    import numpy as np
    import torch
    timer.stage_end()

    # Stage 2: Model instantiation
    timer.stage_start("model_init")
    from src.models.proto_ssm import ProtoSSMv2
    ssm_cfg = cfg.get("proto_ssm", {})
    model = ProtoSSMv2(
        d_input=1536,
        d_model=ssm_cfg.get("d_model", 128),
        d_state=ssm_cfg.get("d_state", 16),
        n_ssm_layers=ssm_cfg.get("n_ssm_layers", 2),
        n_classes=234,
        n_windows=12,
        dropout=ssm_cfg.get("dropout", 0.15),
        n_sites=ssm_cfg.get("n_sites", 20),
        meta_dim=ssm_cfg.get("meta_dim", 16),
        use_cross_attn=ssm_cfg.get("use_cross_attn", True),
        cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
    )
    timer.stage_end()

    # Stage 3: Mock training
    timer.stage_start("proto_ssm_train")
    n_files = 60
    emb = np.random.randn(n_files, 12, 1536).astype(np.float32)
    logits = np.random.randn(n_files, 12, 234).astype(np.float32)
    labels = (np.random.rand(n_files, 12, 234) > 0.95).astype(np.float32)

    from src.training.trainer import train_proto_ssm_single
    train_cfg = cfg.get("proto_ssm_train", {})
    train_cfg["n_epochs"] = min(train_cfg.get("n_epochs", 35), 35)

    model.init_prototypes_from_data(
        torch.tensor(emb.reshape(-1, 1536)),
        torch.tensor(labels.reshape(-1, 234)),
    )

    model, _ = train_proto_ssm_single(
        model, emb, logits, labels,
        site_ids_train=np.zeros(n_files, dtype=np.int64),
        hours_train=np.zeros(n_files, dtype=np.int64),
        cfg=train_cfg,
        verbose=False,
    )
    timer.stage_end()

    # Stage 4: Mock test inference (ProtoSSM forward pass)
    timer.stage_start("proto_ssm_inference")
    n_test = args.mock_test_files
    test_emb = torch.randn(n_test, 12, 1536)
    test_logits = torch.randn(n_test, 12, 234)
    model.eval()
    with torch.no_grad():
        out, _, _ = model(
            test_emb, test_logits,
            site_ids=torch.zeros(n_test, dtype=torch.long),
            hours=torch.zeros(n_test, dtype=torch.long),
        )
    timer.stage_end()

    # Stage 5: Mock MLP probes
    timer.stage_start("mlp_probes")
    from sklearn.neural_network import MLPClassifier
    n_probes = 100  # typical number of eligible classes
    for _ in range(n_probes):
        X = np.random.randn(720, 77).astype(np.float32)
        y = (np.random.rand(720) > 0.95).astype(int)
        if y.sum() > 0:
            clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=10, random_state=42)
            clf.fit(X, y)
    timer.stage_end()

    # Stage 6: Score fusion
    timer.stage_start("score_fusion")
    scores = np.random.randn(n_test * 12, 234).astype(np.float32)
    from src.scoring.calibration import apply_temperature_and_scale
    temps = np.ones(234, dtype=np.float32) * 1.1
    probs = apply_temperature_and_scale(scores, temps, n_windows=12, top_k=2)
    timer.stage_end()

    print()
    timer.print_report()


if __name__ == "__main__":
    main()
