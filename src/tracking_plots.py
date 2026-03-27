"""wandb diagnostic plots for evaluation runs. No-op if wandb inactive."""

import numpy as np
from src import tracking


def log_evaluation_diagnostics(
    best_blend, y_flat, primary_labels, class_name_map,
    per_class_auc, weight_results, fold_aucs, fold_alphas, fold_histories,
):
    """Log all evaluation diagnostics to wandb. No-op if no active run."""
    if tracking._run is None:
        return

    try:
        import wandb
    except ImportError:
        return

    try:
        auc_vals = list(per_class_auc.values())

        # 1. Per-class AUC table (sortable)
        rows = []
        for label, auc in sorted(per_class_auc.items(), key=lambda x: x[1]):
            n_pos = int(y_flat[:, primary_labels.index(label)].sum())
            taxa = class_name_map.get(label, "Unknown")
            rows.append([label, taxa, auc, n_pos])
        tracking.log({"per_class_auc": wandb.Table(
            columns=["species", "taxon", "auc", "n_positives"], data=rows,
        )})

        # 2. Class AUC histogram
        tracking.log({"charts/class_auc_histogram": wandb.Histogram(auc_vals)})

        # 3. AUC by taxon group
        from collections import defaultdict
        taxon_aucs = defaultdict(list)
        for label, auc in per_class_auc.items():
            taxon_aucs[class_name_map.get(label, "Unknown")].append(auc)
        tracking.log({"auc_by_taxon": wandb.Table(
            columns=["taxon", "mean_auc", "n_classes"],
            data=[[t, np.mean(a), len(a)] for t, a in sorted(taxon_aucs.items(), key=lambda x: -np.mean(x[1]))],
        )})

        # 4. Class frequency vs AUC scatter
        freq_data = []
        for label, auc in per_class_auc.items():
            ci = primary_labels.index(label)
            freq_data.append([label, int(y_flat[:, ci].sum()), auc, class_name_map.get(label, "Unknown")])
        freq_table = wandb.Table(columns=["species", "n_positives", "auc", "taxon"], data=freq_data)
        tracking.log({"frequency_vs_auc": wandb.plot.scatter(
            freq_table, "n_positives", "auc", title="Class Frequency vs AUC",
        )})

        # 5. Ensemble weight sweep
        sweep_table = wandb.Table(
            columns=["proto_weight", "auc"],
            data=[[float(w), float(a)] for w, a in weight_results],
        )
        tracking.log({"ensemble_weight_sweep": wandb.plot.line(
            sweep_table, "proto_weight", "auc", title="Ensemble Weight vs AUC",
        )})

        # 6. Per-fold AUC bars
        fold_table = wandb.Table(
            columns=["fold", "auc"],
            data=[[f"Fold {i+1}", float(a)] for i, a in enumerate(fold_aucs)],
        )
        tracking.log({"fold_aucs": wandb.plot.bar(fold_table, "fold", "auc", title="Per-Fold AUC")})

        # 7. Fusion alpha distribution
        if fold_alphas:
            mean_alphas = np.stack(fold_alphas).mean(axis=0)
            tracking.log({"charts/fusion_alpha_distribution": wandb.Histogram(mean_alphas.tolist())})
            tracking.log({
                "fusion/ssm_dominant_classes": int((mean_alphas > 0.5).sum()),
                "fusion/perch_dominant_classes": int((mean_alphas <= 0.5).sum()),
            })

        # 8. Per-fold training curves
        for fi, hist in enumerate(fold_histories):
            for ep, (tl, vl) in enumerate(zip(hist.get("train_loss", []), hist.get("val_loss", []))):
                tracking.log({f"folds/fold{fi+1}_train_loss": tl, f"folds/fold{fi+1}_val_loss": vl}, step=ep)

        # 9. Score distribution
        tracking.log({"charts/score_distribution": wandb.Histogram(best_blend.flatten()[:10000].tolist())})

        # 10. Summary stats
        sorted_species = sorted(per_class_auc.items(), key=lambda x: x[1])
        tracking.log_summary({
            "bottom_10_species": {s: round(a, 4) for s, a in sorted_species[:10]},
            "top_10_species": {s: round(a, 4) for s, a in sorted_species[-10:]},
            "n_classes_below_0.7": sum(1 for _, a in per_class_auc.items() if a < 0.7),
            "n_classes_above_0.9": sum(1 for _, a in per_class_auc.items() if a > 0.9),
            "median_class_auc": float(np.median(auc_vals)),
        })

        print(f"  Logged {len(per_class_auc)} per-class AUCs + 10 diagnostic charts to wandb")

    except Exception as e:
        print(f"  wandb diagnostics error (non-fatal): {e}")
