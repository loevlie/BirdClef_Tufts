"""wandb diagnostic plots for evaluation runs. No-op if wandb inactive."""

from collections import defaultdict

import numpy as np
from src import tracking


def log_evaluation_diagnostics(
    best_blend, y_flat, primary_labels, class_name_map,
    per_class_auc, weight_results, fold_aucs, fold_alphas, fold_histories,
):
    """Log evaluation diagnostics to wandb. No-op if no active run."""
    if tracking._run is None:
        return

    try:
        import wandb
    except ImportError:
        return

    try:
        auc_vals = list(per_class_auc.values())

        # 1. Per-class AUC table — the most actionable view.
        #    Sortable by species, taxon, AUC, sample count.
        rows = []
        for label, auc in sorted(per_class_auc.items(), key=lambda x: x[1]):
            ci = primary_labels.index(label)
            rows.append([label, class_name_map.get(label, "Unknown"),
                         auc, int(y_flat[:, ci].sum())])
        tracking.log({"per_class_auc": wandb.Table(
            columns=["species", "taxon", "auc", "n_positives"], data=rows,
        )})

        # 2. Frequency vs AUC scatter — are rare species harder?
        freq_table = wandb.Table(
            columns=["species", "n_positives", "auc", "taxon"],
            data=[[r[0], r[3], r[2], r[1]] for r in rows],
        )
        tracking.log({"frequency_vs_auc": wandb.plot.scatter(
            freq_table, "n_positives", "auc", title="Class Frequency vs AUC",
        )})

        # 3. AUC by taxon — quick view of which taxa are weakest
        taxon_aucs = defaultdict(list)
        for label, auc in per_class_auc.items():
            taxon_aucs[class_name_map.get(label, "Unknown")].append(auc)
        tracking.log({"auc_by_taxon": wandb.Table(
            columns=["taxon", "mean_auc", "min_auc", "n_classes"],
            data=[[t, np.mean(a), np.min(a), len(a)]
                  for t, a in sorted(taxon_aucs.items(), key=lambda x: -np.mean(x[1]))],
        )})

        # 4. Ensemble weight sweep — shows optimal ProtoSSM/MLP blend
        sweep_table = wandb.Table(
            columns=["proto_weight", "auc"],
            data=[[float(w), float(a)] for w, a in weight_results],
        )
        tracking.log({"ensemble_weight_sweep": wandb.plot.line(
            sweep_table, "proto_weight", "auc", title="Ensemble Weight vs AUC",
        )})

        # 5. Per-fold AUC bars — check fold stability
        fold_table = wandb.Table(
            columns=["fold", "auc"],
            data=[[f"Fold {i+1}", float(a)] for i, a in enumerate(fold_aucs)],
        )
        tracking.log({"fold_aucs": wandb.plot.bar(
            fold_table, "fold", "auc", title="Per-Fold AUC",
        )})

        # Summary stats for the wandb table view
        sorted_species = sorted(per_class_auc.items(), key=lambda x: x[1])
        tracking.log_summary({
            "bottom_10_species": {s: round(a, 4) for s, a in sorted_species[:10]},
            "top_10_species": {s: round(a, 4) for s, a in sorted_species[-10:]},
            "n_classes_below_0.7": sum(1 for _, a in per_class_auc.items() if a < 0.7),
            "n_classes_above_0.9": sum(1 for _, a in per_class_auc.items() if a > 0.9),
            "median_class_auc": float(np.median(auc_vals)),
        })

        print(f"  Logged {len(per_class_auc)} per-class AUCs + 5 charts to wandb")

    except Exception as e:
        print(f"  wandb diagnostics error (non-fatal): {e}")
