"""neuropt hyperparameter search wrapper for BirdCLEF 2026."""
import json
from pathlib import Path

from src import tracking


def run_neuropt_search(
    train_fn,
    search_space_cfg: dict,
    neuropt_cfg: dict,
    output_dir: str | Path,
    ml_context: str = "",
):
    """Run neuropt ArchSearch and save results. Logs each eval to wandb live."""
    from neuropt import ArchSearch
    from .spaces import build_search_space

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "neuropt_state.json"
    log_path = str(output_dir / "neuropt_search.jsonl")

    space = build_search_space(search_space_cfg)

    search = ArchSearch(
        train_fn=train_fn,
        search_space=space,
        backend=neuropt_cfg.get("backend", "claude"),
        log_path=log_path,
        batch_size=neuropt_cfg.get("batch_size", 3),
        minimize=False,
        ml_context=ml_context,
    )

    # Accumulate all evals for the growing table
    param_names = sorted(search_space_cfg.keys())
    all_evals = []

    # Save state + log to wandb after each eval
    orig_run = search._run_one
    eval_count = [0]

    def _save_and_log(cfg):
        r = orig_run(cfg)
        eval_count[0] += 1
        score = r.get("score", 0) if isinstance(r, dict) else 0

        # Save state JSON
        with open(state_path, "w") as f:
            json.dump({
                "best_score": search.best_score,
                "best_config": search.best_config,
                "total": search.total_experiments,
            }, f, indent=2, default=str)

        # Log scalar metrics (for time series charts)
        tracking.log({
            "neuropt/score": score,
            "neuropt/best_score": search.best_score,
        }, step=eval_count[0])

        # Accumulate eval and log as table (for scatter/parallel coords)
        all_evals.append({"score": score, **cfg})
        try:
            import wandb
            if tracking._run is not None:
                cols = ["score"] + param_names
                rows = [[e.get(c) for c in cols] for e in all_evals]
                table = wandb.Table(columns=cols, data=rows)
                tracking.log({"neuropt/search_table": table})
        except Exception:
            pass

        return r

    search._run_one = _save_and_log

    max_evals = neuropt_cfg.get("max_evals", 60)
    search.run(max_evals=max_evals, resume=True)

    return {
        "best_score": search.best_score,
        "best_config": search.best_config,
        "total": search.total_experiments,
        "state_path": str(state_path),
        "log_path": log_path,
    }
