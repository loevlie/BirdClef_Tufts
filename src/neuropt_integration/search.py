"""neuropt hyperparameter search wrapper for BirdCLEF 2026."""
import json
import time
from pathlib import Path


def run_neuropt_search(
    train_fn,
    search_space_cfg: dict,
    neuropt_cfg: dict,
    output_dir: str | Path,
    ml_context: str = "",
):
    """Run neuropt ArchSearch and save results."""
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

    # Save state after each eval
    orig_run = search._run_one
    def _save_and_run(cfg):
        r = orig_run(cfg)
        with open(state_path, "w") as f:
            json.dump({
                "best_score": search.best_score,
                "best_config": search.best_config,
                "total": search.total_experiments,
            }, f, indent=2, default=str)
        return r
    search._run_one = _save_and_run

    max_evals = neuropt_cfg.get("max_evals", 60)
    search.run(max_evals=max_evals, resume=True)

    return {
        "best_score": search.best_score,
        "best_config": search.best_config,
        "total": search.total_experiments,
        "state_path": str(state_path),
        "log_path": log_path,
    }
