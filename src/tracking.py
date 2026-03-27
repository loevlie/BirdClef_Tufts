"""Experiment tracking via Weights & Biases (optional).

Logs the experiment journal: config → OOF results → public LB score → notes.
If wandb is not installed or WANDB_MODE=disabled, all calls are no-ops.
"""

import os
from pathlib import Path

_run = None


def init(project="birdclef-2026", name=None, config=None, tags=None, notes=None):
    """Initialize a wandb run.

    Parameters
    ----------
    project : str
        wandb project name.
    name : str
        Run name (e.g. "v16-baseline" or "ssl-pretrain-v1").
    config : dict
        Full experiment config (gets logged as hyperparameters).
    tags : list[str]
        Tags for filtering (e.g. ["train", "baseline"], ["evaluate", "stratified"]).
    notes : str
        Free-text notes about this experiment.
    """
    global _run
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
        _run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            reinit=True,
        )
        return _run
    except Exception:
        return None


def log(metrics: dict, step=None):
    """Log metrics at a step (e.g. per-epoch loss)."""
    if _run is None:
        return
    _run.log(metrics, step=step)


def log_summary(metrics: dict):
    """Log summary metrics (final values shown in the wandb table)."""
    if _run is None:
        return
    for k, v in metrics.items():
        _run.summary[k] = v


def log_public_lb(score: float):
    """Log a public leaderboard score after submitting to Kaggle.

    Call this manually after you get a Kaggle score:
        uv run python -c "from src.tracking import log_public_lb_retroactive; log_public_lb_retroactive('run-name', 0.925)"
    """
    if _run is None:
        return
    _run.summary["public_lb"] = score


def log_public_lb_retroactive(run_path: str, score: float, project="birdclef-2026"):
    """Update public LB score on a past run.

    Usage:
        uv run python -c "
        from src.tracking import log_public_lb_retroactive
        log_public_lb_retroactive('username/birdclef-2026/run_id', 0.925)
        "
    """
    try:
        import wandb
        api = wandb.Api()
        run = api.run(run_path)
        run.summary["public_lb"] = score
        run.summary.update()
        print(f"Updated {run_path} with public_lb={score}")
    except Exception as e:
        print(f"Failed to update run: {e}")


def log_artifact(path, name=None, type="model"):
    """Log a file as a wandb artifact."""
    if _run is None:
        return
    import wandb
    art = wandb.Artifact(name or Path(path).stem, type=type)
    art.add_file(str(path))
    _run.log_artifact(art)


def finish():
    """Finish the active run."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None
