# Contributing

## Branch Strategy

- `main` — Protected. Requires passing CI + one review.
- `feature/<name>` — Code changes (new module, refactor, bug fix). Short-lived.
- `exp/<name>` — Experiment branches. Can be longer-lived.
- `release/v<N>` — Created before each Kaggle submission.

## Workflow

1. Create a branch: `git checkout -b feature/my-change`
2. Make changes in `src/`
3. Add/update tests in `tests/`
4. Run `uv run pytest -k "not e2e"`
5. Export a notebook: `uv run python scripts/export_notebook.py`
6. Open a PR

## Adding a New Model

1. Create `src/models/my_model.py`
2. Add to `build/bundle.py` MODULE_GROUPS
3. Add tests in `tests/test_models.py`
4. Create an experiment config in `configs/experiments/`

## Adding a New Experiment

1. Create `configs/experiments/my_experiment.yaml` with `base: configs/base.yaml`
2. Run: `uv run python scripts/train.py --config configs/experiments/my_experiment.yaml --data-dir data/competition`
3. Results land in `experiments/runs/`
