# neuropt Hyperparameter Search

[neuropt](https://github.com/loevlie/neuropt) uses Claude to intelligently search hyperparameter space.

## Setup

```bash
pip install neuropt anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run a Search

```bash
uv run python scripts/optimize.py --data-dir data/competition
```

This runs 3-fold GroupKFold CV for each configuration, saving results to `experiments/runs/`.

## Search Space

Defined in `configs/base.yaml` under `neuropt.search_space`:

| Param | Range | Default | Type |
|-------|-------|---------|------|
| lr | 5e-4 to 3e-3 | 5.5e-4 | log_uniform |
| weight_decay | 8e-4 to 5e-3 | 1e-3 | log_uniform |
| distill_weight | 0.05 to 0.25 | 0.23 | uniform |
| label_smoothing | 0.01 to 0.05 | 0.019 | uniform |
| mixup_alpha | 0.1 to 0.45 | 0.25 | uniform |
| focal_gamma | 1.0 to 3.0 | 1.15 | uniform |
| swa_start_frac | 0.6 to 0.85 | 0.68 | uniform |
| pos_weight_cap | 20 to 45 | 41.0 | uniform |

Architecture params are **LOCKED** during search.

## Use Results

Results are baked into submission notebooks at export time:

```bash
uv run python scripts/export_notebook.py --neuropt-state experiments/runs/latest/neuropt_state.json
```

No API key needed at submission time.

## Tips

- Use 3-fold CV (not single split) for reliable validation with ~60 files
- Start with narrow search around proven values
- 60 evals typically sufficient for 8 params
