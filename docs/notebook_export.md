# Notebook Export

The bundler in `build/bundle.py` flattens `src/` into a single self-contained Kaggle notebook. No imports from `src.*` remain in the output -- everything is inlined.

## Export a Notebook

```bash
uv run python scripts/export_notebook.py
uv run python scripts/export_notebook.py \
    --neuropt-state experiments/runs/latest/neuropt_state.json
```

Output lands in `build/output/submission.ipynb` by default. Use `-o` to override.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/base.yaml` | YAML config path |
| `--neuropt-state` | `None` | Path to `neuropt_state.json` to bake in best params |
| `-o` / `--output` | `build/output/submission.ipynb` | Output notebook path |

## Uploading to Kaggle

After exporting, go to Kaggle → New Notebook → File → Upload Notebook → select `build/output/submission.ipynb`.

Then add these inputs via the "Add Input" sidebar:

| Type | What to search | What to select |
|------|---------------|----------------|
| Competition | `birdclef-2026` | BirdCLEF+ 2026 |
| Model | `google/bird-vocalization-classifier` | **Perch** by Google → tensorflow2 / perch_v2_cpu / V1 |
| Dataset | `dennyloevlie/birdclef2026-pipeline-inputs` | BirdCLEF 2026 Pipeline Inputs |
| Notebook | `tf-wheels ashok205` | **tf_wheels** (1 upvote, by ashok205) |

The **Pipeline Inputs** dataset contains everything in one package:
- Pre-computed Perch embeddings (skips ~5 min training inference)
- ONNX Perch model (3x faster test inference)
- Perch labels.csv (species mapping)

The notebook auto-detects what's available and prints an **INPUT VERIFICATION** table at the top so you can confirm.

Then set **Internet → Off**, **Accelerator → None (CPU)**, and submit.

## How the Bundler Works

1. **TF install cell** -- pip installs TF 2.20 wheels from a Kaggle dataset (needed for Perch v2)
2. **Mode cell** -- sets `MODE = "submit"`
3. **Third-party imports** -- collects and deduplicates all `import` / `from` statements across modules (excluding `src.*` and relative imports)
4. **Baked config** -- serialises the full config dict as a JSON literal in a code cell. This is where neuropt results end up.
5. **Module groups** -- each group becomes a markdown header + code cell. Internal imports (`from src.*`, `from .`) are stripped. Module order follows `MODULE_GROUPS` in `build/bundle.py`.
6. **Validation** -- checks every code cell for leftover `from src.` / `import src.` / relative imports. Fails the export if any are found.

### Module Groups (in order)

| Group | Modules |
|-------|---------|
| Constants & Config | `constants.py` |
| Data Utilities | `parsing.py`, `taxonomy.py`, `sites.py`, `reshape.py` |
| Evaluation | `metrics.py`, `smoothing.py`, `features.py` |
| Models | `ssm.py`, `attention.py`, `proto_ssm.py`, `residual_ssm.py` |
| Training | `losses.py`, `augmentation.py`, `trainer.py`, `oof.py`, `probes.py` |
| Inference | `audio.py`, `perch.py`, `tta.py` |
| Scoring | `priors.py`, `fusion.py`, `calibration.py` |
| Submission | `generate.py` |
| Timer | `wallclock.py` |

## Baking neuropt Results

When `--neuropt-state` is provided, the export script calls `load_and_apply_best(config_dict, neuropt_state_path)` which:

1. Reads the neuropt state JSON
2. Extracts the best parameter set found during search
3. Overwrites the corresponding keys in the config dict (e.g., `proto_ssm_train.lr`, `proto_ssm_train.mixup_alpha`)
4. The modified config is then serialised into the notebook

This means the exported notebook contains the optimised hyperparameters as literal values. No neuropt dependency or API key is needed at submission time.

## Adding a New Module

1. Create the module in `src/`
2. Add it to the appropriate group in `MODULE_GROUPS` in `build/bundle.py`
3. Re-export: `uv run python scripts/export_notebook.py`
4. Verify no validation errors

If your module has internal imports (`from src.foo import bar`), they will be stripped automatically. Make sure all referenced names are defined in a module that appears earlier in the group ordering.
