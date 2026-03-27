# 🐦 BirdCLEF 2026

> Identify 234 wildlife species from 60-second soundscape recordings in Brazil's Pantanal wetlands.

## 🚀 Quick Start

```bash
git clone git@github.com:loevlie/BirdClef_Tufts.git && cd BirdClef_Tufts
uv sync                                              # reproducible env
source .venv/bin/activate                            # activate the environment uv created
kaggle competitions download -c birdclef-2026        # download data
unzip birdclef-2026.zip -d data/competition
```

## 📋 Commands

All scripts default to `configs/base.yaml` — pass `--config` to override.

```bash
uv run python scripts/train.py --data-dir data/competition     # 🏋️ Train
uv run python scripts/evaluate.py --data-dir data/competition   # 📊 K-fold OOF evaluation
uv run python scripts/optimize.py --data-dir data/competition   # 🔍 neuropt hyperparameter search
uv run python scripts/profile_time.py                           # ⏱️ Profile wall time
uv run python scripts/export_notebook.py                        # 📦 Export submission notebook
uv run pytest                                                   # ✅ Run tests
```

## 🏗️ Pipeline

```
🎵 OGG Audio → 🧠 Perch v2 Embeddings → ⚡ ProtoSSM v4 → 🎯 Score Fusion → 📄 Submission
                                              ↑
                                    MLP Probes + Site/Hour Priors
```

## 📂 Structure

```
src/                    Modular Python package
├── models/             ProtoSSM, SelectiveSSM, ResidualSSM
├── training/           Trainer, losses, augmentation, OOF
├── inference/          Perch embeddings, audio I/O, TTA
├── scoring/            Priors, fusion, calibration
├── config/             YAML config with inheritance
└── neuropt_integration/LLM-powered hyperparameter search

configs/                YAML configs (base + experiments)
scripts/                CLI: train, evaluate, optimize, export, profile
build/                  Notebook bundler (src/ → self-contained .ipynb)
docs/                   MkDocs documentation
```

## 🧪 Experiments

Create configs that inherit from base — only override what you change:

```yaml
# configs/experiments/my_experiment.yaml
base: configs/base.yaml
proto_ssm_train:
  n_epochs: 40
  lr: 1.0e-3
```

Then: `uv run python scripts/train.py --config configs/experiments/my_experiment.yaml --data-dir data/competition`

## 🔮 neuropt

[neuropt](https://github.com/loevlie/neuropt) uses Claude to intelligently search hyperparameters. Best params get baked into submission notebooks — no API key needed on Kaggle.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/optimize.py --data-dir data/competition
```

## 📖 Docs

Full documentation at [loevlie.github.io/BirdClef_Tufts](https://loevlie.github.io/BirdClef_Tufts/)

## ⚠️ Competition Constraints

- **CPU-only**, 90 min wall time
- **No internet** during submission
- **Self-contained notebook** required → `scripts/export_notebook.py` handles this
