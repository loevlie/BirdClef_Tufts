# 🐦 BirdCLEF 2026

Acoustic species identification in the Pantanal wetlands. 234 species from 60-second soundscape recordings.

## Quickstart

```bash
uv sync                                                          # install
uv run python scripts/train.py --data-dir data/competition       # train
uv run python scripts/evaluate.py --data-dir data/competition    # evaluate
uv run python scripts/export_notebook.py                         # export notebook
uv run pytest                                                    # tests
```

All scripts default to `configs/base.yaml`. Pass `--config` to use a different config.

## Architecture

The pipeline uses **Perch v2 embeddings** (Google's pre-trained bird vocalization classifier) fed into a **ProtoSSM v4** (Prototypical State Space Model) with:

- Mamba-style selective SSM for temporal modeling
- Cross-attention for non-local patterns
- Per-class prototypical learning
- Gated fusion with Perch logits
- MLP probe ensemble for per-class refinement
- Site/hour prior probability tables

## Competition Constraints

- CPU-only, 90 minute wall time
- No internet during submission
- Self-contained notebook required
