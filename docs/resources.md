# Useful Kaggle Links

Competition discussions, notebooks, and resources worth tracking.

## Discussions

| Link | What | Why It Matters |
|------|------|---------------|
| [Faster Perch inference](https://www.kaggle.com/competitions/birdclef-2026/discussion/685318) | Optimized Perch embedding extraction | Could significantly cut our wall time bottleneck |

## Notebooks

| Link | What | Score |
|------|------|-------|
| [ProtoSSM v5 Maximum Ensemble](https://www.kaggle.com/code/yaroslavkholmirzayev/birdclef-2026-protossm-v5-maximum-ensemble) | Similar approach to ours (Perch + ProtoSSM). Our pipeline is based on this. | 0.925 public LB |
| [Pantanal Distill BirdCLEF2026 Improvement](https://www.kaggle.com/code/pradeeshrajan/pantanal-distill-birdclef2026-improvement) | Knowledge distillation approach for BirdCLEF | 0.926 public LB |

## Datasets

### dennyloevlie/birdclef2026-pipeline-inputs

Our combined submission dataset. Only input needed besides the competition data. Contains:

| File | Source | Credit |
|------|--------|--------|
| `perch_v2.onnx` | [justinchuby/Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx) | Justin Chuby (ONNX conversion), Google (original Perch v2 model) |
| `labels.csv` | [google/bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier) | Google |
| `full_perch_arrays.npz` | Computed locally from train soundscapes | Us |
| `full_perch_meta.parquet` | Computed locally from train soundscapes | Us |
| `full_oof_meta_features.npz` | Computed locally (5-fold OOF) | Us |
| `onnxruntime-*.whl` | [PyPI](https://pypi.org/project/onnxruntime/) | Microsoft (ONNX Runtime) |
