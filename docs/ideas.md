# 💡 Ideas for Improving Our Score

Current best: **0.925 public LB**. Here's where we think the biggest gains are.

## 🔴 High Impact

### Better use of `train_audio/` (iNat + XC recordings)
We currently only use `train_soundscapes/` for training. The `train_audio/` folder has **thousands of individual species recordings** that could be used for:
- Pre-training or fine-tuning the ProtoSSM on species-level data
- Building better per-class prototypes
- Data augmentation (mix species recordings into synthetic soundscapes)

### Smarter ensemble / stacking
The current blend is a simple linear weight between ProtoSSM and MLP probes. Ideas:
- Train a lightweight meta-learner (e.g. LightGBM) on OOF predictions
- Per-class ensemble weights instead of one global weight
- Stack predictions from multiple model variants

### More robust cross-validation
With only 59 labeled soundscape files, validation is noisy. A single split can mislead. Ideas:
- Stratified GroupKFold ensuring each fold has all sites
- Repeated K-fold (3x5-fold) and average
- Use neuropt with 3-fold CV instead of single split (already supported — just run `scripts/optimize.py`)

## 🟡 Medium Impact

### Temporal modeling improvements
The ProtoSSM processes 12 windows sequentially. Ideas:
- Longer context (overlap windows, use 2.5s stride instead of 5s)
- Hierarchical model: per-window features → file-level aggregation
- Attention over the full file rather than just local SSM context

### Better calibration
Current per-taxon temperature scaling is coarse (Aves vs texture). Ideas:
- Per-species temperature learned on OOF
- Platt scaling on OOF predictions
- Isotonic regression per class

### Pseudo-labeling
Use confident predictions on unlabeled soundscapes to expand training data:
1. Run the current model on all 10,658 `train_soundscapes/`
2. Threshold high-confidence predictions as pseudo-labels
3. Retrain with expanded labeled set

## 🟢 Quick Wins

### Tune post-processing on OOF
The fusion parameters (`lambda_event`, `smooth_texture`, etc.) were tuned once and frozen. Re-tuning on current OOF might help. Run neuropt with the post-processing params in the search space.

### Try different Perch features
- Use the full Perch logits (not just mapped species) as additional features
- Concatenate multiple Perch output layers if available
- PCA dim search (currently fixed at 64)

### Wall-time optimization
Faster inference = more room for model complexity. Profile with `scripts/profile_time.py` and identify bottlenecks. Ideas:
- Increase `batch_files` for Perch inference (test memory limits)
- Reduce MLP probe count (skip classes with very few positives)
- Cache more aggressively

## 🛠️ How to Start

1. Pick an idea, create `configs/experiments/my_idea.yaml`
2. Train: `uv run python scripts/train.py --config configs/experiments/my_idea.yaml --data-dir data/competition`
3. Evaluate: `uv run python scripts/evaluate.py --config configs/experiments/my_idea.yaml --data-dir data/competition`
4. If OOF AUC improves, export and submit: `uv run python scripts/export_notebook.py --config configs/experiments/my_idea.yaml`
