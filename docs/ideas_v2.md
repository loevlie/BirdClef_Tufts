# Ideas to Beat 0.924 (No OOM)

Current best: **0.924 public LB** (d_model=128, per-class weights, v18 training params)

## Priority Queue

### 1. Pseudo-labeling (IN PROGRESS)
Run our model on all 10,658 soundscapes, threshold high-confidence predictions as labels, retrain on 10x more data. Currently training on only 59 labeled files.
- **Why it could work**: 10x more training data directly addresses overfitting
- **Risk**: noisy pseudo-labels could hurt if threshold is wrong
- **Memory**: same model, same inference

### 2. Train_audio prototypes
Extract Perch embeddings from thousands of individual species recordings. Compute per-species average embedding. Use cosine similarity to these prototypes as 234 additional MLP features.
- **Why it could work**: gives species-level signal from thousands of recordings vs 59 soundscapes
- **Memory**: just 234 extra floats per window

### 3. File-level aggregation
Train a model on file-level features (max/mean/std of 12 window predictions). Broadcast back.
- **Why it could work**: captures "species is somewhere in this file" pattern
- **Memory**: trivial

### 4. Better MLP features
Add: window position (0-11), file-level max/mean of Perch logits, delta between consecutive windows, ratio to file mean.
- **Why it could work**: more features = better probes at no cost
- **Memory**: trivial

### 5. Cross-validated ridge stacking
LogisticRegression on ProtoSSM OOF + MLP OOF + base scores. Simpler than LGBM, less overfit.
- **Why it could work**: proper meta-learning instead of fixed blend
- **Memory**: trivial

### 6. Taxon-specific ProtoSSMs
Separate small ProtoSSMs for Aves vs Insecta/Amphibia.
- **Why it could work**: different temporal patterns for different taxa
- **Memory**: same total, two smaller models

## Evaluated and Rejected
- LightGBM stacking: only +0.0007 locally
- Co-occurrence boosting: only +0.0003 locally
- d_model=320: OOM on Kaggle
- d_model=192: OOM on Kaggle
- Early stopping (epochs=40): hurt d_model=128 (0.916 vs 0.922)
- SSL pretraining: weight transfer worked but no OOF gain (may need more tuning)
