# Credits & Acknowledgments

## Foundation

Our pipeline builds on the ProtoSSM (Prototypical State Space Model) approach introduced by:

- **[ProtoSSM v5 Maximum Ensemble](https://www.kaggle.com/code/yaroslavkholmirzayev/birdclef-2026-protossm-v5-maximum-ensemble)** by Yaroslav Kholmirzayev (0.925 public LB)
  - Perch v2 embedding extraction + score fusion pipeline
  - ProtoSSM architecture (Mamba-style SSM with cross-attention)
  - Site/hour prior probability tables
  - Per-taxon temperature scaling

## Models & Data

- **[Google Perch v2](https://www.kaggle.com/models/google/bird-vocalization-classifier)** — Pre-trained bird vocalization classifier used for embedding extraction
- **[Perch ONNX conversion](https://huggingface.co/justinchuby/Perch-onnx)** by Justin Chuby — 3x faster inference via ONNX Runtime
- **[Pantanal Distill BirdCLEF2026 Improvement](https://www.kaggle.com/code/pradeeshrajan/pantanal-distill-birdclef2026-improvement)** by Pradeesh Rajan (0.926 public LB) — Knowledge distillation reference

## Tools

- **[neuropt](https://github.com/loevlie/neuropt)** — LLM-powered hyperparameter optimization
- **[ONNX Runtime](https://onnxruntime.ai/)** by Microsoft — Fast inference engine

## Our Contributions

- Modular repo structure with YAML config system
- Site-stratified K-fold cross-validation (fixed fold imbalance)
- neuropt hyperparameter search with stratified CV (+0.036 OOF improvement)
- ONNX Perch integration for 3x faster inference
- Self-supervised pretraining via masked window prediction (in progress)
- wandb experiment tracking with diagnostic plots
- Automated notebook bundling with single-dataset Kaggle deployment
