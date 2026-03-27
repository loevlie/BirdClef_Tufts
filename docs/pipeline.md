# Pipeline Stages

The pipeline runs as a sequential chain of stages. In submit mode, all stages execute within a 540s wall-clock budget managed by `WallTimer`. In train mode, there is no budget constraint.

```
Competition CSVs --> Labels --> Perch Cache --> ProtoSSM Train --> Inference --> Probes --> Fusion --> Submission CSV
```

## 1. Load Competition Data

**Function:** `pipeline.load_competition_data(data_dir)`

| | |
|---|---|
| Input | `taxonomy.csv`, `sample_submission.csv`, `train_soundscapes_labels.csv` |
| Output | `taxonomy` DataFrame, `PRIMARY_LABELS` list (234 species), `N_CLASSES` int |

## 2. Prepare Labels

**Function:** `pipeline.prepare_labels(soundscape_labels, PRIMARY_LABELS)`

| | |
|---|---|
| Input | Soundscape labels DataFrame, primary label list |
| Output | Deduplicated `sc_clean` DataFrame, multi-hot `Y_SC` array `(N_rows, 234)`, list of fully-labeled files (all 12 windows present) |

Parses filenames to extract site/hour metadata. Identifies fully-labeled files (exactly 12 windows) for training.

## 3. Build Perch Mapping

**Function:** `pipeline.build_perch_mapping(taxonomy, model_dir, PRIMARY_LABELS, Y_SC, label_to_idx)`

| | |
|---|---|
| Input | Taxonomy, Perch model directory, label arrays |
| Output | `BC_INDICES` mapping array, `MAPPED_MASK`, genus proxy mappings, `fuse_kwargs` dict, class index groups (active/inactive, texture/event, mapped/unmapped) |

Maps competition species to Perch's BirdClassifier label space. Species without a direct match get genus-level proxy mappings. Classes are partitioned into groups (mapped/unmapped, texture/event, active/inactive) that drive downstream fusion logic.

## 4. Load or Compute Perch Cache

**Function:** `pipeline.load_or_compute_cache(...)`

| | |
|---|---|
| Input | Fully-labeled file list, mapping dict, optional `infer_fn` |
| Output | `meta_full` DataFrame, `scores_full_raw` array, `emb_full` array `(N_files*12, 1536)` |

Checks for cached `.parquet` + `.npz` files in `cache_dir` or `cache_input_dir`. If no cache exists and an `infer_fn` is provided, runs Perch inference and writes the cache. In submit mode with `require_full_cache_in_submit: true`, fails fast if cache is missing.

## 5. Align Truth to Cache

**Function:** `pipeline.align_truth_to_cache(full_truth, Y_SC, meta_full)`

| | |
|---|---|
| Input | Ground truth DataFrame, label matrix, cached metadata |
| Output | `Y_FULL` array aligned to cache row order |

Reorders ground truth rows to match the cached metadata ordering. Asserts filename and row_id alignment.

## 6. OOF Meta-Features

**Function:** `pipeline.load_or_compute_oof_meta(...)`

| | |
|---|---|
| Input | Raw Perch scores, metadata, labels, fusion config |
| Output | `oof_base`, `oof_prior` arrays, `oof_fold_id` per-row fold assignments |

Builds honest out-of-fold base and prior meta-features via `build_oof_base_prior`. Results are cached to `full_oof_meta_features.npz`.

## 7. ProtoSSM Training

**Module:** `src/training/trainer.py`

| | |
|---|---|
| Input | Perch embeddings, logits, multi-hot labels, site/hour metadata |
| Output | Trained `ProtoSSMv2` model weights |

Trains the selective SSM on cached Perch embeddings. Uses AdamW + SWA, focal loss + distillation loss + prototype margin loss, mixup augmentation, and early stopping.

## 8. ProtoSSM Inference

**Module:** `src/models/proto_ssm.py`

| | |
|---|---|
| Input | Test embeddings `(N_test, 12, 1536)`, Perch logits, site/hour IDs |
| Output | Per-window species scores `(N_test, 12, 234)` |

Forward pass through the trained model. Gated fusion blends SSM logits with Perch logits using learnable per-class alpha.

## 9. MLP Probes

**Module:** `src/training/probes.py`

| | |
|---|---|
| Input | PCA-reduced embeddings, per-class labels |
| Output | Per-class probe predictions blended at weight `probe.alpha` |

Trains one MLP (or logistic) probe per eligible class (classes with `>= min_pos` positive examples). Provides independent per-class scores that complement the SSM.

## 10. Residual SSM (Optional)

**Module:** `src/models/residual_ssm.py`

| | |
|---|---|
| Input | First-pass scores + embeddings |
| Output | Additive corrections blended at `residual_ssm.correction_weight` |

Second-pass correction head. Skipped if `timer.remaining() < residual_ssm_min_remaining`.

## 11. Score Fusion and Calibration

**Modules:** `src/scoring/fusion.py`, `src/scoring/calibration.py`

| | |
|---|---|
| Input | All score streams, prior tables, temperature config |
| Output | Final calibrated probabilities `(N_test * 12, 234)` |

Fuses event-level and texture-level score streams with configurable lambdas and temporal smoothing. Applies per-class temperature scaling and top-k filtering.

## 12. Submission

**Module:** `src/submission/generate.py`

| | |
|---|---|
| Input | Calibrated probabilities, `sample_submission.csv` template |
| Output | `submission.csv` |

Writes final predictions in competition format. Applies `file_level_top_k` filtering (default: keep top 2 species per file).

## Class Temperature

**Function:** `pipeline.build_class_temperatures(PRIMARY_LABELS, CLASS_NAME_MAP, temp_cfg)`

| | |
|---|---|
| Input | Label list, class name mapping, temperature config |
| Output | Per-class temperature vector `(234,)` |

Assigns `temperature.aves` (1.10) to bird classes and `temperature.texture` (0.95) to texture taxa (Amphibia, Insecta).
