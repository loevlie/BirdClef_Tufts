# Configuration Reference

All config lives in YAML files. `configs/base.yaml` is the default; experiment configs override it via `base: configs/base.yaml`.

## Top-Level

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | str | `"submit"` | `"submit"` for Kaggle inference, `"train"` for local training |

## paths

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_dir` | str | `/kaggle/input/competitions/birdclef-2026` | Competition data root |
| `model_dir` | str | `/kaggle/input/models/.../perch_v2_cpu/1` | Perch v2 model directory |
| `cache_dir` | str | `cache/` | Writable embedding cache directory |
| `cache_input_dir` | str | `/kaggle/input/perch-meta` | Read-only pre-computed cache dataset |

## pipeline

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `batch_files` | int | `32` | Audio files per Perch embedding batch |
| `proxy_reduce` | str | `"max"` | Reduction over proxy scores (`"max"` or `"mean"`) |
| `require_full_cache_in_submit` | bool | `true` | Fail fast if cache missing in submit mode |
| `verbose` | bool | `false` | Extra logging |
| `run_oof_baseline` | bool | `false` | Compute OOF baseline scores |
| `run_probe_check` | bool | `false` | Sanity-check probe predictions |
| `dryrun_n_files` | int | `20` | Files to process in dryrun/debug mode |

## proto_ssm (architecture -- LOCKED)

These are fixed and should not be tuned.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `d_model` | int | `128` | Hidden dimension |
| `d_state` | int | `16` | SSM state size |
| `n_ssm_layers` | int | `2` | Number of SSM layers |
| `dropout` | float | `0.15` | Dropout rate |
| `n_prototypes` | int | `1` | Prototypes per class |
| `n_sites` | int | `20` | Site-embedding cardinality |
| `meta_dim` | int | `16` | Metadata embedding dimension |
| `use_cross_attn` | bool | `true` | Enable cross-attention layer |
| `cross_attn_heads` | int | `4` | Number of cross-attention heads |

## proto_ssm_train (neuropt-tunable)

Parameters marked with a star are included in the neuropt search space.

| Key | Type | Default | neuropt | Description |
|-----|------|---------|---------|-------------|
| `n_epochs` | int | `35` | -- | Training epochs |
| `lr` | float | `5.5e-4` | log_uniform [5e-4, 3e-3] | Learning rate |
| `weight_decay` | float | `1e-3` | log_uniform [8e-4, 5e-3] | AdamW weight decay |
| `val_ratio` | float | `0.15` | -- | Validation holdout fraction |
| `patience` | int | `15` | -- | Early-stopping patience (epochs) |
| `pos_weight_cap` | float | `41.0` | uniform [20, 45] | Cap on positive class weight |
| `distill_weight` | float | `0.23` | uniform [0.05, 0.25] | Distillation loss weight |
| `proto_margin` | float | `0.1` | -- | Prototype margin loss |
| `label_smoothing` | float | `0.019` | uniform [0.01, 0.05] | Label smoothing factor |
| `oof_n_splits` | int | `3` | -- | OOF cross-validation folds |
| `mixup_alpha` | float | `0.25` | uniform [0.1, 0.45] | Mixup interpolation alpha |
| `focal_gamma` | float | `1.15` | uniform [1.0, 3.0] | Focal loss gamma |
| `swa_start_frac` | float | `0.68` | uniform [0.6, 0.85] | Fraction of training before SWA |
| `swa_lr` | float | `5e-4` | -- | SWA learning rate |

## fusion

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lambda_event` | float | `0.4` | Weight for event-level scores |
| `lambda_texture` | float | `1.0` | Weight for texture-level scores |
| `lambda_proxy_texture` | float | `0.8` | Weight for proxy-texture blend |
| `smooth_texture` | float | `0.35` | Temporal smoothing for texture stream |
| `smooth_event` | float | `0.15` | Temporal smoothing for event stream |

## residual_ssm

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `d_model` | int | `64` | Hidden dimension |
| `d_state` | int | `8` | SSM state size |
| `n_ssm_layers` | int | `1` | Number of SSM layers |
| `dropout` | float | `0.1` | Dropout rate |
| `correction_weight` | float | `0.3` | Blend weight for residual correction |
| `n_epochs` | int | `20` | Training epochs |
| `lr` | float | `1e-3` | Learning rate |
| `patience` | int | `8` | Early-stopping patience |

## temperature

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `aves` | float | `1.10` | Temperature for bird (Aves) logits |
| `texture` | float | `0.95` | Temperature for texture taxa (Amphibia, Insecta) |

## Inference (top-level)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `file_level_top_k` | int | `2` | Keep top-k species per file |
| `tta_shifts` | list[int] | `[0]` | Test-time augmentation shifts (seconds) |

## probe

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | str | `"mlp"` | `"mlp"` or `"logistic"` |
| `pca_dim` | int | `64` | PCA dimension for probe input |
| `min_pos` | int | `8` | Minimum positive examples to train a probe |
| `C` | float | `0.5` | Logistic regularisation strength |
| `alpha` | float | `0.4` | Probe blend weight |

### probe.mlp

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hidden_layer_sizes` | list[int] | `[128]` | MLP hidden layers |
| `activation` | str | `"relu"` | Activation function |
| `max_iter` | int | `100` | Max training iterations |
| `early_stopping` | bool | `true` | Enable early stopping |
| `validation_fraction` | float | `0.15` | Validation split for early stopping |
| `n_iter_no_change` | int | `10` | Early stopping patience |
| `learning_rate_init` | float | `0.001` | Initial learning rate |
| `l2_alpha` | float | `0.01` | L2 regularisation |

## timer

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `budget_seconds` | float | `5400.0` | Total wall-clock budget (90 min) |
| `residual_ssm_min_remaining` | float | `240.0` | Skip residual SSM if less time remains |

## neuropt

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `search_space` | dict | *(see below)* | Parameter search ranges |
| `batch_size` | int | `3` | Parallel evaluations per neuropt round |
| `max_evals` | int | `60` | Maximum total evaluations |
| `backend` | str | `"claude"` | neuropt backend |

### neuropt.search_space

| Param | Type | Range |
|-------|------|-------|
| `lr` | log_uniform | [5e-4, 3e-3] |
| `weight_decay` | log_uniform | [8e-4, 5e-3] |
| `distill_weight` | uniform | [0.05, 0.25] |
| `label_smoothing` | uniform | [0.01, 0.05] |
| `mixup_alpha` | uniform | [0.1, 0.45] |
| `focal_gamma` | uniform | [1.0, 3.0] |
| `swa_start_frac` | uniform | [0.6, 0.85] |
| `pos_weight_cap` | uniform | [20, 45] |
