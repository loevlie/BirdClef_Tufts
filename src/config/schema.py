"""Typed dataclass schema for the BirdCLEF pipeline configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Tuple


# ── Architecture ─────────────────────────────────────────────────────────────

@dataclass
class ProtoSSMArch:
    d_model: int = 128
    d_state: int = 16
    n_ssm_layers: int = 2
    dropout: float = 0.15
    n_prototypes: int = 1
    n_sites: int = 20
    meta_dim: int = 16
    use_cross_attn: bool = True
    cross_attn_heads: int = 4


# ── Training ─────────────────────────────────────────────────────────────────

@dataclass
class ProtoSSMTrain:
    n_epochs: int = 35
    lr: float = 5.5e-4
    weight_decay: float = 1e-3
    val_ratio: float = 0.15
    patience: int = 15
    pos_weight_cap: float = 41.0
    distill_weight: float = 0.23
    proto_margin: float = 0.1
    label_smoothing: float = 0.019
    oof_n_splits: int = 3
    mixup_alpha: float = 0.25
    focal_gamma: float = 1.15
    swa_start_frac: float = 0.68
    swa_lr: float = 5e-4


# ── Fusion ───────────────────────────────────────────────────────────────────

@dataclass
class FusionParams:
    lambda_event: float = 0.4
    lambda_texture: float = 1.0
    lambda_proxy_texture: float = 0.8
    smooth_texture: float = 0.35
    smooth_event: float = 0.15


# ── Residual SSM ─────────────────────────────────────────────────────────────

@dataclass
class ResidualSSMConfig:
    d_model: int = 64
    d_state: int = 8
    n_ssm_layers: int = 1
    dropout: float = 0.1
    correction_weight: float = 0.3
    n_epochs: int = 20
    lr: float = 1e-3
    patience: int = 8


# ── Temperature ──────────────────────────────────────────────────────────────

@dataclass
class TemperatureConfig:
    aves: float = 1.10
    texture: float = 0.95


# ── Probe ────────────────────────────────────────────────────────────────────

@dataclass
class ProbeConfig:
    backend: str = "mlp"
    pca_dim: int = 64
    min_pos: int = 8
    C: float = 0.50
    alpha: float = 0.40
    hidden_layer_sizes: Tuple[int, ...] = (128,)
    activation: str = "relu"
    max_iter: int = 100
    early_stopping: bool = True
    validation_fraction: float = 0.15
    n_iter_no_change: int = 10
    learning_rate_init: float = 0.001
    l2_alpha: float = 0.01


# ── Timer ────────────────────────────────────────────────────────────────────

@dataclass
class TimerConfig:
    budget_seconds: float = 5400.0
    residual_ssm_min_remaining: float = 240.0


# ── Pipeline ─────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    batch_files: int = 32
    proxy_reduce: str = "max"
    require_full_cache_in_submit: bool = True
    verbose: bool = True
    run_oof_baseline: bool = False
    run_probe_check: bool = False
    dryrun_n_files: int = 20


# ── Paths ────────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    data_dir: str = "/kaggle/input/birdclef-2026"
    model_dir: str = "/kaggle/input/birdclef-2026-models"
    cache_dir: str = "/kaggle/working/cache"
    cache_input_dir: str = "/kaggle/input/birdclef-2026-cache"


# ── Top-level ────────────────────────────────────────────────────────────────

@dataclass
class Config:
    arch: ProtoSSMArch = field(default_factory=ProtoSSMArch)
    train: ProtoSSMTrain = field(default_factory=ProtoSSMTrain)
    fusion: FusionParams = field(default_factory=FusionParams)
    residual_ssm: ResidualSSMConfig = field(default_factory=ResidualSSMConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    timer: TimerConfig = field(default_factory=TimerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    mode: str = "submit"
    tta_shifts: List[int] = field(default_factory=lambda: [0])
    file_level_top_k: int = 2

    # Map dataclass field names to the notebook's CFG key names.
    _SECTION_KEYS = {
        "arch": "proto_ssm",
        "train": "proto_ssm_train",
        "fusion": "best_fusion",
        "residual_ssm": "residual_ssm",
        "temperature": "temperature",
        "probe": "probe",
        "timer": "timer",
        "pipeline": "pipeline",
        "paths": "paths",
    }

    def to_dict(self) -> dict:
        """Return a nested CFG-style dict matching what the notebook expects.

        Sub-dataclasses become nested dicts keyed by the notebook's names
        (e.g. ``arch`` -> ``"proto_ssm"``).  Scalar fields land at the
        top level.
        """
        out: dict = {}
        for field_name, val in asdict(self).items():
            if isinstance(val, dict):
                cfg_key = self._SECTION_KEYS.get(field_name, field_name)
                out[cfg_key] = val
            else:
                out[field_name] = val
        return out
