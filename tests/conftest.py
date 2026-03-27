"""Shared fixtures for BirdCLEF 2026 test suite."""

import numpy as np
import pytest
import torch

from src.config.schema import Config


@pytest.fixture
def mini_config():
    """Minimal Config with default values for fast tests."""
    return Config()


@pytest.fixture
def synthetic_embeddings():
    """Random float32 embeddings shaped (3, 12, 1536)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 12, 1536)).astype(np.float32)


@pytest.fixture
def synthetic_labels():
    """Random multi-hot uint8 labels shaped (3, 12, 10)."""
    rng = np.random.default_rng(42)
    return (rng.random((3, 12, 10)) > 0.8).astype(np.uint8)


@pytest.fixture
def synthetic_logits():
    """Random float32 logits shaped (3, 12, 10)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 12, 10)).astype(np.float32)
