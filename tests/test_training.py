"""Tests for training losses and augmentation."""

import numpy as np
import torch
import pytest


class TestFocalBCE:
    def test_gamma_zero_matches_bce(self):
        from src.training.losses import focal_bce_with_logits

        logits = torch.randn(4, 10)
        targets = (torch.rand(4, 10) > 0.5).float()
        focal_loss = focal_bce_with_logits(logits, targets, gamma=0.0)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        assert torch.allclose(focal_loss, bce_loss, atol=1e-5)

    def test_returns_scalar(self):
        from src.training.losses import focal_bce_with_logits

        logits = torch.randn(4, 10)
        targets = (torch.rand(4, 10) > 0.5).float()
        loss = focal_bce_with_logits(logits, targets, gamma=2.0)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        from src.training.losses import focal_bce_with_logits

        logits = torch.randn(4, 10, requires_grad=True)
        targets = (torch.rand(4, 10) > 0.5).float()
        loss = focal_bce_with_logits(logits, targets, gamma=2.0)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (4, 10)

    def test_no_reduction(self):
        from src.training.losses import focal_bce_with_logits

        logits = torch.randn(4, 10)
        targets = (torch.rand(4, 10) > 0.5).float()
        loss = focal_bce_with_logits(logits, targets, gamma=2.0, reduction="none")
        assert loss.shape == (4, 10)


class TestMixupFiles:
    def test_preserves_shapes(self):
        from src.training.augmentation import mixup_files

        rng = np.random.default_rng(42)
        n, t, d, c = 8, 12, 1536, 234
        emb = rng.standard_normal((n, t, d)).astype(np.float32)
        logits = rng.standard_normal((n, t, c)).astype(np.float32)
        labels = (rng.random((n, t, c)) > 0.8).astype(np.float32)
        site_ids = rng.integers(0, 20, size=n)
        hours = rng.integers(0, 24, size=n)
        families = rng.standard_normal((n, t, 10)).astype(np.float32)

        out = mixup_files(emb, logits, labels, site_ids, hours, families, alpha=0.3)
        emb_out, logits_out, labels_out, sites_out, hours_out, fam_out = out

        assert emb_out.shape == emb.shape
        assert logits_out.shape == logits.shape
        assert labels_out.shape == labels.shape
        # Discrete features are passed through unchanged
        np.testing.assert_array_equal(sites_out, site_ids)
        np.testing.assert_array_equal(hours_out, hours)

    def test_alpha_zero_is_identity(self):
        from src.training.augmentation import mixup_files

        rng = np.random.default_rng(42)
        emb = rng.standard_normal((4, 12, 64)).astype(np.float32)
        logits = rng.standard_normal((4, 12, 10)).astype(np.float32)
        labels = (rng.random((4, 12, 10)) > 0.5).astype(np.float32)
        site_ids = np.zeros(4, dtype=np.int64)
        hours = np.zeros(4, dtype=np.int64)

        out = mixup_files(emb, logits, labels, site_ids, hours, None, alpha=0.0)
        np.testing.assert_array_equal(out[0], emb)
        np.testing.assert_array_equal(out[1], logits)
        np.testing.assert_array_equal(out[2], labels)
