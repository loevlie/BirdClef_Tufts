"""Tests for evaluation metrics, smoothing, and feature engineering."""

import numpy as np
import pytest


class TestMacroAUC:
    def test_known_values(self):
        from src.evaluation.metrics import macro_auc_skip_empty

        # Perfect predictions: AUC should be 1.0
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        auc = macro_auc_skip_empty(y_true, y_score)
        assert auc == pytest.approx(1.0)

    def test_handles_all_zero_columns(self):
        from src.evaluation.metrics import macro_auc_skip_empty

        # Column 1 is all zeros -- should be skipped, not crash
        y_true = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
        y_score = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9],
                            [0.8, 0.5, 0.2], [0.2, 0.5, 0.8]])
        auc = macro_auc_skip_empty(y_true, y_score)
        assert 0.0 <= auc <= 1.0


class TestSmoothCols:
    def test_preserves_shape(self):
        from src.evaluation.smoothing import smooth_cols_fixed12

        scores = np.random.default_rng(0).random((24, 10)).astype(np.float32)
        result = smooth_cols_fixed12(scores, cols=[0, 1, 2], alpha=0.35)
        assert result.shape == (24, 10)

    def test_no_mutation(self):
        from src.evaluation.smoothing import smooth_cols_fixed12

        scores = np.random.default_rng(0).random((24, 10)).astype(np.float32)
        original = scores.copy()
        _ = smooth_cols_fixed12(scores, cols=[0, 1, 2])
        np.testing.assert_array_equal(scores, original)

    def test_alpha_zero_is_identity(self):
        from src.evaluation.smoothing import smooth_cols_fixed12

        scores = np.random.default_rng(0).random((12, 5)).astype(np.float32)
        result = smooth_cols_fixed12(scores, cols=[0, 1], alpha=0.0)
        np.testing.assert_array_equal(result, scores)


class TestSmoothEvents:
    def test_preserves_shape(self):
        from src.evaluation.smoothing import smooth_events_fixed12

        scores = np.random.default_rng(0).random((24, 10)).astype(np.float32)
        result = smooth_events_fixed12(scores, cols=[3, 4, 5], alpha=0.15)
        assert result.shape == (24, 10)


class TestSeqFeatures:
    def test_output_shapes(self):
        from src.evaluation.features import seq_features_1d

        v = np.random.default_rng(0).random(24).astype(np.float32)
        prev_v, next_v, mean_v, max_v, std_v = seq_features_1d(v)
        assert prev_v.shape == (24,)
        assert next_v.shape == (24,)
        assert mean_v.shape == (24,)
        assert max_v.shape == (24,)
        assert std_v.shape == (24,)


class TestBuildClassFeatures:
    def test_output_shape(self):
        from src.evaluation.features import build_class_features

        n = 24
        d = 8
        rng = np.random.default_rng(0)
        emb_proj = rng.random((n, d)).astype(np.float32)
        raw_col = rng.random(n).astype(np.float32)
        prior_col = rng.random(n).astype(np.float32)
        base_col = rng.random(n).astype(np.float32)

        feats = build_class_features(emb_proj, raw_col, prior_col, base_col)
        # d + 14 extra features (11 scalar + 3 interaction)
        assert feats.shape == (n, d + 14)
        assert feats.dtype == np.float32
