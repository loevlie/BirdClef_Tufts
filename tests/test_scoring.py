"""Tests for scoring calibration and prior tables."""

import numpy as np
import pandas as pd
import pytest


class TestFileLevelConfidenceScale:
    def test_preserves_shape(self):
        from src.scoring.calibration import file_level_confidence_scale

        preds = np.random.default_rng(0).random((24, 10)).astype(np.float32)
        result = file_level_confidence_scale(preds, n_windows=12, top_k=2)
        assert result.shape == (24, 10)

    def test_non_negative_output(self):
        from src.scoring.calibration import file_level_confidence_scale

        preds = np.random.default_rng(0).random((12, 5)).astype(np.float32)
        result = file_level_confidence_scale(preds, n_windows=12, top_k=2)
        assert (result >= 0).all()


class TestApplyTemperatureAndScale:
    def test_output_range(self):
        from src.scoring.calibration import apply_temperature_and_scale

        scores = np.random.default_rng(0).standard_normal((24, 10)).astype(np.float32)
        temps = np.ones(10, dtype=np.float32)
        result = apply_temperature_and_scale(scores, temps, n_windows=12, top_k=2)
        assert result.shape == (24, 10)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_no_file_scaling(self):
        from src.scoring.calibration import apply_temperature_and_scale

        scores = np.random.default_rng(0).standard_normal((12, 5)).astype(np.float32)
        temps = np.ones(5, dtype=np.float32)
        result = apply_temperature_and_scale(scores, temps, n_windows=12, top_k=0)
        # With top_k=0 we skip file-level scaling; output is pure sigmoid
        expected = 1.0 / (1.0 + np.exp(-scores))
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestFitPriorTables:
    def test_returns_all_expected_keys(self):
        from src.scoring.priors import fit_prior_tables

        prior_df = pd.DataFrame({
            "site": ["S01", "S01", "S02", "S02"],
            "hour_utc": [3, 3, 18, 18],
        })
        Y = np.array([
            [1, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ], dtype=np.float32)

        tables = fit_prior_tables(prior_df, Y)
        expected_keys = {
            "global_p", "site_to_i", "site_n", "site_p",
            "hour_to_i", "hour_n", "hour_p",
            "sh_to_i", "sh_n", "sh_p",
        }
        assert set(tables.keys()) == expected_keys

    def test_global_p_shape(self):
        from src.scoring.priors import fit_prior_tables

        prior_df = pd.DataFrame({
            "site": ["S01", "S02"],
            "hour_utc": [6, 18],
        })
        Y = np.array([[1, 0], [0, 1]], dtype=np.float32)
        tables = fit_prior_tables(prior_df, Y)
        assert tables["global_p"].shape == (2,)


class TestPriorLogitsFromTables:
    def test_output_shape(self):
        from src.scoring.priors import fit_prior_tables, prior_logits_from_tables

        prior_df = pd.DataFrame({
            "site": ["S01", "S01", "S02", "S02"],
            "hour_utc": [3, 3, 18, 18],
        })
        Y = np.array([
            [1, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ], dtype=np.float32)
        tables = fit_prior_tables(prior_df, Y)

        sites = ["S01", "S02", "S01"]
        hours = [3, 18, 3]
        logits = prior_logits_from_tables(sites, hours, tables)
        assert logits.shape == (3, 3)
        assert logits.dtype == np.float32

    def test_unknown_site_uses_global(self):
        from src.scoring.priors import fit_prior_tables, prior_logits_from_tables

        prior_df = pd.DataFrame({
            "site": ["S01", "S01"],
            "hour_utc": [6, 6],
        })
        Y = np.array([[1, 0], [0, 1]], dtype=np.float32)
        tables = fit_prior_tables(prior_df, Y)

        logits = prior_logits_from_tables(["UNKNOWN"], [6], tables)
        assert logits.shape == (1, 2)
        assert np.isfinite(logits).all()
