"""Tests for data parsing and taxonomy utilities."""

import pandas as pd
import pytest


class TestParseSoundscapeFilename:
    def test_extracts_correct_fields(self):
        from src.data.parsing import parse_soundscape_filename

        result = parse_soundscape_filename("BC2026_Train_0001_S08_20250606_030007.ogg")
        assert result["file_id"] == "0001"
        assert result["site"] == "S08"
        assert result["time_utc"] == "030007"
        assert result["hour_utc"] == 3
        assert result["month"] == 6

    def test_invalid_filename_returns_nones(self):
        from src.data.parsing import parse_soundscape_filename

        result = parse_soundscape_filename("not_a_valid_file.wav")
        assert result["file_id"] is None
        assert result["site"] is None
        assert result["hour_utc"] == -1

    def test_test_prefix(self):
        from src.data.parsing import parse_soundscape_filename

        result = parse_soundscape_filename("BC2026_Test_0042_S03_20250115_180000.ogg")
        assert result["file_id"] == "0042"
        assert result["site"] == "S03"
        assert result["hour_utc"] == 18
        assert result["month"] == 1


class TestParseSoundscapeLabels:
    def test_semicolon_separated(self):
        from src.data.parsing import parse_soundscape_labels

        result = parse_soundscape_labels("species_a;species_b;species_c")
        assert result == ["species_a", "species_b", "species_c"]

    def test_empty_na(self):
        from src.data.parsing import parse_soundscape_labels

        assert parse_soundscape_labels(pd.NA) == []
        assert parse_soundscape_labels(float("nan")) == []

    def test_strips_whitespace(self):
        from src.data.parsing import parse_soundscape_labels

        result = parse_soundscape_labels(" sp_a ; sp_b ; sp_c ")
        assert result == ["sp_a", "sp_b", "sp_c"]


class TestTextureTaxa:
    def test_contains_expected_values(self):
        from src.data.taxonomy import TEXTURE_TAXA

        assert "Amphibia" in TEXTURE_TAXA
        assert "Insecta" in TEXTURE_TAXA
        assert isinstance(TEXTURE_TAXA, set)


class TestBuildTaxonomyGroups:
    def test_returns_valid_structure(self):
        from src.data.taxonomy import build_taxonomy_groups

        taxonomy_df = pd.DataFrame({
            "primary_label": ["sp_a", "sp_b", "sp_c"],
            "family": ["Fam1", "Fam2", "Fam1"],
        })
        primary_labels = ["sp_a", "sp_b", "sp_c"]
        n_groups, class_to_group, grp_to_idx = build_taxonomy_groups(taxonomy_df, primary_labels)

        assert n_groups == 2  # Fam1, Fam2
        assert len(class_to_group) == 3
        assert all(isinstance(g, int) for g in class_to_group)
        assert isinstance(grp_to_idx, dict)
        assert set(grp_to_idx.keys()) == {"Fam1", "Fam2"}

    def test_missing_columns_fallback(self):
        from src.data.taxonomy import build_taxonomy_groups

        taxonomy_df = pd.DataFrame({
            "primary_label": ["sp_a", "sp_b"],
            "some_other_col": [1, 2],
        })
        primary_labels = ["sp_a", "sp_b"]
        n_groups, class_to_group, grp_to_idx = build_taxonomy_groups(taxonomy_df, primary_labels)

        assert n_groups >= 1
        assert "Unknown" in grp_to_idx
