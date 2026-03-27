"""Tests for the notebook bundler."""

import os
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestBundleNotebook:
    def _bundle(self):
        from build.bundle import bundle_notebook

        config_dict = {"mode": "submit", "proto_ssm": {"d_model": 128}}
        return bundle_notebook(config_dict, root=ROOT)

    @pytest.mark.slow
    def test_produces_valid_notebook_structure(self):
        nb = self._bundle()
        assert "nbformat" in nb
        assert nb["nbformat"] == 4
        assert "cells" in nb
        assert len(nb["cells"]) > 0
        for cell in nb["cells"]:
            assert "cell_type" in cell
            assert cell["cell_type"] in ("code", "markdown")
            assert "source" in cell

    @pytest.mark.slow
    def test_validate_notebook_no_errors(self):
        from build.bundle import validate_notebook

        nb = self._bundle()
        errors = validate_notebook(nb)
        assert errors == [], f"Validation errors: {errors}"

    @pytest.mark.slow
    def test_no_src_imports_in_bundled_output(self):
        nb = self._bundle()
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            assert "from src." not in source, f"Cell {i} contains 'from src.' import"
            assert "import src." not in source, f"Cell {i} contains 'import src.' import"


class TestReadAndStripModule:
    def test_strips_internal_imports(self):
        from build.bundle import read_and_strip_module

        code = read_and_strip_module("src/evaluation/smoothing.py", root=ROOT)
        assert "from src." not in code
        assert "from ." not in code
        # Real code should still be present
        assert "smooth_cols_fixed12" in code


class TestMakeCells:
    def test_code_cell_structure(self):
        from build.bundle import make_code_cell

        cell = make_code_cell("x = 1")
        assert cell["cell_type"] == "code"
        assert cell["source"] == ["x = 1"]
        assert cell["outputs"] == []

    def test_markdown_cell_structure(self):
        from build.bundle import make_markdown_cell

        cell = make_markdown_cell("# Title")
        assert cell["cell_type"] == "markdown"
        assert cell["source"] == ["# Title"]
