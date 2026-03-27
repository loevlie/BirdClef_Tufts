"""Bundle src/ modules into a self-contained Kaggle submission notebook."""
import ast
import json
import re
from pathlib import Path

# Define the order modules should appear in the notebook
MODULE_GROUPS = [
    ("Constants & Config", ["src/constants.py"]),
    ("Data Utilities", [
        "src/data/parsing.py",
        "src/data/taxonomy.py",
        "src/data/sites.py",
        "src/data/reshape.py",
    ]),
    ("Evaluation", [
        "src/evaluation/metrics.py",
        "src/evaluation/smoothing.py",
        "src/evaluation/features.py",
    ]),
    ("Models", [
        "src/models/ssm.py",
        "src/models/attention.py",
        "src/models/proto_ssm.py",
        "src/models/residual_ssm.py",
    ]),
    ("Training", [
        "src/training/losses.py",
        "src/training/augmentation.py",
        "src/training/trainer.py",
        "src/training/oof.py",
        "src/training/probes.py",
    ]),
    ("Inference", [
        "src/inference/audio.py",
        "src/inference/perch.py",
        "src/inference/tta.py",
    ]),
    ("Scoring", [
        "src/scoring/priors.py",
        "src/scoring/fusion.py",
        "src/scoring/calibration.py",
    ]),
    ("Submission", [
        "src/submission/generate.py",
    ]),
    ("Timer", [
        "src/timer/wallclock.py",
    ]),
]


def read_and_strip_module(filepath: str, root: str = ".") -> str:
    """Read a Python module and strip internal imports."""
    path = Path(root) / filepath
    source = path.read_text()

    lines = source.split("\n")
    cleaned = []
    for line in lines:
        # Remove `from src.xxx import yyy` and `import src.xxx`
        if re.match(r'\s*from\s+src\.', line) or re.match(r'\s*import\s+src\.', line):
            continue
        # Remove relative imports like `from .xxx import yyy`
        if re.match(r'\s*from\s+\.', line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def collect_third_party_imports(modules: list[str], root: str = ".") -> list[str]:
    """Collect and deduplicate third-party import statements."""
    imports = set()
    for filepath in modules:
        path = Path(root) / filepath
        source = path.read_text()
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Skip internal imports
                if "src." in stripped or stripped.startswith("from ."):
                    continue
                imports.add(stripped)
    return sorted(imports)


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


def make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }


def bundle_notebook(config_dict: dict, root: str = ".", tf_install_code: str = "") -> dict:
    """Bundle src/ into a self-contained notebook dict."""
    cells = []

    # Cell 0: TF install
    if tf_install_code:
        cells.append(make_code_cell(tf_install_code))

    # Cell 1: Mode switch
    cells.append(make_code_cell('MODE = "submit"\nassert MODE in {"train", "submit"}\nprint("MODE =", MODE)'))

    # Cell 2: Third-party imports
    all_modules = [m for _, modules in MODULE_GROUPS for m in modules]
    imports = collect_third_party_imports(all_modules, root)
    import_code = "\n".join(imports) + "\n\nimport warnings\nwarnings.filterwarnings('ignore')\n"
    cells.append(make_markdown_cell("## Imports"))
    cells.append(make_code_cell(import_code))

    # Cell 3: Baked config
    cells.append(make_markdown_cell("## Configuration"))
    cfg_code = f"import json\nCFG = {json.dumps(config_dict, indent=2, default=str)}\nprint(json.dumps(CFG, indent=2))"
    cells.append(make_code_cell(cfg_code))

    # Module groups
    for group_name, modules in MODULE_GROUPS:
        cells.append(make_markdown_cell(f"## {group_name}"))
        group_code = []
        for mod in modules:
            code = read_and_strip_module(mod, root)
            if code.strip():
                group_code.append(f"# --- {Path(mod).stem} ---\n{code}")
        cells.append(make_code_cell("\n\n".join(group_code)))

    # Build notebook structure
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    return nb


def validate_notebook(nb: dict) -> list[str]:
    """Check for leftover src imports."""
    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "from src." in source or "import src." in source:
            errors.append(f"Cell {i}: contains 'from src.' or 'import src.' import")
        if "from ." in source:
            # Check it's not in a string
            for line in source.split("\n"):
                stripped = line.strip()
                if stripped.startswith("from .") and not stripped.startswith("#"):
                    errors.append(f"Cell {i}: contains relative import: {stripped}")
    return errors
