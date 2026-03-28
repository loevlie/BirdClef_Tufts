#!/usr/bin/env python3
"""Export a self-contained Kaggle submission notebook."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from build.bundle import bundle_notebook, validate_notebook

TF_INSTALL = '''# Install onnxruntime from bundled wheel (no internet needed)
import glob
_whl = glob.glob("/kaggle/input/*/onnxruntime*.whl")
if _whl:
    !pip install -q {_whl[0]}
else:
    !pip install -q onnxruntime  # fallback if internet available
'''


def main():
    parser = argparse.ArgumentParser(description="Export submission notebook")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--neuropt-state", default=None, help="Path to neuropt_state.json")
    parser.add_argument("-o", "--output", default="build/output/submission.ipynb")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    # Apply neuropt best params if provided
    if args.neuropt_state:
        from src.neuropt_integration.config_apply import load_and_apply_best
        load_and_apply_best(config_dict, args.neuropt_state)

    # Force submit mode
    config_dict["mode"] = "submit"

    root = str(Path(__file__).parent.parent)
    nb = bundle_notebook(config_dict, root=root, tf_install_code=TF_INSTALL)

    errors = validate_notebook(nb)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(nb, f, indent=1)

    n_cells = len(nb["cells"])
    print(f"Wrote {output} ({n_cells} cells)")


if __name__ == "__main__":
    main()
