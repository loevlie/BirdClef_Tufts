"""Apply neuropt best config to a BirdCLEF CFG dict."""
import json
from pathlib import Path


def apply_neuropt_config(cfg_dict: dict, nc: dict):
    """Apply neuropt best config to a CFG dict. Architecture stays LOCKED."""
    if nc is None:
        return
    t = cfg_dict.get("proto_ssm_train", {})
    for k in ["lr", "weight_decay", "distill_weight", "label_smoothing",
              "mixup_alpha", "focal_gamma", "swa_start_frac", "pos_weight_cap"]:
        if k in nc:
            t[k] = float(nc[k])
    temp = cfg_dict.get("temperature", {})
    if "temperature_aves" in nc:
        temp["aves"] = float(nc["temperature_aves"])
    if "temperature_texture" in nc:
        temp["texture"] = float(nc["temperature_texture"])


def load_neuropt_state(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        state = json.load(f)
    return state


def load_and_apply_best(cfg_dict: dict, state_path: str | Path, min_improvement: float = 0.001):
    """Load neuropt state and apply best config if it beats baseline."""
    state = load_neuropt_state(state_path)
    if state is None:
        print("No neuropt state found — using default config.")
        return False
    best = state.get("best_config")
    score = state.get("best_score", 0)
    baseline = state.get("baseline", 0)
    if best and score > baseline + min_improvement:
        apply_neuropt_config(cfg_dict, best)
        print(f"Applied neuropt config (AUC={score:.4f}, baseline={baseline:.4f})")
        return True
    else:
        print(f"neuropt did not beat baseline — using default config.")
        return False
