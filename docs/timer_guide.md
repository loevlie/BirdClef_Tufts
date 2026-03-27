# Wall Timer and Budget System

The Kaggle submission environment has a 90-minute wall-clock limit. The timer system tracks per-stage time and makes runtime decisions about which optional stages to run.

## WallTimer API

`src/timer/wallclock.py` -- single class, no dependencies beyond `time`.

```python
from src.timer.wallclock import WallTimer

timer = WallTimer(budget_seconds=5400.0)

# Track a stage
timer.stage_start("proto_ssm_train")
# ... training code ...
timer.stage_end()

# Query remaining budget
timer.remaining()        # seconds left
timer.elapsed()          # seconds used

# Conditional skip
if timer.should_skip("residual_ssm", min_remaining=240.0):
    print("Skipping residual SSM -- not enough time")

# Final report
timer.print_report()
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `elapsed()` | `float` | Seconds since timer creation |
| `remaining()` | `float` | `budget - elapsed()` |
| `stage_start(name)` | -- | Start timing a named stage. Auto-ends any open stage. |
| `stage_end()` | `float` | End current stage, return its duration |
| `should_skip(stage, min_remaining)` | `bool` | `True` if `remaining() < min_remaining` |
| `report()` | `dict` | `{"elapsed", "remaining", "budget", "stages"}` |
| `print_report()` | -- | Formatted table with per-stage time, % of budget, and SAFE/TIGHT/OVER verdict |

### Report Output

```
Stage                             Time(s)   % Budget
--------------------------------------------------
imports                               2.1       0.4%
model_init                            0.3       0.1%
proto_ssm_train                     145.2      26.9%
proto_ssm_inference                   1.8       0.3%
mlp_probes                           89.4      16.6% [!]
score_fusion                          0.2       0.0%
--------------------------------------------------
Total:                              239.0      44.3%
VERDICT: SAFE (55.7% margin)
```

Stages using >40% of the budget are flagged with `[!]`.

Verdict thresholds: **SAFE** (>10% margin), **TIGHT** (0--10%), **OVER BUDGET** (<0%).

## Config Keys

Two keys in `timer:` control behavior:

| Key | Default | Effect |
|-----|---------|--------|
| `budget_seconds` | `5400.0` | Total time budget |
| `residual_ssm_min_remaining` | `240.0` | Skip residual SSM if less time remains |

## Profiling Wall Time

The profiler runs the full pipeline with mock data to estimate timing without needing real competition data.

```bash
uv run python scripts/profile_time.py
uv run python scripts/profile_time.py --mock-test-files 500
uv run python scripts/profile_time.py --mock-test-files 500 --budget 540
```

### What the Profiler Measures

| Stage | What It Does |
|-------|-------------|
| `imports` | Import numpy, torch |
| `model_init` | Instantiate ProtoSSMv2 from config |
| `proto_ssm_train` | Train on 60 mock files, real epoch count from config |
| `proto_ssm_inference` | Forward pass on `--mock-test-files` files |
| `mlp_probes` | Train 100 MLP probes (typical eligible class count) |
| `score_fusion` | Temperature scaling and calibration |

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/base.yaml` | YAML config path |
| `--mock-test-files` | `200` | Number of simulated test files |
| `--budget` | from config | Override budget in seconds |

## Budget Strategy

The pipeline is designed to fit within 540s with margin. The key time sinks are:

1. **ProtoSSM training** (~25--30% of budget) -- controlled by `n_epochs` and `patience`
2. **MLP probes** (~15--20%) -- scales with number of eligible classes
3. **Residual SSM** (optional) -- only runs if enough time remains

If profiling shows TIGHT or OVER BUDGET, reduce `n_epochs`, raise `patience` thresholds, or reduce `probe.mlp.max_iter`.
