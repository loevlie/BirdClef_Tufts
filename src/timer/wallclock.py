"""Wall-clock budget timer for Kaggle submission time management."""

import time


class WallTimer:
    def __init__(self, budget_seconds: float = 5400.0):
        self.start = time.time()
        self.budget = budget_seconds
        self.stages: dict[str, float] = {}
        self._stage_start = None
        self._current_stage = None

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.budget - self.elapsed()

    def stage_start(self, name: str):
        if self._current_stage:
            self.stage_end()
        self._current_stage = name
        self._stage_start = time.time()

    def stage_end(self) -> float:
        if self._current_stage and self._stage_start:
            dt = time.time() - self._stage_start
            self.stages[self._current_stage] = dt
            self._current_stage = None
            self._stage_start = None
            return dt
        return 0.0

    def should_skip(self, stage: str, min_remaining: float) -> bool:
        return self.remaining() < min_remaining

    def report(self) -> dict:
        return {
            "elapsed": self.elapsed(),
            "remaining": self.remaining(),
            "budget": self.budget,
            "stages": dict(self.stages),
        }

    def print_report(self):
        total = self.elapsed()
        print(f"\n{'Stage':<30} {'Time(s)':>8} {'% Budget':>10}")
        print("-" * 50)
        for name, dt in self.stages.items():
            pct = dt / self.budget * 100
            flag = " [!]" if pct > 40 else ""
            print(f"{name:<30} {dt:>8.1f} {pct:>9.1f}%{flag}")
        print("-" * 50)
        margin = self.remaining() / self.budget * 100
        verdict = "SAFE" if margin > 10 else "TIGHT" if margin > 0 else "OVER BUDGET"
        print(f"{'Total:':<30} {total:>8.1f} {total/self.budget*100:>9.1f}%")
        print(f"VERDICT: {verdict} ({margin:.1f}% margin)")
