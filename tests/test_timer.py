"""Tests for the WallTimer budget timer."""

import time
import pytest


class TestWallTimer:
    def test_elapsed_increases(self):
        from src.timer.wallclock import WallTimer

        timer = WallTimer(budget_seconds=100.0)
        t0 = timer.elapsed()
        time.sleep(0.05)
        t1 = timer.elapsed()
        assert t1 > t0

    def test_remaining_decreases(self):
        from src.timer.wallclock import WallTimer

        timer = WallTimer(budget_seconds=100.0)
        r0 = timer.remaining()
        time.sleep(0.05)
        r1 = timer.remaining()
        assert r1 < r0

    def test_should_skip_when_budget_exhausted(self):
        from src.timer.wallclock import WallTimer

        timer = WallTimer(budget_seconds=0.0)
        assert timer.should_skip("some_stage", min_remaining=10.0) is True

    def test_should_skip_when_budget_ample(self):
        from src.timer.wallclock import WallTimer

        timer = WallTimer(budget_seconds=1000.0)
        assert timer.should_skip("some_stage", min_remaining=10.0) is False

    def test_stage_start_end_records_timing(self):
        from src.timer.wallclock import WallTimer

        timer = WallTimer(budget_seconds=100.0)
        timer.stage_start("test_stage")
        time.sleep(0.05)
        dt = timer.stage_end()
        assert dt > 0
        assert "test_stage" in timer.stages
        assert timer.stages["test_stage"] > 0

    def test_stage_auto_close_on_new_start(self):
        from src.timer.wallclock import WallTimer

        timer = WallTimer(budget_seconds=100.0)
        timer.stage_start("stage_a")
        time.sleep(0.02)
        timer.stage_start("stage_b")
        # stage_a should have been auto-closed
        assert "stage_a" in timer.stages
        timer.stage_end()
        assert "stage_b" in timer.stages

    def test_report_contains_expected_keys(self):
        from src.timer.wallclock import WallTimer

        timer = WallTimer(budget_seconds=100.0)
        timer.stage_start("test_stage")
        timer.stage_end()
        report = timer.report()
        assert "elapsed" in report
        assert "remaining" in report
        assert "budget" in report
        assert "stages" in report
        assert report["budget"] == 100.0
        assert "test_stage" in report["stages"]
