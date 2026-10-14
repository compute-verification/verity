from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench_arm


def test_the_commit_arm_is_gated_against_stock_fp8():
    assert (bench_arm.COMMIT.id, bench_arm.COMMIT.control, bench_arm.COMMIT.precision) == ("sp-commit", "stock-fp8", "fp8")
    assert bench_arm.COMMIT.protocols == ()


def test_settings_refuse_by_name():
    assert bench_arm.settings({}) == {"window_mb": 256, "window_slots": 8, "levels": "drop"}
    with pytest.raises(ValueError, match="VERITY_SP_ARM_WINDOW_MB"):
        bench_arm.settings({"VERITY_SP_ARM_WINDOW_MB": "96"})
    with pytest.raises(ValueError, match="VERITY_SP_ARM_WINDOW_SLOTS"):
        bench_arm.settings({"VERITY_SP_ARM_WINDOW_SLOTS": "2"})
    with pytest.raises(ValueError, match="VERITY_SP_ARM_LEVELS"):
        bench_arm.settings({"VERITY_SP_ARM_LEVELS": "host"})


def _rc(steps, errors_n=0, expected=10, committed=10, root=b"\x01" * 32):
    return SimpleNamespace(
        run_root=root,
        steps=[SimpleNamespace(n_bytes=b, n_leaves=b // 256, t_step_end=t) for b, t in steps],
        root_ready_delay_s=0.25,
        t_finalize_s=0.3,
        extra={"errors_n": errors_n, "errors": [], "expected_tensors": expected, "committed_tensors": committed,
               "gpu_tree_ms": {"leaf_ms": 1.0, "tree_ms": 2.0}},
    )


def test_a_run_counts_only_when_every_staged_tensor_is_committed_without_error():
    good = bench_arm.record_of(_rc([(512, 1.0), (1024, 2.0)]), 2.1)
    assert bench_arm.run_passed(good)
    assert good["committed_bytes"] == 1536 and good["steps"] == 2
    assert good["root_after_serve_end_s"] == pytest.approx(0.15)
    assert not bench_arm.run_passed(bench_arm.record_of(_rc([(512, 1.0)], errors_n=1), None))
    assert not bench_arm.run_passed(bench_arm.record_of(_rc([(512, 1.0)], committed=9), None))
    assert not bench_arm.run_passed(bench_arm.record_of(_rc([(512, 1.0)], root=b""), None))
    assert not bench_arm.run_passed(bench_arm.record_of(_rc([]), None))


class _FakeCollector:
    name = "fake"

    def __init__(self):
        self._step_index = 0
        self.resets = 0
        self.finalized = 0
        self.closed = False

    def reset(self):
        self.resets += 1
        self._step_index = 0

    def finalize(self):
        self.finalized += 1
        return _rc([(256, 0.0)] * self._step_index)

    def close(self):
        self.closed = True


def test_each_driver_call_closes_the_run_in_flight_and_opens_the_next():
    state = bench_arm._Committing()
    state.com = _FakeCollector()
    state.spans()                      # gate run of point 1 opens
    state.com._step_index = 3
    state.spans()                      # gate run of point 2 closes point 1's
    state.com._step_index = 2
    gate = state.gate()
    assert gate["passed"] and gate["runs"] == 2
    state.spans()                      # measured run opens
    state.com._step_index = 5
    state.spans()                      # after the measured run: closed, a stepless run opens
    assert state.extras()["steps"] == 5.0 and "serve_to_root_s" in state.extras()
    state.spans()                      # the stepless run is dropped
    assert len(state.runs) == 3
    state.close()
    assert state.com.closed and len(state.runs) == 3


def test_a_failed_gate_run_fails_the_arm_gate():
    state = bench_arm._Committing()
    state.com = _FakeCollector()
    state.spans()
    state.com._step_index = 1
    state.com.finalize = lambda: _rc([(256, 0.0)], errors_n=2)
    assert not state.gate()["passed"]
