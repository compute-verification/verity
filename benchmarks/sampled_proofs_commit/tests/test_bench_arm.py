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
    assert bench_arm.settings({}) == {"window_mb": 256, "window_slots": 8, "levels": "drop", "host_gb": 80.0}
    with pytest.raises(ValueError, match="VERITY_SP_ARM_HOST_GB"):
        bench_arm.settings({"VERITY_SP_ARM_HOST_GB": "0"})
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


class _Block(bytearray):
    def numel(self):
        return len(self)


class _FakePool:
    def __init__(self):
        self.free, self.all = [], []

    def take(self, nbytes):
        blk = _Block(max(nbytes, 64))
        self.all.append(blk)
        return memoryview(blk)[:nbytes]

    def take_block(self, nbytes):
        for i, b in enumerate(self.free):
            if len(b) >= nbytes:
                return self.free.pop(i)
        blk = _Block(max(nbytes, 64))
        self.all.append(blk)
        return blk

    def give_back(self, blk):
        self.free.append(blk)

    @property
    def bytes_total(self):
        return sum(len(b) for b in self.all)


def _fake_windowed(fail_at=None, sizes=(1000, 100)):
    com = SimpleNamespace(pool=_FakePool(), _gpu_levels={}, _done={})

    def step(job):
        for n in sizes:
            com.pool.take(n)
        com._gpu_levels[job.step] = ["levels"]
        com._done[job.step] = SimpleNamespace(n_bytes=8000)
        if job.step == fail_at:
            raise RuntimeError("worker step failed")

    com._commit_step_windowed = step
    return com


def test_dropped_levels_are_reused_by_every_later_step_even_after_a_failed_one():
    com = _fake_windowed(fail_at=1)
    peak = bench_arm._guard_steps(com, drop_levels=True, cap_bytes=1 << 30)
    for step in range(6):
        try:
            com._commit_step_windowed(SimpleNamespace(step=step))
        except RuntimeError:
            assert step == 1
    assert com.pool.bytes_total == 1100 and len(com.pool.free) == 2
    assert all(v == [] for v in com._gpu_levels.values())
    assert peak["step_bytes"] == 8000 and peak["pool_bytes"] == 1100
    other = com.pool.take(10)                      # outside a step: the pool's own slice, not a tracked block
    assert len(other) == 10 and len(com.pool.free) == 2


def test_a_small_take_never_takes_a_big_free_block():
    com = _fake_windowed(sizes=(100, 4000))        # a window's leaves first, the step's levels once its length is known
    bench_arm._guard_steps(com, drop_levels=True, cap_bytes=1 << 30)
    com.pool.free = [_Block(5000), _Block(1000)]
    com.pool.all = list(com.pool.free)
    for step in range(4):
        com._commit_step_windowed(SimpleNamespace(step=step))
    assert com.pool.bytes_total == 6000


def test_steps_are_refused_by_name_once_the_pool_is_over_the_cap():
    com = _fake_windowed()
    peak = bench_arm._guard_steps(com, drop_levels=False, cap_bytes=2000)
    com._commit_step_windowed(SimpleNamespace(step=0))
    com._commit_step_windowed(SimpleNamespace(step=1))
    with pytest.raises(RuntimeError, match="VERITY_SP_ARM_HOST_GB"):
        com._commit_step_windowed(SimpleNamespace(step=2))
    assert peak["refused_steps"] == 1 and com.pool.bytes_total == 2200


def test_a_failed_gate_run_fails_the_arm_gate():
    state = bench_arm._Committing()
    state.com = _FakeCollector()
    state.spans()
    state.com._step_index = 1
    state.com.finalize = lambda: _rc([(256, 0.0)], errors_n=2)
    assert not state.gate()["passed"]
