"""The reference pool (`harness/refs.py`) and the gate's prefetch hook."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import arm as A  # noqa: E402
import bench as H  # noqa: E402
import refs as RF  # noqa: E402


def _square(x: int) -> tuple[int, int]:
    return x * x, os.getpid()


@pytest.fixture
def fresh_pool(monkeypatch):
    monkeypatch.setattr(RF, "_POOL", None)
    yield
    if RF._POOL is not None:
        RF._POOL.shutdown()


def test_cpus_is_the_fill_jobs_budget_within_the_cpus_the_process_may_use(monkeypatch):
    allowed = len(os.sched_getaffinity(0))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert RF.cpus() == 1
    monkeypatch.setenv("OMP_NUM_THREADS", str(allowed + 7))
    assert RF.cpus() == allowed
    monkeypatch.setenv("OMP_NUM_THREADS", "eight")
    assert RF.cpus() == allowed
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert RF.cpus() == allowed


def test_pool_is_none_on_one_cpu_else_one_forked_pool_of_the_jobs_cpus(monkeypatch, fresh_pool):
    monkeypatch.setattr(RF, "cpus", lambda: 1)
    assert RF.pool() is None
    monkeypatch.setattr(RF, "cpus", lambda: 2)
    p = RF.pool()
    assert p is RF.pool() and p._max_workers == 2
    results = [f.result() for f in [p.submit(_square, i) for i in range(4)]]
    assert [r for r, _ in results] == [0, 1, 4, 9] and os.getpid() not in {pid for _, pid in results}


class _Dev:
    stream = 7

    def __init__(self, log):
        self.log = log

    def sync(self):
        self.log.append("sync")


class _Call:
    outputs = {"y": A.Buffer(0, 4)}

    def __init__(self, i, log):
        self.i, self.log = i, log

    def launch(self, stream):
        self.log.append(f"launch {self.i}")


class _Arm:
    def __init__(self, log, ok=True):
        self.log, self.ok = log, ok

    def gate(self, dev, call, shape, flags):
        self.log.append(f"gate {call.i}")
        return A.GateResult(self.ok, {"y": "all"})


class _PrefetchingArm(_Arm):
    def prefetch(self, dev, call, shape, flags):
        self.log.append(f"prefetch {call.i}")


@pytest.mark.parametrize("ok", [True, False])
def test_the_gate_prefetches_every_call_it_gates_before_the_first_and_decides_as_before(ok):
    log: list[str] = []
    calls = [_Call(i, log) for i in range(4)]
    passed, detail = H._arm_gate(_Dev(log), _PrefetchingArm(log, ok), None, frozenset(), calls, [0, 3])()
    assert passed is ok and list(detail) == (["0", "3"] if ok else ["0"])
    assert log[:4] == ["sync", "prefetch 0", "prefetch 3", "launch 0"]
    assert log[4:] == (["sync", "gate 0", "launch 3", "sync", "gate 3"] if ok else ["sync", "gate 0"])


def test_an_arm_without_prefetch_is_gated_exactly_as_before():
    log: list[str] = []
    calls = [_Call(i, log) for i in range(4)]
    assert H._arm_gate(_Dev(log), _Arm(log), None, frozenset(), calls, [0, 3])()[0]
    assert log == ["launch 0", "sync", "gate 0", "launch 3", "sync", "gate 3"]


def test_prefetch_is_optional_in_the_interface_but_refused_when_not_callable():
    class Member(_PrefetchingArm):
        name, stream, precision, inputs = "a", "fp8", "fp8-e4m3", "e4m3"
        flag_sets = {"hash": frozenset({"h"}), "nohash": frozenset()}
        build = domain = work = prepare = None

    assert A.VERSION == "v0.6" and not hasattr(A.BenchArm, "prefetch")
    A.check_arm(Member([]), {"fp8-e4m3"})
    bad = Member([])
    bad.prefetch = True
    with pytest.raises(A.ArmError, match="prefetch must be callable"):
        A.check_arm(bad, {"fp8-e4m3"})
