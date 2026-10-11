"""`cli.parallel`, the per-target processes behind `circuit-check --jobs`, on its own: `test_circuit_check` builds every target at
collection, which this needs none of."""
from __future__ import annotations

import os
import signal

import pytest

from circuit_check import cli


def test_every_item_is_yielded_once():
    assert sorted(v for _, v, _, _ in cli.parallel(abs, [-1, -2, -3], jobs=2)) == [1, 2, 3]


_KEPT: list = []


def _allocate(mb: int) -> int:
    """A fake target that allocates ``mb`` MiB and keeps it, as a check that leaves its memo resident does."""
    block = bytearray(mb << 20)
    block[::4096] = b"x" * len(block[::4096])
    _KEPT.append(block)
    return mb


def test_a_target_s_peak_is_its_own():
    """A 600 MiB target's peak shows it, and the small target after it, in the same single job, does not inherit it: a worker
    that outlived the first kept its 600 MiB under the second's peak (the 104 GiB plateaus in the first heavy.json)."""
    got = {x: (peak, start) for x, _, peak, start in cli.parallel(_allocate, [600, 1], jobs=1)}
    (big, big_start), (small, small_start) = got[600], got[1]
    assert big - big_start >= 0.55
    assert small - small_start < 0.05 and small < big - 0.4


def _die_on_7(x: int) -> int:
    if x == 7:
        os.kill(os.getpid(), signal.SIGKILL)
    return x


def test_a_target_whose_process_is_killed_is_named_and_the_rest_still_run():
    got = {x: (v, peak) for x, v, peak, _ in cli.parallel(_die_on_7, [1, 7, 2, 3], jobs=2)}
    killed, peak = got.pop(7)
    assert isinstance(killed, cli.Killed) and killed["how"] == "signal 9" and peak > 0
    assert {x: v for x, (v, _) in got.items()} == {1: 1, 2: 2, 3: 3}


def _raise(x: int) -> int:
    raise ValueError(f"no {x}")


def test_a_target_that_raises_is_reported_not_lost():
    [(_, v, _, _)] = cli.parallel(_raise, [5], jobs=1)
    assert isinstance(v, cli.Killed) and v["how"] == "raised ValueError: no 5"


def test_heavy_targets_run_alone():
    """`admit` under a 160 GB budget: a 150 GiB target starts only when nothing runs, and nothing starts beside it; light ones
    fill what is left, past a heavy one that does not fit; one over the budget runs alone."""
    assert cli.admit([150, 16, 16], [], jobs=8, budget=160) == [0]
    assert cli.admit([16, 16], [150], jobs=8, budget=160) == []
    assert cli.admit([150, 16, 16], [100], jobs=8, budget=160) == [1, 2]
    assert cli.admit([16] * 12, [], jobs=8, budget=160) == list(range(8))
    assert cli.admit([200, 16], [], jobs=8, budget=160) == [0]
    assert cli.admit([16], [200], jobs=8, budget=160) == []


def test_a_budget_still_runs_every_target_each_in_a_process_of_its_own():
    got = list(cli.parallel(_pid, [1, 2, 3], jobs=3, weight=lambda _: 3, budget=4))
    assert sorted(x for x, *_ in got) == [1, 2, 3] and len({v for _, v, _, _ in got}) == 3


def _pid(_: int) -> int:
    return os.getpid()


def test_jobs_0_takes_one_worker_per_160_gb_the_tightest_cgroup_leaves(tmp_path):
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text("MemTotal: 2000000000 kB\nMemAvailable: 1600000000 kB\n")
    (tmp_path / "proc/self").mkdir()
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice/user-1001.slice/session-7.scope\n")
    scope = tmp_path / "sys/fs/cgroup/user.slice/user-1001.slice/session-7.scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text("max\n")
    (scope.parent / "memory.max").write_text(str(1600 << 30))
    (scope.parent / "memory.current").write_text(str(1200 << 30))
    assert cli.memory_gb(tmp_path) == 400
    assert cli.auto_jobs(tmp_path) == min(2, len(os.sched_getaffinity(0)))
    (scope.parent / "memory.current").write_text(str(1590 << 30))
    assert cli.auto_jobs(tmp_path) == 1


def test_the_budget_reserves_this_process_s_own_task_list(monkeypatch):
    """`--mem-gb 149` (a 160 GB worker) with the parent holding its 9.6 GiB task list leaves 139.4 GiB for the targets, so a
    151 GiB target is over it (run alone, and warned about) and a 137.8 GiB one fits.  `--mem-gb` is a cap, never more than the
    machine has free: a check slot's mem= on a node whose other check already took most of it leaves only what is free."""
    monkeypatch.setattr(cli, "_rss_gb", lambda: 9.6)
    monkeypatch.setattr(cli, "memory_gb", lambda: 1500)
    budget = cli.budget_gb(149)
    assert budget == pytest.approx(139.4)
    assert cli.admit([151.02, 16], [], jobs=8, budget=budget) == [0]
    assert cli.admit([137.81], [], jobs=8, budget=budget) == [0] and cli.admit([16], [137.81], jobs=8, budget=budget) == []
    monkeypatch.setattr(cli, "memory_gb", lambda: 200)
    assert cli.budget_gb(None) == 200 and cli.budget_gb(560) == 200
