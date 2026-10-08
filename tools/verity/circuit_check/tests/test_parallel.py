"""`cli.parallel`, the worker pool behind `circuit-check --jobs`, on its own: `test_circuit_check` builds every target at
collection, which this needs none of."""
from __future__ import annotations

import os
from concurrent.futures.process import BrokenProcessPool

import pytest

from circuit_check import cli


def test_every_item_is_yielded_once():
    assert sorted(cli.parallel(abs, [-1, -2, -3], jobs=2)) == [1, 2, 3]


def _allocate(mb: int) -> float:
    block = bytearray(mb << 20)
    block[::4096] = b"x" * len(block[::4096])
    return cli.peak_gb()


def _die_on_7(x: int) -> int:
    if x == 7:
        os._exit(9)
    return x


def test_a_target_s_peak_is_its_own():
    """The peak a worker reports for an item is that item's: a 600 MB allocation shows, and the small item after it in the same
    worker does not inherit it (`reset_peak`)."""
    big, small = cli.parallel(_allocate, [600, 1], jobs=1)
    assert big >= 0.55
    assert small < big - 0.4


def test_a_worker_that_dies_is_named_and_the_rest_still_run():
    got = list(cli.parallel(_die_on_7, [1, 7, 2, 3], jobs=2))
    killed = [g for g in got if isinstance(g, cli.Killed)]
    assert [k["item"] for k in killed] == [7] and killed[0]["peak_gb"] > 0
    assert sorted(g for g in got if not isinstance(g, cli.Killed)) == [1, 2, 3]


def test_a_death_no_status_file_accounts_for_still_fails_the_run(monkeypatch):
    monkeypatch.setattr(cli, "_tracked", _exit_untracked)
    with pytest.raises(BrokenProcessPool):
        list(cli.parallel(abs, [1, 2], jobs=2))


def _exit_untracked(fn, item):
    os._exit(3)


def test_fresh_gives_each_item_a_worker_of_its_own():
    """The heavy targets' phase: under a budget only one fits, and each runs in a new process."""
    pids = list(cli.parallel(_pid, [1, 2, 3], jobs=3, weight=lambda _: 3, budget=4, fresh=True))
    assert len(set(pids)) == 3


def _pid(_: int) -> int:
    return os.getpid()


def test_jobs_0_takes_one_worker_per_128_gb_the_tightest_cgroup_leaves(tmp_path):
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text("MemTotal: 2000000000 kB\nMemAvailable: 1600000000 kB\n")
    (tmp_path / "proc/self").mkdir()
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice/user-1001.slice/session-7.scope\n")
    scope = tmp_path / "sys/fs/cgroup/user.slice/user-1001.slice/session-7.scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text("max\n")
    (scope.parent / "memory.max").write_text(str(1600 << 30))
    (scope.parent / "memory.current").write_text(str(1300 << 30))
    assert cli.memory_gb(tmp_path) == 300
    assert cli.auto_jobs(tmp_path) == min(2, len(os.sched_getaffinity(0)))
    (scope.parent / "memory.current").write_text(str(1590 << 30))
    assert cli.auto_jobs(tmp_path) == 1
