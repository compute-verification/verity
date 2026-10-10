"""`cli.parallel`, the worker pool behind `circuit-check --jobs`, on its own: `test_circuit_check` builds every target at
collection, which this needs none of."""
from __future__ import annotations

import os
from concurrent.futures.process import BrokenProcessPool

import pytest

from circuit_check import cli


def test_every_item_is_yielded_once():
    assert sorted(cli.parallel(abs, [-1, -2, -3], jobs=2)) == [1, 2, 3]


def test_a_worker_that_dies_fails_the_run_instead_of_hanging_it():
    with pytest.raises(BrokenProcessPool):
        list(cli.parallel(os._exit, [3, 3], jobs=2))


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
