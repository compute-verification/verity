"""a5's Lean checks beside the sessions (`a5.LeanPool`): up to its workers at once, each starting in the order submitted once
its memory estimate fits the budget beside the running checks' and the node's free memory, one always starting with none
running, and the results in the order submitted; the budget and the estimate (`lean_budget`, `lean_need`); and the run's
shared archive (`a0.archiving`), under which one circuit is put once."""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a0  # noqa: E402
import a5  # noqa: E402

GB = 2**30


class Checks:
    """Checks that say when they start and finish only when the test releases them."""

    def __init__(self, n: int):
        self.started = [threading.Event() for _ in range(n)]
        self.release = [threading.Event() for _ in range(n)]

    def __call__(self, i: int) -> int:
        self.started[i].set()
        self.release[i].wait()
        return i


@contextmanager
def pool(checks: Checks, needs: list[float], workers: int = 3, budget: float = 100 * GB,
         free: list[float] | None = None) -> Iterator[a5.LeanPool]:
    """A pool running ``checks`` with estimates ``needs``; every check released before it shuts down, so a failed
    assertion fails rather than waits."""
    free = free if free is not None else [1000 * GB]
    p = a5.LeanPool(workers, budget, lambda: free[0])
    try:
        for i, need in enumerate(needs):
            p.submit(checks, i, need)
        yield p
    finally:
        for e in checks.release:
            e.set()
        p.ex.shutdown(wait=True)


def test_checks_run_beside_each_other_and_come_back_in_the_order_submitted():
    c = Checks(3)
    with pool(c, [GB] * 3) as p:
        for e in c.started:
            e.wait()
        for e in reversed(c.release):
            e.set()
        assert p.results() == [0, 1, 2]


@pytest.mark.parametrize("workers", [1, 2])
def test_no_more_checks_run_than_workers(workers):
    c = Checks(workers + 1)
    with pool(c, [GB] * (workers + 1), workers=workers) as p:
        for e in c.started[:workers]:
            e.wait()
        assert not c.started[workers].is_set()
        c.release[0].set()
        c.started[workers].wait()
        for e in c.release:
            e.set()
        assert p.results() == list(range(workers + 1))


def test_a_check_starts_only_once_its_estimate_fits_the_budget_beside_the_running_ones():
    c = Checks(3)
    with pool(c, [6 * GB, 6 * GB, GB], budget=10 * GB) as p:
        c.started[0].wait()
        assert not c.started[1].is_set() and not c.started[2].is_set()
        c.release[0].set()
        c.started[1].wait()
        c.started[2].wait()
        c.release[1].set()
        c.release[2].set()
        assert p.results() == [0, 1, 2]


def test_a_check_waits_for_free_memory_and_with_none_running_starts_whatever_its_estimate():
    c, free = Checks(2), [5 * GB]
    with pool(c, [GB, 8 * GB], free=free) as p:
        c.started[0].wait()
        assert not c.started[1].is_set()
        c.release[0].set()
        c.started[1].wait()
        c.release[1].set()
        assert p.results() == [0, 1]


def test_a_check_over_the_budget_runs_alone():
    c = Checks(2)
    with pool(c, [20 * GB, GB], budget=4 * GB) as p:
        c.started[0].wait()
        assert not c.started[1].is_set()
        c.release[0].set()
        c.started[1].wait()
        c.release[1].set()
        assert p.results() == [0, 1]


def test_a_failed_check_frees_its_place_and_its_failure_comes_back():
    def check(i: int) -> int:
        if i == 0:
            raise RuntimeError("lean")
        return i

    with a5.LeanPool(1, 10 * GB, lambda: 1000 * GB) as p:
        p.submit(check, 0, 20 * GB)
        p.submit(check, 1, GB)
        with pytest.raises(RuntimeError, match="lean"):
            p.results()
        assert p.futures[1].result() == 1


def test_the_budget_is_the_audits_memory_less_the_provers_reserve():
    reserve = a5.PROVER_RESERVE_GB * GB
    assert a5.lean_budget(argparse.Namespace(audit_mem_gb=290), lambda: 0) == 290 * GB - reserve
    assert a5.lean_budget(argparse.Namespace(), lambda: 500 * GB) == 500 * GB - reserve
    assert a5.lean_budget(argparse.Namespace(audit_mem_gb=None), lambda: 400 * GB) == 400 * GB - reserve


def test_a_checks_estimate_is_its_circuit_files_bytes_times_the_factor(tmp_path):
    (tmp_path / "circuit.txt").write_bytes(b"x" * 7)
    assert a5.lean_need({"dir": tmp_path}) == 7 * a5.LEAN_BYTES_PER_CIRCUIT_BYTE
    assert a5.lean_need({"dir": tmp_path / "none"}) == 0


def test_the_flags_reach_the_audit():
    ap = argparse.ArgumentParser()
    a5.lean_options(ap)
    assert vars(ap.parse_args([])) == {"lean_workers": a5.LEAN_WORKERS, "audit_mem_gb": None}
    assert vars(ap.parse_args(["--lean-workers", "2", "--audit-mem-gb", "290"])) == {"lean_workers": 2, "audit_mem_gb": 290.0}
    with pytest.raises(SystemExit):
        ap.parse_args(["--lean-workers", "0"])


def test_a_runs_checks_archive_one_circuit_under_one_lock():
    held: dict = {}
    first, again, other = a0.archiving(held, (1, 2)), a0.archiving(held, (1, 2)), a0.archiving(held, (1, 3))
    assert first is again and first is not other
    with a0.archiving(None, (1, 2)):
        pass
