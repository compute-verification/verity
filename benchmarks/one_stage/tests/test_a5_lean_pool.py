"""a5's Lean checks beside the sessions (`a5.LeanPool`): up to its workers at once, each starting in the order submitted once
its memory estimate fits the budget beside the running checks' and the running session's and the node's free memory, one
always starting with no check and no session running, a session starting once the running checks leave it room, and the
results in the order submitted; the budget and the estimates (`audit_memory`, `lean_need`, `session_need`); and the run's
shared archive (`a0.archiving`), under which one circuit is put once."""

from __future__ import annotations

import argparse
import inspect
import json
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


def test_a_reading_that_raises_in_a_checks_gate_is_its_outcome_and_the_next_check_takes_its_turn():
    """Beside a running session, the reading of free memory raises the first time a check's gate asks it (a cgroup file
    that does not read as the kernel writes it) and reads again after. That check's outcome is the error, the next
    check runs in its turn, and leaving the pool returns. A gate that kept the failed check's turn would leave the next
    check waiting for ever, and leaving the pool with it."""
    reads: list[int] = []

    def available() -> float:
        reads.append(len(reads))
        if len(reads) == 2:
            raise ValueError("memory.max: 'garbage' is neither a number nor max")
        return 100 * GB

    with a5.LeanPool(2, 100 * GB, available) as p:
        with p.session(GB):
            p.submit(lambda i: i, 0, GB)
            p.submit(lambda i: i, 1, GB)
            assert p.futures[1].result() == 1
            assert isinstance(p.futures[0].exception(), ValueError)
        with pytest.raises(ValueError, match="garbage"):
            p.results()
    assert len(reads) == 3 and (p.next, p.running, p.used) == (2, 0, 0)


def test_a_check_starts_beside_the_running_session_only_within_the_budget_and_else_once_it_ends():
    c = Checks(3)
    with pool(c, [], budget=10 * GB) as p:
        with p.session(6 * GB):
            p.submit(c, 0, 4 * GB)
            c.started[0].wait()
            p.submit(c, 1, GB)
            assert not c.started[1].is_set()
        c.started[1].wait()
        c.release[0].set()
        c.release[1].set()
        with p.session(6 * GB):
            p.submit(c, 2, 20 * GB)
            assert not c.started[2].is_set()
        c.started[2].wait()
        c.release[2].set()
        assert p.results() == [0, 1, 2]


def test_a_session_starts_once_the_running_checks_leave_its_estimate_within_the_budget():
    c, entered = Checks(2), threading.Event()

    def session() -> None:
        with p.session(6 * GB):
            entered.set()

    with pool(c, [4 * GB, 4 * GB], budget=10 * GB) as p:
        c.started[0].wait()
        c.started[1].wait()
        t = threading.Thread(target=session)
        t.start()
        assert not entered.is_set()
        c.release[0].set()
        entered.wait()
        t.join()
        c.release[1].set()
        assert p.results() == [0, 1]
        with p.session(20 * GB):
            pass


def test_the_budget_is_the_audits_memory():
    assert a5.audit_memory(argparse.Namespace(audit_mem_gb=290), lambda: 0) == 290 * GB
    assert a5.audit_memory(argparse.Namespace(), lambda: 500 * GB) == 500 * GB
    assert a5.audit_memory(argparse.Namespace(audit_mem_gb=None), lambda: 400 * GB) == 400 * GB


def _machine(tmp_path: Path, free_gb: int, here: str | None, levels: dict[str, tuple]) -> tuple[Path, Path, Path]:
    """A machine's three readings as files: MemAvailable of ``free_gb``, the process's cgroup v2 line for ``here`` (None:
    no such file), and under the hierarchy's mount each of ``levels`` (its path from the mount, "" the mount itself)
    with its `memory.max`, its `memory.current` in GB and, where a third is given, a `memory.stat` holding that many GB
    of file cache, split between its two lists, beside as much shared memory; a fourth, a dict, is more of the level's
    files by name (`memory.high`) and more lines of its `memory.stat` by key, each in bytes."""
    meminfo, cgroup, root = tmp_path / "meminfo", tmp_path / "cgroup", tmp_path / "sys"
    meminfo.write_text(f"MemTotal:       99999999 kB\nMemAvailable:   {free_gb * GB // 1024} kB\n")
    if here is not None:
        cgroup.write_text(f"0::{here}\n")
    for path, (limit, used, *cache) in levels.items():
        (root / path).mkdir(parents=True, exist_ok=True)
        (root / path / "memory.max").write_text(f"{limit}\n")
        (root / path / "memory.current").write_text(f"{used * GB}\n")
        for gb in cache[:1]:
            (root / path / "memory.stat").write_text(f"anon {GB}\nfile {2 * gb * GB}\nshmem {gb * GB}\n"
                                                     f"inactive_file {gb * GB - GB // 2}\nactive_file {GB // 2}\n")
        for name, value in (cache[1] if cache[1:] else {}).items():
            if name.startswith("memory."):
                (root / path / name).write_text(f"{value}\n")
            else:
                with (root / path / "memory.stat").open("a") as f:
                    f.write(f"{name} {value}\n")
    return meminfo, cgroup, root


SCOPE = "/job.slice/audit.scope"


@pytest.mark.parametrize("here, levels, room, level", [
    (SCOPE, {"job.slice": ("max", 300), "job.slice/audit.scope": ("max", 200)}, None, None),
    (SCOPE, {"job.slice": ("max", 300), "job.slice/audit.scope": (str(290 * GB), 200)}, 90, SCOPE),
    (SCOPE, {"job.slice": (str(320 * GB), 300), "job.slice/audit.scope": (str(290 * GB), 200)}, 20, "/job.slice"),
    (SCOPE, {"job.slice": (str(320 * GB), 300), "job.slice/audit.scope": ("max", 200)}, 20, "/job.slice"),
    ("/", {"": (str(64 * GB), 60)}, 4, "/"),
    (SCOPE, {"job.slice/audit.scope": (str(100 * GB), 130)}, 0, SCOPE),
    (SCOPE, {"job.slice/audit.scope": (str(290 * GB), 290, 60)}, 60, SCOPE),
    (SCOPE, {"job.slice": (str(320 * GB), 318, 8), "job.slice/audit.scope": (str(290 * GB), 290, 60)}, 10, "/job.slice"),
    (SCOPE, {"": ("max", 1), "other.slice": (str(GB), 1)}, None, None),
    ("/../audit.scope", {"": (str(64 * GB), 14), "audit.scope": (str(GB), 1)}, 50, "/"),
    (None, {"job.slice": (str(GB), 1)}, None, None),
])
def test_what_is_available_is_the_tightest_of_the_nodes_and_each_limit_over_the_process(tmp_path, here, levels, room,
                                                                                        level):
    """`a5.available_memory` on a machine's files, never this one's: with no limit over the process the node's
    MemAvailable; with one at its own level, or a tighter one above it, the least room (`memory.max` less
    `memory.current`, not under 0) and the level it is read at, where a scope that sits at its limit on file cache
    still has that cache's room and its shared memory's none; a limit on a cgroup the process is not under is not
    read, a path that leaves the mount reads the mount's own limit alone, and with no cgroup v2 file it is MemAvailable
    as before. The number carries its terms and names the one that binds."""
    files = _machine(tmp_path, 500, here, levels)
    assert a5.cgroup_room(*files[1:]) == (None if room is None else (room * GB, level))
    free = a5.available_memory(*files)
    assert free == (500 if room is None else room) * GB and isinstance(free, int)
    assert (free.node, free.room, free.level) == (500 * GB, None if room is None else room * GB, level)
    assert free.binds == ("node" if room is None else "cgroup")
    assert a5.memory_terms(free) == {"free": free, "node_mem_available": 500 * GB, "binds": free.binds,
                                     "cgroup_room": None if room is None else room * GB, "cgroup_level": level}
    (tmp_path / "small").mkdir()
    small = a5.available_memory(*_machine(tmp_path / "small", 3, here, levels))
    assert small == (3 if room is None else min(3, room)) * GB
    assert small.binds == ("cgroup" if room is not None and room < 3 else "node")


def test_a_scope_without_a_memory_stat_counts_all_it_holds_and_a_file_that_does_not_read_raises(tmp_path):
    meminfo, cgroup, root = _machine(tmp_path, 500, SCOPE, {"job.slice/audit.scope": (str(290 * GB), 250, 60)})
    assert a5.available_memory(meminfo, cgroup, root) == 100 * GB
    (root / "job.slice/audit.scope/memory.stat").unlink()
    assert a5.available_memory(meminfo, cgroup, root) == 40 * GB
    (root / "job.slice/audit.scope/memory.stat").write_text("inactive_file many\n")
    with pytest.raises(ValueError):
        a5.available_memory(meminfo, cgroup, root)
    (root / "job.slice/audit.scope/memory.current").write_text("\n")
    with pytest.raises(ValueError):
        a5.available_memory(meminfo, cgroup, root)


SCOPE_AT = "job.slice/audit.scope"


@pytest.mark.parametrize("more, room", [
    ({}, 60),
    ({"file_dirty": 10 * GB}, 50), ({"file_writeback": 10 * GB}, 50), ({"file_mapped": 10 * GB}, 50),
    ({"file_dirty": 10 * GB, "file_writeback": 20 * GB, "file_mapped": 5 * GB}, 25),
    ({"file_dirty": 40 * GB, "file_mapped": 40 * GB}, 0),
    ({"memory.high": "max"}, 60), ({"memory.high": 280 * GB}, 50), ({"memory.high": 300 * GB}, 60),
    ({"memory.high": 280 * GB, "file_dirty": 10 * GB}, 40), ({"memory.high": 200 * GB}, 0),
])
def test_the_cgroups_room_errs_low(tmp_path, more, room):
    """A scope limited to 290 GB that holds 290 GB, 60 GB of it file cache. The cache counted free is less its dirty
    pages, its pages under writeback and its mapped pages, each alone and together, and never under nothing; the limit
    is the lesser of `memory.high` and `memory.max` where `memory.high` is a number, and `memory.max` where it is `max`
    or the scope has no such file."""
    files = _machine(tmp_path, 500, SCOPE, {SCOPE_AT: (str(290 * GB), 290, 60, more)})
    assert a5.cgroup_room(*files[1:]) == (room * GB, SCOPE) and a5.available_memory(*files) == room * GB


def test_more_held_back_than_the_cache_holds_counts_no_cache_free_and_takes_nothing_from_the_room(tmp_path):
    """A scope 90 GB under its limit whose dirty and mapped pages come to more than its two file lists: no cache is
    counted free, and the room stays the 90 GB the limit leaves."""
    files = _machine(tmp_path, 500, SCOPE, {SCOPE_AT: (str(290 * GB), 200, 60, {"file_dirty": 40 * GB, "file_mapped": 40 * GB})})
    assert a5.file_cache(files[2] / SCOPE_AT) == 0 and a5.cgroup_room(*files[1:]) == (90 * GB, SCOPE)


def test_a_level_limited_by_memory_high_alone_is_read_and_a_memory_high_that_does_not_read_raises(tmp_path):
    """A slice with no `memory.max` number and a `memory.high` over the scope is a limit like any other, the tightest
    taken; a `memory.high` that is there and isn't a number or `max` raises."""
    files = _machine(tmp_path, 500, SCOPE, {"job.slice": ("max", 300, 0, {"memory.high": 310 * GB}),
                                            SCOPE_AT: (str(290 * GB), 200)})
    assert a5.cgroup_room(*files[1:]) == (10 * GB, "/job.slice")
    (files[2] / SCOPE_AT / "memory.high").write_text("soon\n")
    with pytest.raises(ValueError):
        a5.available_memory(*files)


def _memory_lines(out: str, pool: str) -> list[dict]:
    """The lines a pool printed of its memory, each without the pool's name."""
    said = [json.loads(x[3:])["memory"] for x in out.splitlines() if x.startswith('A5 {"memory"')]
    return [{k: v for k, v in x.items() if k != "pool"} for x in said if x["pool"] == pool]


def test_the_pool_says_the_memory_it_starts_with_and_which_term_binds(tmp_path, capsys):
    """One line as the pool starts: its workers and budget, the node's MemAvailable, the cgroup's room and the level it
    is read at, and which of the two binds; a caller's own reading is the bytes alone."""
    files = _machine(tmp_path, 500, SCOPE, {"job.slice": (str(320 * GB), 318, 8), "job.slice/audit.scope": ("max", 1)})
    with a5.LeanPool(2, 290 * GB, lambda: a5.available_memory(*files)):
        pass
    assert _memory_lines(capsys.readouterr().out, "lean") == [{
        "workers": 2, "budget": 290 * GB, "free": 10 * GB, "node_mem_available": 500 * GB, "cgroup_room": 10 * GB,
        "cgroup_level": "/job.slice", "binds": "cgroup"}]
    (tmp_path / "bare").mkdir()
    bare = _machine(tmp_path / "bare", 7, None, {})
    with a5.LeanPool(1, 290 * GB, lambda: a5.available_memory(*bare)):
        pass
    with a5.LeanPool(1, 290 * GB, lambda: 5 * GB):
        pass
    assert _memory_lines(capsys.readouterr().out, "lean") == [
        {"workers": 1, "budget": 290 * GB, "free": 7 * GB, "node_mem_available": 7 * GB, "cgroup_room": None,
         "cgroup_level": None, "binds": "node"},
        {"workers": 1, "budget": 290 * GB, "free": 5 * GB}]


def test_a_check_held_back_by_memory_is_said_once_with_its_estimate_and_the_term_that_held_it(tmp_path, capsys):
    """A scope with 4 GB of room on a node with 500 GB free: the second check, estimated at 8 GB, waits for the first,
    and the pool says so in one line however often it is asked, naming the cgroup's level; a third held after it adds
    no line. The same wait under a budget names the budget and what runs within it; a pool that never waits says
    nothing but its start. (The test asks the gate itself, so the line is there whether or not the pool's own thread
    has asked yet.)"""
    files = _machine(tmp_path, 500, SCOPE, {"job.slice/audit.scope": (str(64 * GB), 60)})
    c = Checks(3)
    p = a5.LeanPool(3, 100 * GB, lambda: a5.available_memory(*files))
    try:
        for i, need in enumerate([GB, 8 * GB, 9 * GB]):
            p.submit(c, i, need)
        c.started[0].wait()
        for _ in range(3):
            with p.cv:
                assert not p._fits(1, 8 * GB)
        assert not c.started[1].is_set()
        c.release[0].set()
        c.started[1].wait()
        with p.cv:
            assert not p._fits(2, 9 * GB)
        c.release[1].set()
        c.started[2].wait()
    finally:
        for e in c.release:
            e.set()
        p.ex.shutdown(wait=True)
    start, *held = _memory_lines(capsys.readouterr().out, "lean")
    assert start["binds"] == "cgroup" and held == [{
        "waits": "check", "need": 8 * GB, "free": 4 * GB, "node_mem_available": 500 * GB, "cgroup_room": 4 * GB,
        "cgroup_level": SCOPE, "binds": "cgroup"}]

    c = Checks(2)
    with pool(c, [6 * GB, 6 * GB], budget=10 * GB) as p:
        c.started[0].wait()
        for _ in range(2):
            with p.cv:
                assert not p._fits(1, 6 * GB)
        c.release[0].set()
        c.started[1].wait()
    assert _memory_lines(capsys.readouterr().out, "lean")[1:] == [
        {"waits": "check", "need": 6 * GB, "binds": "budget", "budget": 10 * GB, "running": 6 * GB}]

    c = Checks(2)
    with pool(c, [GB, GB]) as p:
        c.started[1].wait()
    assert len(_memory_lines(capsys.readouterr().out, "lean")) == 1


def test_a_session_that_waits_for_the_running_checks_is_said_too(capsys):
    """A session of 5 GB beside a check of 8 GB under a budget of 10 GB waits for the check, and the pool says so once
    before it waits. The check is released as the session reaches its wait, so the session enters either way."""
    c = Checks(1)
    with pool(c, [8 * GB], budget=10 * GB) as p:
        c.started[0].wait()
        wait_for = p.cv.wait_for

        def released(fits):
            c.release[0].set()
            return wait_for(fits)

        p.cv.wait_for = released
        with p.session(5 * GB):
            assert not p.running and p.held == 5 * GB
    assert _memory_lines(capsys.readouterr().out, "lean")[1:] == [
        {"waits": "session", "need": 5 * GB, "binds": "budget", "budget": 10 * GB, "running": 8 * GB}]


def test_a_machine_with_only_cgroup_v1_lines_reads_memavailable_alone(tmp_path):
    meminfo, cgroup, root = _machine(tmp_path, 500, None, {"": (str(GB), 0)})
    cgroup.write_text("12:memory:/job\n1:name=systemd:/job\n")
    assert a5.cgroup_room(cgroup, root) is None and a5.available_memory(meminfo, cgroup, root) == 500 * GB
    meminfo.write_text("MemTotal: 1 kB\n")
    with pytest.raises(OSError, match="MemAvailable"):
        a5.available_memory(meminfo, cgroup, root)


def test_the_audit_reads_this_machines_own_files_by_default():
    """The audit hands `available_memory` to its pools with no argument, so its defaults are the kernel's files."""
    assert [str(p.default) for p in inspect.signature(a5.available_memory).parameters.values()] == [
        "/proc/meminfo", "/proc/self/cgroup", "/sys/fs/cgroup"]


def test_the_estimates_are_the_circuit_files_bytes_times_their_factors(tmp_path):
    (tmp_path / "circuit.txt").write_bytes(b"x" * 7)
    assert a5.lean_need({"dir": tmp_path}) == 7 * a5.LEAN_BYTES_PER_CIRCUIT_BYTE
    assert a5.lean_need({"dir": tmp_path / "none"}) == 0
    assert a5.session_need(tmp_path / "circuit.txt") == 7 * a5.SESSION_BYTES_PER_CIRCUIT_BYTE
    assert a5.session_need(tmp_path / "none") == 0


def test_the_flags_reach_the_audit():
    """With no flag an audit runs one Lean check at a time and its negatives one after another; more of either is its
    command's to ask for."""
    ap = argparse.ArgumentParser()
    a5.lean_options(ap)
    assert a5.LEAN_WORKERS == 1
    assert vars(ap.parse_args([])) == {"lean_workers": 1, "audit_mem_gb": None, "negative_workers": 1}
    assert vars(ap.parse_args(["--lean-workers", "2", "--audit-mem-gb", "290"])) == {
        "lean_workers": 2, "audit_mem_gb": 290.0, "negative_workers": 1}
    with pytest.raises(SystemExit):
        ap.parse_args(["--lean-workers", "0"])


def test_behind_the_zk_server_the_checks_run_one_at_a_time_and_a_flag_above_one_says_it_changes_nothing(capsys):
    """`a5.checks_at_once`: the flag's own number outside the ZK server; one on a proof server, as before; one behind
    the ZK server, with one printed line where the flag asked for more, and none where it asked for one."""
    assert a5.checks_at_once(3, False, False) == 3 and a5.checks_at_once(3, True, False) == 1
    assert a5.checks_at_once(1, False, True) == 1 and capsys.readouterr().out == ""
    assert a5.checks_at_once(3, False, True) == 1 and a5.checks_at_once(2, True, True) == 1
    said = [json.loads(x[3:])["lean_workers"] for x in capsys.readouterr().out.splitlines()]
    assert [(x["asked"], x["at_once"]) for x in said] == [(3, 1), (2, 1)]
    assert said[0]["why"] == ("behind the ZK server the kept verifier process takes one member at a time, so the Lean "
                              "checks run one at a time and --lean-workers 3 changes nothing")


def test_a_runs_checks_archive_one_circuit_under_one_lock():
    held: dict = {}
    first, again, other = a0.archiving(held, (1, 2)), a0.archiving(held, (1, 2)), a0.archiving(held, (1, 3))
    assert first is again and first is not other
    with a0.archiving(None, (1, 2)):
        pass
