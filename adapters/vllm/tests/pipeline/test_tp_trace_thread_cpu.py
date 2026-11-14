"""`verity_vllm.pipeline.tp.trace.thread_cpu`: the CPU seconds of a rank's threads, read from stand-in `/proc/self/task/<tid>/stat`
files: summed by the thread's Python name, a thread Python did not start under its kernel name, and a thread that ended between
the listing and the read left out.  No thread is started and no process file is read."""
from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from verity_vllm.pipeline.tp import trace as TRACE

TICK = 100


def _stat(tid: int, comm: str, user: int, system: int) -> str:
    """A `/proc/<pid>/task/<tid>/stat` line: the thread's id, its name in parentheses, then its state and counters, the 14th and 15th
    fields its user and system ticks."""
    return f"{tid} ({comm}) S 1 1 1 0 -1 4194304 10 0 0 0 {user} {system} 0 0 20 0 9 0 12345 1000 100\n"


def _thread_cpu(monkeypatch, stats: Mapping[int, str | BaseException], python: dict[int, str], rank: object = 3) -> dict:
    """`thread_cpu` on a rank whose process lists the threads of ``stats`` (a thread's stat text, or what reading it raises) and
    whose Python threads are ``python`` (native id: name)."""
    def read(path: str) -> str:
        got = stats[int(path.removeprefix("/proc/self/task/").removesuffix("/stat"))]
        if isinstance(got, BaseException):
            raise got
        return got

    asked: list[str] = []

    def listdir(directory: str) -> list[str]:
        asked.append(directory)
        return [str(t) for t in stats]

    def sysconf(name: str) -> int:
        asked.append(name)
        return TICK

    monkeypatch.setattr(TRACE, "os", SimpleNamespace(listdir=listdir, sysconf=sysconf))
    monkeypatch.setattr(TRACE, "Path", lambda p: SimpleNamespace(read_text=lambda: read(p)))
    monkeypatch.setattr(TRACE, "threading", SimpleNamespace(enumerate=lambda: [SimpleNamespace(native_id=t, name=n) for t, n in python.items()]))
    out = TRACE.thread_cpu(SimpleNamespace(_tp2_commit={"rank": rank}))
    assert sorted(asked) == ["/proc/self/task", "SC_CLK_TCK"]
    return out


def test_a_threads_seconds_are_its_user_and_system_ticks_over_the_clock_tick(monkeypatch):
    got = _thread_cpu(monkeypatch, {11: _stat(11, "python3", 150, 50)}, {11: "MainThread"})
    assert got == {"rank": 3, "cpu_s": {"MainThread": 2.0}}


def test_threads_of_one_python_name_add_up_and_each_other_name_has_its_own(monkeypatch):
    stats = {11: _stat(11, "python3", 100, 0), 12: _stat(12, "python3", 30, 20), 13: _stat(13, "python3", 0, 25), 14: _stat(14, "python3", 7, 0)}
    got = _thread_cpu(monkeypatch, stats, {11: "MainThread", 12: "cmt-worker", 13: "cmt-worker", 14: "cmt-hash_0"})
    assert got["cpu_s"] == {"MainThread": 1.0, "cmt-worker": 0.75, "cmt-hash_0": 0.07}


@pytest.mark.parametrize("comm", ["pt_main_thread", "cuda-EvtHandlr", "a (b) c", "x)y"],
                         ids=["a kernel name", "a name with a hyphen", "parentheses and spaces", "a closing parenthesis"])
def test_a_thread_python_did_not_start_is_named_by_its_kernel_name(monkeypatch, comm):
    got = _thread_cpu(monkeypatch, {11: _stat(11, "python3", 100, 0), 21: _stat(21, comm, 40, 10)}, {11: "MainThread"})
    assert got["cpu_s"] == {"MainThread": 1.0, f"native:{comm}": 0.5}


def test_native_threads_of_one_kernel_name_add_up(monkeypatch):
    got = _thread_cpu(monkeypatch, {21: _stat(21, "pool", 40, 10), 22: _stat(22, "pool", 25, 25)}, {})
    assert got["cpu_s"] == {"native:pool": 1.0}


@pytest.mark.parametrize("gone", [FileNotFoundError(2, "No such file or directory"), ProcessLookupError(3, "No such process"),
                                  PermissionError(13, "Permission denied")], ids=["no such file", "no such process", "not permitted"])
def test_a_thread_that_ended_between_the_listing_and_the_read_is_left_out(monkeypatch, gone):
    stats = {11: _stat(11, "python3", 100, 0), 12: gone, 13: _stat(13, "python3", 50, 0)}
    got = _thread_cpu(monkeypatch, stats, {11: "MainThread", 12: "cmt-worker", 13: "cmt-hash_0"})
    assert got["cpu_s"] == {"MainThread": 1.0, "cmt-hash_0": 0.5}


def test_a_read_that_fails_another_way_is_not_taken_for_a_thread_that_ended(monkeypatch):
    with pytest.raises(ValueError, match="not a stat line"):
        _thread_cpu(monkeypatch, {11: ValueError("not a stat line")}, {11: "MainThread"})


def test_a_process_whose_every_thread_ended_has_no_seconds(monkeypatch):
    assert _thread_cpu(monkeypatch, {11: FileNotFoundError(2, "gone")}, {11: "MainThread"}) == {"rank": 3, "cpu_s": {}}


def test_the_rank_is_the_workers_own_as_a_number(monkeypatch):
    assert _thread_cpu(monkeypatch, {}, {}, rank="5") == {"rank": 5, "cpu_s": {}}
