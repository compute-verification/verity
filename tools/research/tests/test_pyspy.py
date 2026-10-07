"""py-spy dumps of a run's Python processes (research.telemetry.pyspy): the runner's stall watch writes one to
logs/stall-<stamp>.txt when the run dir stops changing, once per stall, and `research inspect --py-spy` prints one for a live run,
locally or read on its machine by the script it sends. A fake /proc under tmp_path, a fake clock and a fake py-spy."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from waits import Gate

from research import remote
from research.cli import main
from research.store import custody
from research.telemetry import pyspy
from research.telemetry import run as trun
from research.telemetry.events import Emitter

STACK = "Thread 7 (idle): \"MainThread\"\n    wait (threading.py:320)\n"


def _proc(proc: Path, pid: int, ppid: int, comm: str, exe: str | None = None, start: int = 1000) -> None:
    d = proc / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} " + " ".join(["0"] * 15) + f" {start} 0 0\n")
    if exe:
        (d / "exe").symlink_to(exe)


@pytest.fixture
def proc(tmp_path):
    p = tmp_path / "proc"
    _proc(p, 40, 1, "python3")
    _proc(p, 41, 40, "bash", "/usr/bin/bash")
    _proc(p, 42, 41, "pt_data_worker", "/usr/bin/python3.12")      # a Python process renamed: its exe says what it is
    _proc(p, 43, 41, "nvidia-smi")
    _proc(p, 50, 1, "python3")                                      # not the run's
    return p


def test_the_python_processes_of_the_workloads_tree(proc):
    assert pyspy.python_tree(40, 1000, proc) == [(40, "python3"), (42, "pt_data_worker")]
    assert pyspy.python_tree(40, 999, proc) == []                   # the pid was reused
    assert pyspy.python_tree(77, None, proc) == []


def test_dump_runs_py_spy_per_process_and_says_what_failed():
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        if argv[3] == "2":
            raise subprocess.TimeoutExpired(argv, kw["timeout"])
        if argv[3] == "3":
            return subprocess.CompletedProcess(argv, 1, b"", b"Permission denied (os error 13)")
        return subprocess.CompletedProcess(argv, 0, STACK.encode(), b"")

    assert pyspy.dump([(1, "python3")], run=run, which=lambda _: None) is None
    got = pyspy.dump([(1, "python3"), (2, "python3"), (3, "python3")], run=run, which=lambda _: "/x/py-spy")
    assert seen[0] == ["/x/py-spy", "dump", "--pid", "1", "--nonblocking"]
    assert [r["rc"] for r in got] == [0, None, 1]
    text = pyspy.render(got)
    assert text.startswith(f"== pid 1 (python3)\n{STACK}== pid 2 (python3)\n[py-spy took over 30 s]\n")
    assert "[py-spy rc 1] Permission denied (os error 13)" in text
    assert pyspy.render([]) == "no Python process in the workload's tree\n"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _events(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]


def _watch(d: Path, monkeypatch, *, which=lambda _: "/x/py-spy"):
    clock, stamps = Clock(), iter(["20261007T030000Z", "20261007T050000Z"])
    monkeypatch.setattr(trun, "clock", clock)
    (d / "stdout.log").write_text("loading\n")
    run = lambda argv, **kw: subprocess.CompletedProcess(argv, 0, STACK.encode(), b"")
    w = trun._StallWatch(d, Emitter(d, proc="tele-run"), "run", 900.0, [], lambda: [(7, "python3")], run=run, which=which,
                         stamp=lambda: next(stamps))
    return w, clock


def _at(w, clock, t: float) -> None:
    clock.now = t
    w.poll()


def test_a_stalled_run_is_dumped_once_and_again_only_after_it_progressed(tmp_path, monkeypatch):
    d = tmp_path / "r1"
    d.mkdir()
    w, clock = _watch(d, monkeypatch)
    for t in (30, 60, 600):
        (d / "resources.jsonl").write_text(f"{t}\n" * int(t))      # the recorder's own files are not the workload's progress
        (d / "status.json").write_text(json.dumps({"t": t}))
        _at(w, clock, t)
    assert not (d / "logs").exists()
    _at(w, clock, 900)
    assert (d / "logs" / "stall-20261007T030000Z.txt").read_text() == f"# no progress in {d} for 900 s\n== pid 7 (python3)\n{STACK}"
    (m,) = [e for e in _events(d) if e.get("name") == "stall_dump"]
    assert (m["idle_s"], m["path"], m["pids"], m["failed"]) == (900.0, "logs/stall-20261007T030000Z.txt", [7], [])
    for t in (930, 5000):
        _at(w, clock, t)
    assert len([e for e in _events(d) if e.get("name") == "stall_dump"]) == 1
    (d / "out").mkdir()
    (d / "out" / "step.json").write_text("{}")
    _at(w, clock, 5030)
    _at(w, clock, 5900)
    assert len([e for e in _events(d) if e.get("name") == "stall_dump"]) == 1
    _at(w, clock, 5930)
    assert (d / "logs" / "stall-20261007T050000Z.txt").exists()
    assert [e["idle_s"] for e in _events(d) if e.get("name") == "stall_dump"] == [900.0, 900.0]


def test_no_py_spy_is_said_once(tmp_path, monkeypatch):
    d = tmp_path / "r1"
    d.mkdir()
    w, clock = _watch(d, monkeypatch, which=lambda _: None)
    _at(w, clock, 900)
    (d / "stdout.log").write_text("more\n")
    _at(w, clock, 930)
    _at(w, clock, 1830)
    (m,) = [e for e in _events(d) if e.get("name", "").startswith("stall_dump")]
    assert (m["name"], m["reason"]) == ("stall_dump_skipped", "py-spy is not on PATH")
    assert not (d / "logs").exists()


def test_tele_run_dumps_its_own_workload(tmp_path, monkeypatch):
    """The harness's watch is wired to the workload it started: the pid it dumps is status.json's, found in the real /proc. Every
    reading of the clock is 300 s later; the fake dump releases the workload."""
    d, gate = tmp_path / "r1", Gate(tmp_path / "gate")
    t, dumped = [0.0], []

    def clock():
        t[0] += 300
        return t[0]

    def dump(procs, **kw):
        dumped.append(procs)
        gate.release()
        return [{"pid": p, "comm": c, "rc": 0, "text": STACK} for p, c in procs]

    monkeypatch.setattr(trun, "clock", clock)
    monkeypatch.setattr(trun.pyspy, "dump", dump)
    try:
        assert trun.main(["--attempt-dir", str(d), "--stage", "t", "--no-sampler", "--no-classify", "--",
                          sys.executable, "-c", Gate.WAIT, str(gate.path)]) == 0
    finally:
        gate.close()
    pid = json.loads((d / "status.json").read_text())["pid"]
    assert [p for p, _ in dumped[0]] == [pid]
    (m,) = [e for e in _events(d) if e.get("name") == "stall_dump"]
    assert (d / m["path"]).read_text().endswith(STACK)


def test_a_failing_watch_does_not_stop_the_workload(tmp_path, monkeypatch):
    d, gate = tmp_path / "r1", Gate(tmp_path / "gate")
    t = [0.0]

    def clock():
        t[0] += 300
        return t[0]

    def dump(procs, **kw):
        gate.release()
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trun, "clock", clock)
    monkeypatch.setattr(trun.pyspy, "dump", dump)
    try:
        assert trun.main(["--attempt-dir", str(d), "--stage", "t", "--no-sampler", "--no-classify", "--",
                          sys.executable, "-c", Gate.WAIT, str(gate.path)]) == 0
    finally:
        gate.close()
    (m,) = [e for e in _events(d) if e.get("name") == "stall_watch_failed"]
    assert "No space left" in m["error"]


def test_custody_keeps_the_dumps_and_compare_does_not_read_them(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "stall-20261007T030000Z.txt").write_text(STACK)
    (tmp_path / "big.bin").write_bytes(b"x" * 10)
    kept, trimmed = custody.select(tmp_path, cap=0)
    assert kept == ["logs/stall-20261007T030000Z.txt"] and [t["path"] for t in trimmed] == ["big.bin"]
    assert custody.is_run_file("logs/stall-20261007T030000Z.txt") and not custody.is_run_file("logs/train.txt")


def _fake_py_spy(bin_dir: Path) -> None:
    bin_dir.mkdir()
    (bin_dir / "py-spy").write_text(f"#!/bin/sh\nprintf '%s' '{STACK}'\n")
    (bin_dir / "py-spy").chmod(0o755)


def test_inspect_py_spy_prints_a_local_live_runs_stacks(proc, tmp_path, monkeypatch, capsys):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "job.json").write_text("{}")
    (run / "status.json").write_text(json.dumps({"state": "running", "pid": 40, "pgid": 40, "starttime_ticks": 1000}))
    monkeypatch.setenv("RESEARCH_PROC_ROOT", str(proc))
    monkeypatch.setenv("PATH", str(tmp_path / "nobin"))
    assert main(["inspect", "r1", "--runs-dir", str(run.parent), "--py-spy"]) == 3
    assert "py-spy is not on PATH" in capsys.readouterr().err
    _fake_py_spy(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert main(["inspect", "r1", "--runs-dir", str(run.parent), "--py-spy"]) == 0
    assert capsys.readouterr().out == f"== pid 40 (python3)\n{STACK}== pid 42 (pt_data_worker)\n{STACK}"
    (run / "status.json").write_text(json.dumps({"state": "done", "pid": 40}))
    assert main(["inspect", "r1", "--runs-dir", str(run.parent), "--py-spy"]) == 1
    assert "r1 is done: no live workload to dump" in capsys.readouterr().err


def test_inspect_py_spy_reads_a_remote_run_with_the_script_it_sends(proc, tmp_path, monkeypatch, capsys):
    """As the machine would run it (`python3 - RUN_DIR`, the file on stdin), with ssh replaced by a local subprocess: what is sent
    must run on its own, outside the package."""
    machine_run = tmp_path / "machine" / "r1"
    machine_run.mkdir(parents=True)
    (machine_run / "status.json").write_text(json.dumps({"state": "running", "pid": 40, "starttime_ticks": 1000}))
    local = tmp_path / "control" / "r1"
    local.mkdir(parents=True)
    (local / "status.json").write_text(json.dumps({"state": "running"}))
    (local / "remote.json").write_text(json.dumps({"machine": "vy-nebius-2", "host": "h", "port": 22, "remote_run_dir": str(machine_run)}))
    _fake_py_spy(tmp_path / "bin")
    sent, path = [], [str(tmp_path / "bin")]

    def ssh_run(ssh, argv, *, stdin=None, timeout=120):
        sent.append(argv)
        return subprocess.run([sys.executable, "-", *argv[2:]], input=stdin, capture_output=True, timeout=timeout, cwd=tmp_path, check=False,
                              env={"RESEARCH_PROC_ROOT": str(proc), "PATH": os.pathsep.join(path)})

    monkeypatch.setattr(remote, "ssh_run", ssh_run)
    assert main(["inspect", "r1", "--runs-dir", str(local.parent), "--py-spy"]) == 0
    assert sent == [["python3", "-", str(machine_run)]]
    assert capsys.readouterr().out == f"== pid 40 (python3)\n{STACK}== pid 42 (pt_data_worker)\n{STACK}"
    path[:] = [str(tmp_path / "nobin")]
    assert main(["inspect", "r1", "--runs-dir", str(local.parent), "--py-spy"]) == 1
    assert "no dump on vy-nebius-2: rc 3 py-spy is not on PATH" in capsys.readouterr().err
    (machine_run / "status.json").write_text(json.dumps({"state": "done", "pid": 40}))
    assert main(["inspect", "r1", "--runs-dir", str(local.parent), "--py-spy"]) == 1
    assert "its status.json says done: no live workload to dump" in capsys.readouterr().err
