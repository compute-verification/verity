"""stall_probe without the wall clock: a `Sampler` driven sample by sample on given times over a /proc tree and a
cgroup tree of the test's own, whose stat and wchan files are rewritten between samples as a kernel's would read, keeps
each D episode with its kernel functions, its page faults, its stack and its cgroup's state at its start, and closes it
when the thread runs again, exits or the sampling ends; and the probe as the process `serve` starts samples this one."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import stall_probe as sp  # noqa: E402

ORIGIN = 10.0


def _stat(tid: int, state: str, minor: int, major: int) -> str:
    # the command name may hold spaces and parentheses: the fields are read after the last ")"
    return f"{tid} (py (x) y) {state} 1 1 1 0 -1 4194560 {minor} 0 {major} 0 5 6 0 0 20 0 3 0 100\n"


class Proc:
    def __init__(self, root: Path, pid: int) -> None:
        self.root, self.pid = root, pid
        (root / str(pid) / "task").mkdir(parents=True)

    def task(self, tid: int) -> Path:
        return self.root / str(self.pid) / "task" / str(tid)

    def set(self, tid: int, state: str, minor: int = 0, major: int = 0, wchan: str = "0") -> None:
        self.task(tid).mkdir(exist_ok=True)
        (self.task(tid) / "stat").write_text(_stat(tid, state, minor, major))
        (self.task(tid) / "wchan").write_text(wchan)

    def exit(self, tid: int) -> None:
        # a reaped thread's open stat reads nothing (the kernel's ESRCH is taken the same way)
        (self.task(tid) / "stat").write_text("")


@pytest.fixture
def tree(tmp_path):
    proc, cgroups = tmp_path / "proc", tmp_path / "cgroup"
    warden = Proc(proc, 100)
    (proc / "100" / "cgroup").write_text("0::/run/burst\n")
    cg = cgroups / "run" / "burst"
    cg.mkdir(parents=True)
    (cg / "memory.current").write_text("4096\n")
    (cg / "memory.max").write_text("max\n")
    (cg / "memory.events").write_text("low 0\nhigh 0\nmax 3\noom 0\noom_kill 0\n")
    (cg / "memory.pressure").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=12\n"
                                        "full avg10=0.00 avg60=0.00 avg300=0.00 total=7\n")
    return proc, cgroups, warden, cg


def test_a_d_episode_is_kept_with_its_kernel_functions_faults_stack_and_cgroup_and_ends_when_the_thread_runs(tree):
    proc, cgroups, w, cg = tree
    w.set(100, "S", minor=5)
    w.set(101, "R", minor=1)
    (w.task(100) / "stack").write_text("[<0>] __wait_rcu_gp+0x15b/0x190\n[<0>] expand_files+0x1d4/0x2c0\n")
    s = sp.Sampler({"warden": 100, "publisher": 200}, ORIGIN, {100: "ticker"}, proc=proc, cgroups=cgroups)
    s.sample(9.999)
    w.set(100, "S", minor=7)
    s.sample(10.000)
    w.set(100, "D", minor=7, wchan="__wait_rcu_gp")
    s.sample(10.001)
    (cg / "memory.events").write_text("low 0\nhigh 0\nmax 4\noom 0\noom_kill 0\n")
    w.set(100, "D", minor=9, major=1, wchan="folio_wait_bit_common")
    s.sample(10.002)
    w.set(100, "D", minor=9, major=1, wchan="__wait_rcu_gp")
    s.sample(10.003)
    w.set(100, "S", minor=10, major=1)
    w.exit(101)
    s.sample(10.004)
    s.sample(10.020)
    s.finish(10.021)
    doc = s.doc()

    (e,) = doc["episodes"]
    assert e["process"] == "warden" and e["tid"] == 100 and e["label"] == "ticker"
    assert e["start"] == pytest.approx(0.001) and e["ms"] == pytest.approx(3.0) and e["samples"] == 3
    assert e["wchan"] == {"__wait_rcu_gp": 2, "folio_wait_bit_common": 1}
    assert list(e["wchan"]) == ["__wait_rcu_gp", "folio_wait_bit_common"]
    assert e["faults"] == [3, 1]
    assert e["stack"] == ["__wait_rcu_gp+0x15b/0x190", "expand_files+0x1d4/0x2c0"] and doc["stack"] == "readable"
    assert e["cgroup"]["memory.events"]["max"] == 3 and e["cgroup"]["memory.max"] == "max"
    assert e["cgroup"]["memory.current"] == 4096 and e["cgroup"]["memory.pressure"] == {"some": 12, "full": 7}

    assert doc["by_label"] == {"ticker": {"episodes": 1, "max_ms": e["ms"], "wchan": e["wchan"]}}
    threads = {t["tid"]: t for t in doc["threads"]}
    assert threads[100]["states"] == {"S": 4, "D": 3} and threads[100]["samples"] == 7
    assert threads[100]["faults"] == [10 - 7, 1], "page faults count from the first sample at the origin on"
    assert threads[101]["states"] == {"R": 5} and threads[101]["label"] == "thread" and threads[101]["faults"] == [0, 0]
    assert doc["samples"] == 7 and doc["gaps"] == [[pytest.approx(0.004), pytest.approx(16.0)]]
    assert doc["interval_ms"]["max"] == pytest.approx(16.0) and 0.9 <= doc["interval_ms"]["p50"] <= 1.0
    warden = doc["cgroups"]["warden"]
    assert warden["path"] == str(cg) and warden["start"]["memory.events"]["max"] == 3
    assert warden["end"]["memory.events"]["max"] == 4, "the end's state is read at the end"
    assert doc["cgroups"]["publisher"] == {"path": None, "start": None, "end": None}, "a process that isn't there"


def test_a_thread_is_found_at_the_next_scan_and_an_episode_open_at_the_end_or_at_its_exit_is_closed(tree):
    proc, cgroups, w, _ = tree
    w.set(100, "R")
    s = sp.Sampler({"warden": 100}, ORIGIN, proc=proc, cgroups=cgroups)
    s.sample(10.0)
    w.set(102, "D", minor=4, wchan="rwsem_down_read_slowpath")
    s.sample(10.0 + sp.SCAN_S / 2)
    assert 102 not in s.threads, "threads are listed again every SCAN_S, not every sample"
    w.set(100, "D", wchan="__wait_rcu_gp")
    s.sample(10.0 + sp.SCAN_S)
    assert 102 in s.threads
    w.exit(100)
    s.sample(10.0 + sp.SCAN_S + 0.002)
    s.finish(10.0 + sp.SCAN_S + 0.010)
    doc = s.doc()
    ended = {e["tid"]: e for e in doc["episodes"]}
    assert ended[100]["ms"] == pytest.approx(2.0) and ended[100]["wchan"] == {"__wait_rcu_gp": 1}
    assert ended[102]["ms"] == pytest.approx(10.0) and ended[102]["samples"] == 2
    assert ended[102]["wchan"] == {"rwsem_down_read_slowpath": 2} and ended[102]["stack"] is None
    assert doc["stack"] == "unreadable (root only)"
    assert doc["by_label"]["thread"]["episodes"] == 2 and doc["by_label"]["thread"]["max_ms"] == pytest.approx(10.0)
    assert sp.summary(doc)["faults"] == {"warden thread": [0, 0]}


def test_summary_is_what_warden_json_keeps_each_labels_faults_summed():
    doc = {"by_label": {"ticker": {"episodes": 0}}, "samples": 9, "interval_ms": {"p50": 1.0}, "gaps": [[0.1, 6.0]],
           "stack": "not read",
           "threads": [{"process": "warden", "label": "ticker", "faults": [1, 0]},
                       {"process": "warden", "label": "thread", "faults": [2, 1]},
                       {"process": "warden", "label": "thread", "faults": [3, 0]},
                       {"process": "publisher", "label": "thread", "faults": None}]}
    assert sp.summary(doc) == {"by_label": {"ticker": {"episodes": 0}}, "faults": {"warden ticker": [1, 0],
                                                                                 "warden thread": [5, 1]},
                               "samples": 9, "interval_ms": {"p50": 1.0}, "gaps": 1, "stack": "not read"}


def test_the_probe_process_samples_this_process_for_the_samples_asked(tmp_path):
    out = tmp_path / "stalls.json"
    me = threading.get_native_id()
    got = subprocess.run([sys.executable, str(HERE / "stall_probe.py"), "--out", str(out), "--origin", "0",
                          "--samples", "30", f"--label={me}=main", f"test={os.getpid()}"],
                         capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert json.loads(got.stdout.splitlines()[0])["sampling"] >= 1
    doc = json.loads(out.read_text())
    assert doc["samples"] == 30 and doc["period_ms"] == 1.0
    (main,) = [t for t in doc["threads"] if t["tid"] == me]
    assert main["process"] == "test" and main["label"] == "main" and main["samples"] == 30
    assert main["faults"] is not None and set(main["states"]) <= set("RSDZTtXIWPK")
    assert set(doc["interval_ms"]) == {"p50", "p99", "max"} and "test" in doc["cgroups"]


def test_start_returns_once_the_probe_samples_and_stop_gives_its_summary(tmp_path):
    out = tmp_path / "stalls.json"
    probe = sp.start(out, 0.0, {"test": os.getpid()}, {threading.get_native_id(): "main"})
    summary = sp.stop(probe, out)
    assert probe.returncode == 0
    assert summary["samples"] >= 1 and "test main" in summary["faults"]
    assert set(summary) == {"by_label", "faults", "samples", "interval_ms", "gaps", "stack"}
