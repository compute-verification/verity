"""Memory admission on a shared machine (research.mem): a fake cgroup v2 dir and a fake /proc, the fleet's ramlock convention, and
real launch_request runs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from research import mem as M
from research import remote
from research import runs as R
from research.cli import main
from research.store import attempt as A
from waits import FakeClock, Gate, wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
GB = 10 ** 9


def _dead_pid() -> int:
    p = subprocess.Popen([PY, "-c", "pass"])
    p.wait()
    return p.pid


@pytest.fixture
def machine(tmp_path, monkeypatch):
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.max").write_text(f"{100 * GB}\n")
    (cg / "memory.current").write_text(f"{30 * GB}\n")
    (cg / "memory.stat").write_text(f"anon {18 * GB}\nfile {12 * GB}\ninactive_file {10 * GB}\n")
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal:       924131384 kB\nMemAvailable:   731179236 kB\n")
    monkeypatch.setenv("RESEARCH_CGROUP_DIR", str(cg))
    monkeypatch.setenv("RESEARCH_PROC_ROOT", str(proc))
    ram = tmp_path / "ramlock"
    ram.mkdir()
    (ram / "_coordinator_notice.json").write_text(json.dumps({"owner": "coordinator", "gb": 0, "note": "every job > 20 GB writes a file"}))
    (ram / "p2p4-addr-equiv.json").write_text(json.dumps({"owner": "p2p4", "gb": 30, "started": "t", "note": "no pid"}))
    (ram / "integrator-gone.json").write_text(json.dumps({"owner": "integrator", "gb": 50, "pid": _dead_pid()}))
    (ram / "torn.json").write_text("{")
    return {"cg": cg, "proc": proc, "ram": ram, "root": tmp_path / "root"}


def test_the_reading_is_the_cgroup_not_the_host_and_reservations_count_what_they_have_not_yet_used(machine):
    s = M.snapshot(machine["ram"])
    assert (s["limit_gb"], s["in_use_gb"], s["cgroup"]) == (100, 20, "v2"), "memory.max, and memory.current minus inactive_file"
    assert s["reserved_gb"] == 30 and s["available_gb"] == 50
    why = {x["file"]: x for x in s["reservations"]}
    assert why["integrator-gone.json"]["live"] is False and "gone" in why["integrator-gone.json"]["why"]
    assert why["torn.json"]["live"] is False and "unreadable" in why["torn.json"]["why"]
    # a live pid that already holds 20 of its 50 GB is in `in_use` for those 20: only 30 are still to come
    me = machine["proc"] / str(os.getpid())
    me.mkdir()
    (me / "stat").write_text(f"{os.getpid()} (py thon) S 1 1 1\n")
    (me / "status").write_text(f"Name:\tpython\nVmRSS:\t{20 * GB // 1024} kB\n")
    (machine["ram"] / "mine.json").write_text(json.dumps({"owner": "me", "gb": 50, "pid": os.getpid()}))
    s = M.snapshot(machine["ram"])
    assert s["reserved_gb"] == 60 and s["available_gb"] == 20
    text = M.describe(s, 25)
    assert text.startswith("asked 25 GB; available 20 GB = limit 100 (v2 ") and "mine.json 30/50 GB (me; pid" in text
    assert "not counted: integrator-gone.json (pid" in text


def test_a_run_in_its_own_scope_reads_the_slice_above_it_when_that_has_less_headroom(machine, monkeypatch):
    scope = machine["cg"] / "session-7.scope"          # a Nebius node: each run's own scope under the user slice's cap
    scope.mkdir()
    (scope / "memory.max").write_text(f"{600 * GB}\n")
    (scope / "memory.current").write_text("0\n")
    (scope / "memory.stat").write_text("inactive_file 0\n")
    monkeypatch.setenv("RESEARCH_CGROUP_DIR", str(scope))
    r = M.reading()
    assert (r["limit"], r["in_use"], r["source"]) == (100 * GB, 20 * GB, str(machine["cg"]))
    (machine["cg"] / "memory.max").write_text("max\n")
    assert M.reading()["source"] == str(scope)


def test_without_a_finite_cgroup_limit_the_reading_is_meminfo(machine):
    (machine["cg"] / "memory.max").write_text("max\n")
    r = M.reading()
    assert r["cgroup"] is None and r["limit"] == 924131384 * 1024 and r["in_use"] == (924131384 - 731179236) * 1024


def _request(machine, run_id: str, gb: float, *, gate: Gate | None = None, wait: float = 0.0) -> Path:
    root = machine["root"]
    (root / "requests" / run_id).mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": str(root),
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler"], "command": [PY, "-c", Gate.WAIT, str(gate.path)] if gate else [PY, "-c", "pass"],
           "exclusive": False, "mem_gb": gb, "mem_wait_s": wait, "ramlock_dir": str(machine["ram"]),
           "cost": {"provider": "runpod", "pod_id": "p", "rate_usd_per_hour": 0.1, "rate_observed_utc": "t", "currency": "USD", "reason": None},
           "remote": {"machine": "m", "pod_id": "p", "root": str(root)}}
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p


def _wait(run_dir: Path) -> dict:
    wait_runner(run_dir)
    return R.summarize(run_dir)


def test_launch_request_admits_reserves_refuses_by_name_and_the_runner_releases(machine, tmp_path):
    gate = Gate(tmp_path / "gate")                                 # "big" runs until the test has seen "more" refused
    try:
        first = remote.launch_request(_request(machine, "big", 40, gate=gate))
        assert first["started"], first
        res = machine["ram"] / "research-big.json"
        doc = json.loads(res.read_text())
        assert doc["owner"] == "research" and doc["gb"] == 40 and doc["pid"] == first["runner_pid"] and doc["run_id"] == "big"

        second = remote.launch_request(_request(machine, "more", 20))
        assert second["started"] is False and second["fault"] == "MemoryNotAvailable"
        refused = json.loads((machine["root"] / "runs" / "more" / "refused.json").read_text())
        assert refused["fault"] == "MemoryNotAvailable" and "asked 20 GB; available 10 GB" in refused["reason"] and "research-big.json 40/40 GB" in refused["reason"]
        assert not (machine["ram"] / "research-more.json").exists(), "a refused run holds no reservation"
        gate.release()
        run_dir = machine["root"] / "runs" / "big"
        assert _wait(run_dir)["state"] == "done", (run_dir / "launcher.log").read_text()
    finally:
        gate.close()
    assert not res.exists(), "the runner removes its reservation at exit"
    mem = json.loads((run_dir / "job.json").read_text())["run"]["remote"]["mem"]
    assert mem["gb"] == 40 and mem["available_gb"] == 50 and mem["reservation"] == str(res)
    assert A.build(run_dir)["execution"]["mem_gb_reserved"] == 40
    assert "RESEARCH_MEM_RESERVATION" not in (run_dir / "job.json").read_text(), "the reservation path is the runner's, not the workload's"

    assert remote.launch_request(_request(machine, "after", 20))["started"], "its memory is available again"
    _wait(machine["root"] / "runs" / "after")


def test_mem_wait_admits_when_memory_frees_up_within_the_wait(machine, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(M, "time", clock.module)
    blocker = machine["ram"] / "someone.json"
    blocker.write_text(json.dumps({"owner": "someone", "gb": 45}))
    clock.at(5.0, blocker.unlink)
    out = M.admit(20, ramlock_dir=machine["ram"], run_id="w", run_dir=machine["root"] / "w", wait_s=10, poll_s=1)
    assert out["waited_s"] == 5.0 and out["available_gb"] == 50
    blocker.write_text(json.dumps({"owner": "someone", "gb": 45}))
    with pytest.raises(remote.MemoryNotAvailable, match="waited 3s"):
        M.admit(20, ramlock_dir=machine["ram"], run_id="w2", run_dir=machine["root"] / "w2", wait_s=3, poll_s=1)


def test_two_simultaneous_admissions_cannot_both_take_the_last_memory(machine):
    got, lost = [], []
    barrier = threading.Barrier(2)

    def one(i: int) -> None:
        barrier.wait()
        try:
            got.append(M.admit(30, ramlock_dir=machine["ram"], run_id=f"c{i}", run_dir=machine["root"] / f"c{i}"))
        except remote.MemoryNotAvailable as e:
            lost.append(str(e))

    ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(got) == 1 and len(lost) == 1 and "available 20 GB" in lost[0]


def test_a_terminal_run_dirs_reservation_is_stale(machine):
    rd = machine["root"] / "runs" / "ended"
    rd.mkdir(parents=True)
    (rd / "status.json").write_text(json.dumps({"state": "failed"}))
    (machine["ram"] / "research-ended.json").write_text(json.dumps({"owner": "research", "gb": 40, "pid": os.getpid(), "run_id": "ended", "run_dir": str(rd)}))
    assert M.snapshot(machine["ram"])["available_gb"] == 50


def test_cli_mem_gb_needs_a_machine_and_research_mem_prints_the_reading(machine, capsys):
    assert main(["run", "--mem-gb", "4", "--", "true"]) == 2
    assert "--mem-gb admits a run on a shared machine" in capsys.readouterr().err
    assert main(["mem", "--ramlock", str(machine["ram"]), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["available_gb"] == 50
