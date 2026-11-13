"""Memory admission on a shared machine (research.mem): a fake cgroup v2 dir and a fake /proc, the fleet's ramlock convention, and
real launch_request runs."""
from __future__ import annotations

import calendar
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


def _claimed(run_dir: Path) -> Path:
    """A run directory as a launch has it when it asks for memory: made (remote.launch_request's claim)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    s = M.snapshot(machine["ram"])
    assert (s["limit_gb"], s["in_use_gb"], s["available_gb"], s["source"]) == (100, 20, 50, str(machine["cg"]))
    (machine["cg"] / "memory.max").write_text("max\n")
    s = M.snapshot(machine["ram"])
    assert (s["source"], s["available_gb"]) == (str(scope), 570), "the scope is then the machine's: the reservations count against it"


def test_the_machines_reservations_count_against_the_slice_not_the_runs_own_scope(machine, monkeypatch):
    """A run's fresh 900 GiB session scope less 1,219 GB of reservations held in other scopes (two check slots' 601 GB) would
    read -253 GB and refuse every --mem-gb run, with 357 GB left in the 1,600 GiB user slice."""
    G = M.GB
    (machine["cg"] / "memory.max").write_text(f"{1718 * G:.0f}\n")
    (machine["cg"] / "memory.current").write_text(f"{142 * G:.0f}\n")
    (machine["cg"] / "memory.stat").write_text("inactive_file 0\n")
    scope = machine["cg"] / "session-148161.scope"
    scope.mkdir()
    (scope / "memory.max").write_text(f"{966 * G:.0f}\n")
    (scope / "memory.current").write_text("0\n")
    (scope / "memory.stat").write_text("inactive_file 0\n")
    monkeypatch.setenv("RESEARCH_CGROUP_DIR", str(scope))
    for slot in ("a", "b"):
        (machine["ram"] / f"check-{slot}.json").write_text(json.dumps({"owner": "check", "gb": 594.5}))   # 30 + 2 x 594.5 = 1,219
    s = M.snapshot(machine["ram"])
    assert (s["source"], s["reserved_gb"], s["available_gb"]) == (str(machine["cg"]), 1219, 357)
    assert M.admit(128, ramlock_dir=machine["ram"], run_id="r128", run_dir=_claimed(machine["root"] / "r128"))["available_gb"] == 357
    with pytest.raises(remote.MemoryNotAvailable, match=r"asked 300 GB; available 229 GB = limit 1718 \(v2 "):
        M.admit(300, ramlock_dir=machine["ram"], run_id="r300", run_dir=_claimed(machine["root"] / "r300"))
    for f in machine["ram"].glob("*.json"):
        f.unlink()
    s = M.snapshot(machine["ram"])
    assert (s["source"], s["available_gb"]) == (str(scope), 966), "an empty ledger: the run's own cap binds, and says so"
    assert M.describe(s).startswith(f"available 966 GB = limit 966 (v2 {scope}, this run's own) - in use 0; the machine's {machine['cg']} "
                                    "leaves 1576 = 1718 - in use 142 - reserved 0")


def test_without_a_finite_cgroup_limit_the_reading_is_meminfo(machine):
    (machine["cg"] / "memory.max").write_text("max\n")
    (r,) = M.readings()
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


def test_the_workload_reads_its_reservation_as_RESEARCH_RUN_MEM_GB(machine, tmp_path):
    """A reservation sets no memory.max, so suites.py sizing its workers from the cgroup would see the whole node."""
    seen = tmp_path / "seen"
    p = _request(machine, "sized", 12.5)
    req = json.loads(p.read_text())
    req["command"] = [PY, "-c", f"import os; open({str(seen)!r}, 'w').write(os.environ.get('RESEARCH_RUN_MEM_GB', 'unset'))"]
    p.write_text(json.dumps(req))
    assert remote.launch_request(p)["started"]
    assert _wait(machine["root"] / "runs" / "sized")["state"] == "done"
    assert seen.read_text() == "12.5"


def test_mem_wait_admits_when_memory_frees_up_within_the_wait(machine, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(M, "time", clock.module)
    blocker = machine["ram"] / "someone.json"
    blocker.write_text(json.dumps({"owner": "someone", "gb": 45}))
    clock.at(5.0, blocker.unlink)
    out = M.admit(20, ramlock_dir=machine["ram"], run_id="w", run_dir=_claimed(machine["root"] / "w"), wait_s=10, poll_s=1)
    assert out["waited_s"] == 5.0 and out["available_gb"] == 50
    blocker.write_text(json.dumps({"owner": "someone", "gb": 45}))
    with pytest.raises(remote.MemoryNotAvailable, match="waited 3s"):
        M.admit(20, ramlock_dir=machine["ram"], run_id="w2", run_dir=_claimed(machine["root"] / "w2"), wait_s=3, poll_s=1)


def test_two_simultaneous_admissions_cannot_both_take_the_last_memory(machine):
    got, lost = [], []
    barrier = threading.Barrier(2)

    def one(i: int) -> None:
        barrier.wait()
        try:
            got.append(M.admit(30, ramlock_dir=machine["ram"], run_id=f"c{i}", run_dir=_claimed(machine["root"] / f"c{i}")))
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


def _no_pid(machine, run_id: str, gb: float, run_dir: Path, started: str = "2026-10-01T09:00:00Z") -> Path:
    """A run's reservation as admit() writes it: no pid yet, and the run's directory."""
    path = machine["ram"] / f"research-{run_id}.json"
    path.write_text(json.dumps({"owner": "research", "gb": gb, "started": started, "pid": None, "run_id": run_id, "run_dir": str(run_dir), "note": ""}))
    return path


def test_a_reservation_with_no_pid_whose_run_directory_is_gone_is_not_counted_and_its_line_says_how_to_remove_it(machine, monkeypatch):
    """A launch that dies between reserving and starting its runner leaves a reservation that names no pid, so no process's
    end ever ends it.  Once its run's directory is removed nothing holds it, and counted it takes its whole size from every later
    run for as long as its file lies there.  While the directory is there it counts whole, as a run that is starting does, and
    one with no pid that names no directory (written by hand) counts as it did."""
    monkeypatch.setattr(M, "time", FakeClock(calendar.timegm((2026, 10, 19, 9, 0, 0))).module)
    rd = _claimed(machine["root"] / "runs" / "r-gone")
    res = _no_pid(machine, "r-gone", 40, rd)
    s = M.snapshot(machine["ram"])
    row = {x["file"]: x for x in s["reservations"]}
    assert (s["reserved_gb"], s["available_gb"]) == (70, 10) and row[res.name]["live"] and row[res.name]["why"] == "no pid"
    rd.rmdir()
    s = M.snapshot(machine["ram"])
    row = {x["file"]: x for x in s["reservations"]}
    assert (s["reserved_gb"], s["available_gb"]) == (30, 50), "only the hand-written 30 GB, which names no directory"
    assert row["p2p4-addr-equiv.json"]["live"] and row["p2p4-addr-equiv.json"]["why"] == "no pid"
    assert (row[res.name]["live"], row[res.name]["gb"], row[res.name]["outstanding"]) == (False, 40, 0)
    assert row[res.name]["why"] == (f"stale: 40 GB reserved 18 d ago for run r-gone, which names no pid and whose directory {rd} is gone, "
                                    f"so nothing holds it; `rm {res}` removes it")
    assert "; not counted: integrator-gone.json (pid " in M.describe(s) and f", {res.name} ({row[res.name]['why']})" in M.describe(s)
    assert M.admit(45, ramlock_dir=machine["ram"], run_id="next", run_dir=_claimed(machine["root"] / "runs" / "next"))["available_gb"] == 50
    res.write_text(json.dumps({**json.loads(res.read_text()), "started": "t"}))
    (row,) = [x for x in M.reservations(machine["ram"]) if x["file"] == res.name]
    assert not row["live"] and "stale: 40 GB reserved at a time its record doesn't give (started 't') for run r-gone, " in row["why"]
    assert [M._ago(t) for t in ("2026-10-19T08:15:00Z", "2026-10-18T03:00:00Z", "2026-10-17T09:00:00Z")] == ["45 min ago", "30 h ago", "2 d ago"]


def test_a_reservation_with_no_pid_whose_run_ended_or_was_refused_is_not_counted(machine):
    """A reservation with no pid whose run directory is there goes by that run's own record: one that ended, or that the machine
    refused, holds nothing."""
    for run_id, record, doc in (("r-ended", "status.json", {"state": "cancelled"}), ("r-refused", "refused.json", {"reason": "no"})):
        rd = _claimed(machine["root"] / "runs" / run_id)
        (rd / record).write_text(json.dumps(doc))
        _no_pid(machine, run_id, 40, rd)
    s = M.snapshot(machine["ram"])
    assert (s["reserved_gb"], s["available_gb"]) == (30, 50)
    assert {x["file"]: x["why"] for x in s["reservations"] if x["file"].startswith("research-")} == {
        "research-r-ended.json": "run r-ended ended", "research-r-refused.json": "run r-refused ended"}


def test_admit_reserves_only_for_a_run_directory_that_is_there(machine):
    """admit() writes a reservation with no pid, which counts by its run's directory: for a directory that isn't there it would
    write one that counts for nothing, and the next run would be admitted to the same memory."""
    with pytest.raises(ValueError, match="the run directory .*r-unmade isn't there.*make the directory before asking for its memory"):
        M.admit(20, ramlock_dir=machine["ram"], run_id="r-unmade", run_dir=machine["root"] / "runs" / "r-unmade")
    assert not (machine["ram"] / "research-r-unmade.json").exists()


def test_a_run_that_is_starting_keeps_its_reservation_until_its_runners_pid_is_named(machine, tmp_path, monkeypatch):
    """A launch reserves before its runner exists, so its reservation names no pid from admit() until set_pid(), which follows
    the runner's start with nothing to wait on between.  All that time the run's directory is there (the launch claims it before
    it asks for memory), so the reservation counts whole and the next launch can't take the same memory."""
    seen: dict = {}
    set_pid = M.set_pid

    def at_the_end_of_the_window(path, pid) -> None:
        (seen["row"],) = [x for x in M.reservations(machine["ram"]) if x["file"] == Path(path).name]
        seen["doc"], seen["available_gb"] = json.loads(Path(path).read_text()), M.snapshot(machine["ram"])["available_gb"]
        set_pid(path, pid)

    monkeypatch.setattr(M, "set_pid", at_the_end_of_the_window)
    gate = Gate(tmp_path / "gate")
    try:
        first = remote.launch_request(_request(machine, "starting", 40, gate=gate))
        assert first["started"], first
        run_dir = machine["root"] / "runs" / "starting"
        assert seen["doc"]["pid"] is None and seen["doc"]["run_dir"] == str(run_dir)
        assert (seen["row"]["live"], seen["row"]["why"], seen["row"]["outstanding"], seen["available_gb"]) == (True, "no pid", 40 * GB, 10)
        assert json.loads((machine["ram"] / "research-starting.json").read_text())["pid"] == first["runner_pid"]
        gate.release()
        assert _wait(run_dir)["state"] == "done", (run_dir / "launcher.log").read_text()
    finally:
        gate.close()


def test_cli_mem_gb_needs_a_machine_and_research_mem_prints_the_reading(machine, capsys):
    assert main(["run", "--mem-gb", "4", "--", "true"]) == 2
    assert "--mem-gb admits a run on a shared machine" in capsys.readouterr().err
    assert main(["mem", "--ramlock", str(machine["ram"]), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["available_gb"] == 50


def test_a_runs_own_reservation_and_its_check_slots_count_once():
    """A run holding 96 GB from `research run --mem-gb` and 279 GB for the check slot it took is one tree of processes; summed,
    they would refuse a 96 GB job with 390 GB free."""
    G = M.GB
    res = [{"file": "research-r1.json", "live": True, "run_id": "r1", "outstanding": 96 * G},
           {"file": "check-p0.json", "live": True, "run_id": "r1", "outstanding": 279 * G},
           {"file": "research-r2.json", "live": True, "run_id": "r2", "outstanding": 50 * G},
           {"file": "check-t.json", "live": False, "run_id": "r0", "outstanding": 0}]
    assert M.reserved_bytes(res) == (279 + 50) * G


def test_active_file_cache_is_reclaimable_and_not_in_use(machine):
    """930 GB current of which 454 GB active_file (weights already read) and 93 GB inactive_file: counting the active cache as
    used would refuse a run with all 8 GPUs idle."""
    G = int(M.GB)
    (machine["cg"] / "memory.max").write_text(f"{1500 * G}\n")
    (machine["cg"] / "memory.current").write_text(f"{930 * G}\n")
    (machine["cg"] / "memory.stat").write_text(f"anon {366 * G}\nfile {557 * G}\nshmem {11 * G}\nactive_file {454 * G}\ninactive_file {93 * G}\n")
    (r,) = M.readings()
    assert r["in_use"] == (930 - 454 - 93) * G


def _ticket(ram: Path, ns: int, gb: float, run_id: str, pid: int | None = None, **extra) -> Path:
    """A waiter's ticket as admit() writes it; by default this process's, so it is live."""
    pid = pid or os.getpid()
    (ram / M.WAITING).mkdir(exist_ok=True)
    p = ram / M.WAITING / f"{ns}.{pid}.json"
    p.write_text(json.dumps({"run_id": run_id, "gb": gb, "pid": pid, "since": "t", **extra}))
    return p


def _in_line(ram: Path) -> list[str]:
    return [w["run_id"] for w in M.waiters(ram)]


def test_a_big_head_waiter_is_admitted_when_room_frees_and_later_smaller_runs_wait_behind_it(machine, monkeypatch):
    """A run needing 550 GB in one process must not wait while later, smaller runs take the room as it frees.  Here "big" (45 GB) waits; at t=2 there is room for 10 GB runs but not for it: a run without a wait
    is refused naming it, a waiting one is held until it gives up; at t=6 there is room for big, which goes first."""
    clock = FakeClock(1000.0)
    monkeypatch.setattr(M, "time", clock.module)
    ram, rd = machine["ram"], machine["root"]
    blocker = ram / "someone.json"
    blocker.write_text(json.dumps({"owner": "someone", "gb": 45}))            # available 5
    seen: dict = {}

    def room_for_small() -> None:
        blocker.write_text(json.dumps({"owner": "someone", "gb": 20}))        # available 30
        (seen["big"],) = M.waiters(ram)
        with pytest.raises(remote.MemoryNotAvailable) as e:
            M.admit(10, ramlock_dir=ram, run_id="small", run_dir=_claimed(rd / "small"))
        seen["refused"] = str(e.value)
        with pytest.raises(remote.MemoryNotAvailable) as e:
            M.admit(10, ramlock_dir=ram, run_id="small-w", run_dir=_claimed(rd / "small-w"), wait_s=2, poll_s=1)
        seen["held"], seen["after_held"] = str(e.value), _in_line(ram)
        seen["later"] = _ticket(ram, seen["big"]["ns"] + 10, 5, "later")      # a later waiter that fits all along

    clock.at(1002.0, room_for_small)
    clock.at(1006.0, blocker.unlink)                                          # available 50
    out = M.admit(45, ramlock_dir=ram, run_id="big", run_dir=_claimed(rd / "big"), wait_s=60, poll_s=1)

    assert seen["big"]["run_id"] == "big" and seen["big"]["gb"] == 45 and seen["big"]["pid"] == os.getpid()
    assert seen["refused"].startswith("held behind 1 older waiter(s), first come first served (research.mem), oldest first: big 45 GB")
    assert "asked 10 GB; available 30 GB" in seen["refused"] and "--mem-wait S waits for it" in seen["refused"]
    assert seen["held"].startswith("held behind 1 older waiter(s)") and "big 45 GB" in seen["held"] and "waited 2s" in seen["held"]
    assert seen["after_held"] == ["big"], "a waiter that gives up leaves the line"
    assert out["waited_s"] == 6.0 and out["available_gb"] == 50 and (ram / "research-big.json").exists()
    assert _in_line(ram) == ["later"], "the admitted waiter's ticket is gone, the later one still waits"
    with pytest.raises(remote.MemoryNotAvailable, match="held behind 1 older waiter.*later 5 GB"):
        M.admit(5, ramlock_dir=ram, run_id="next", run_dir=_claimed(rd / "next"))      # 5 GB is free, but "later" asked first
    seen["later"].unlink()
    assert M.admit(5, ramlock_dir=ram, run_id="next", run_dir=_claimed(rd / "next"))["available_gb"] == 5


def test_a_dead_or_reused_pids_ticket_holds_no_place_and_is_removed(machine):
    ram = machine["ram"]
    dead = _ticket(ram, 1, 10, "gone", pid=_dead_pid())
    reused = _ticket(ram, 2, 10, "reused", start=-1)                          # this pid, but another process's start time
    assert _in_line(ram) == []
    assert M.admit(20, ramlock_dir=ram, run_id="r", run_dir=_claimed(machine["root"] / "r"))["available_gb"] == 50
    assert not dead.exists() and not reused.exists()


def test_a_waiter_larger_than_the_limit_blocks_nobody_and_takes_no_place(machine, monkeypatch):
    ram = machine["ram"]
    huge = _ticket(ram, 1, 150, "huge")                                       # the limit is 100 GB: it can never fit
    assert M.admit(20, ramlock_dir=ram, run_id="r", run_dir=_claimed(machine["root"] / "r"))["available_gb"] == 50
    assert huge.exists(), "it is live: only ignored, not removed"
    clock = FakeClock()
    monkeypatch.setattr(M, "time", clock.module)
    seen = []
    clock.at(1.0, lambda: seen.append(_in_line(ram)))
    with pytest.raises(remote.MemoryNotAvailable, match="asked 150 GB.*waited 3s"):
        M.admit(150, ramlock_dir=ram, run_id="huge2", run_dir=_claimed(machine["root"] / "huge2"), wait_s=3, poll_s=1)
    assert seen == [["huge"]], "a run asking for more than the limit waits without a ticket"


def test_the_ticket_is_removed_when_the_wait_ends_by_an_exception(machine, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(M, "time", clock.module)
    seen = []

    def interrupt() -> None:
        seen.append(_in_line(machine["ram"]))
        raise KeyboardInterrupt

    clock.at(1.0, interrupt)
    with pytest.raises(KeyboardInterrupt):
        M.admit(60, ramlock_dir=machine["ram"], run_id="w", run_dir=_claimed(machine["root"] / "w"), wait_s=10, poll_s=1)
    assert seen == [["w"]] and list((machine["ram"] / M.WAITING).iterdir()) == []


def test_research_mem_lists_the_waiters_oldest_first(machine, capsys):
    _ticket(machine["ram"], 20, 30, "second")
    _ticket(machine["ram"], 10, 60, "first")
    assert main(["mem", "--ramlock", str(machine["ram"])]) == 0
    assert f"; waiting, oldest first: first 60 GB (pid {os.getpid()}, since t), second 30 GB" in capsys.readouterr().out
    assert main(["mem", "--ramlock", str(machine["ram"]), "--json"]) == 0
    assert [w["run_id"] for w in json.loads(capsys.readouterr().out)["waiting"]] == ["first", "second"]
