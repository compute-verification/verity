"""research.withdraw: `research cancel` of a run still held on its machine (disk-hold, cpu-hold) ends exactly the hold processes
that name the run's ack FIFO and records the run Withdrawn. On 19 Oct infra withdrew 20 such runs by hand, because the cancel
refused a run with no status.json. The process table here is a directory laid out as /proc, whose signals the test applies."""
from __future__ import annotations

import fcntl
import json
import signal
from pathlib import Path

from research import withdraw as W

RUN = "r20261019-200000-0b1d"


class Table(W.Procs):
    """/proc as files under `root`: SIGSTOP stops a process, SIGTERM is pending until it runs again, SIGCONT runs it (and so ends
    it with a pending SIGTERM). `on_stop[pid]` runs as `pid` stops: what it did up to its stop, such as exec'ing on."""

    def __init__(self, root: Path, *, survives: bool = False):
        super().__init__(root, kill=self._kill, sleep=self._sleep)
        self.sent: list[tuple[int, int]] = []
        self.on_stop: dict[int, callable] = {}
        self.pending: set[int] = set()
        self.survives = survives

    def add(self, pid: int, argv: list[str], state: str = "S") -> None:
        (self.root / str(pid)).mkdir(parents=True)
        (self.root / str(pid) / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
        self.set_state(pid, state)

    def set_state(self, pid: int, state: str) -> None:
        (self.root / str(pid) / "stat").write_text(f"{pid} (python3) {state} " + " ".join(["0"] * 18) + f" {1000 + pid} 0\n")

    def _kill(self, pid: int, sig: int) -> None:
        if not (self.root / str(pid)).is_dir():
            raise ProcessLookupError(pid)
        self.sent.append((pid, sig))
        if sig == signal.SIGSTOP:
            self.on_stop.pop(pid, lambda: None)()
            self.set_state(pid, "T")
        elif sig == signal.SIGTERM:
            self.pending.add(pid)
        elif sig == signal.SIGCONT:
            if pid in self.pending and not self.survives:
                for f in (self.root / str(pid)).iterdir():
                    f.unlink()
                (self.root / str(pid)).rmdir()
            else:
                self.set_state(pid, "S")

    def _sleep(self, s: float) -> None:
        raise AssertionError("a stopped process is seen stopped at once here")

    def pidfd(self, pid: int) -> None:
        return None

    def gone(self, pid: int, fd: None, wait_s: float) -> bool:
        return not (self.root / str(pid)).is_dir()


def _machine(tmp_path: Path):
    root = tmp_path / "research"
    rd, req = root / "runs" / RUN, root / "requests" / RUN
    rd.mkdir(parents=True)
    (req / "custody").mkdir(parents=True)
    (req / "custody" / "cred.json").write_text("{}")
    (req / "request.json").write_text("{}")
    (rd / "inputs").mkdir()
    (rd / "inputs" / "lean-deps-urls.json").write_text("{}")
    (rd / "inputs" / "data.bin").write_text("kept")
    (rd / "launcher.log").write_text("research cpu-hold: waiting for 4 CPUs\n")
    (rd / "launch.json").write_text(json.dumps({"run_id": RUN, "runner_pid": 100, "request": str(req / "request.json")}))
    (root / "runs" / ".exclusive.lock").write_text(RUN + "\n")
    return rd, req


def _args(rd: Path) -> dict:
    return {"run_dir": str(rd), "reason": "superseded by #1900", "by": "infra", "cred_file": "cred.json",
            "cred_dirs": ["custody", "store-read"], "credentials": ["inputs/lean-deps-urls.json", "inputs/.lean-deps-urls.json.*"]}


def _hold(kind: str, ack: Path) -> list[str]:
    return ["/usr/bin/python3", "-m", "research", kind, "--cpus", "4", "--from", "48-95", "--wrap", "9", "--who", RUN, "--ack", str(ack),
            "--", "systemd-run", "--user", "--scope", "python3", "-m", "research", "run", "--id", RUN]


def test_a_held_run_is_withdrawn_its_hold_alone_ended_and_recorded_as_a_manual_cancel(tmp_path):
    rd, req = _machine(tmp_path)
    procs = Table(tmp_path / "proc")
    procs.add(100, _hold("cpu-hold", req / ".launch-ack"))
    procs.add(101, _hold("cpu-hold", req.parent / "r-other" / ".launch-ack"))
    procs.add(102, ["/usr/bin/python3", "-m", "research", "run", "--id", RUN, "--remote-json", json.dumps({"request": str(req)})])
    out = W.withdraw(_args(rd), procs)
    assert out["outcome"] == "withdrawn"
    assert procs.sent == [(100, signal.SIGSTOP), (100, signal.SIGTERM), (100, signal.SIGCONT)], "only the run's own hold is signalled"
    assert (tmp_path / "proc" / "101").is_dir() and (tmp_path / "proc" / "102").is_dir()
    refused = json.loads((rd / "refused.json").read_text())
    assert {k: refused[k] for k in ("run_id", "reason", "fault", "by")} == {"run_id": RUN, "reason": "superseded by #1900",
                                                                             "fault": "Withdrawn", "by": "infra"}
    assert refused["at_utc"].endswith("Z") and refused["holds"] == [{"pid": 100, "hold": "cpu-hold", "starttime_ticks": "1100"}]
    intent, result = [json.loads(x) for x in (rd / "events.jsonl").read_text().splitlines()]
    assert (intent["kind"], intent["actor"], intent["signal"]) == ("cancel_intent", "manual", signal.SIGTERM)
    assert intent["reason"] == "superseded by #1900 (research cancel by infra)" and intent["target"]["pids"] == [100]
    assert (result["kind"], result["delivered"], result["target_still_present"]) == ("cancel_result", True, [])
    assert not (req / "custody" / "cred.json").exists() and not (rd / "inputs" / "lean-deps-urls.json").exists()
    assert (rd / "inputs" / "data.bin").exists() and not (rd.parent / ".exclusive.lock").exists()


def test_a_disk_hold_ahead_of_a_cpu_hold_is_the_runs_and_only_its_own_options_say_whose():
    ack = "/r/requests/A/.launch-ack"
    disk = ["python3", "-m", "research", "disk-hold", "--gb", "20", "--path", "/r/runs", "--ack", ack, "--",
            *_hold("cpu-hold", Path(ack))]
    assert W.hold_of(disk, {ack}) == "disk-hold"
    assert W.hold_of(["/opt/bin/research", "cpu-hold", f"--ack={ack}", "--", "x"], {ack}) == "cpu-hold"
    other = ["python3", "-m", "research", "disk-hold", "--ack", "/r/requests/B/.launch-ack", "--", *_hold("cpu-hold", Path(ack))]
    assert W.hold_of(other, {ack}) is None, "the CMD a hold runs is not the hold's own --ack"
    assert W.hold_of(["python3", "-m", "research", "run", "--ack", ack], {ack}) is None
    assert W.hold_of(["vim", "cpu-hold", "--ack", ack], {ack}) is None


def test_a_hold_that_execs_into_its_runner_as_the_cancel_stops_it_is_continued_and_reported_started(tmp_path):
    """The race: the hold is released and execs its runner (the same pid) between the scan and the stop. Read again once stopped,
    its argv is the runner's, so it is continued, nothing is ended or recorded, and the cancel goes on as a running run's."""
    rd, req = _machine(tmp_path)
    procs = Table(tmp_path / "proc")
    procs.add(100, _hold("cpu-hold", req / ".launch-ack"))

    def exec_runner() -> None:
        (tmp_path / "proc" / "100" / "cmdline").write_bytes(b"python3\0-m\0research\0run\0--id\0" + RUN.encode() + b"\0")
        (rd / "status.json").write_text(json.dumps({"state": "running", "pid": 100, "pgid": 200}))

    procs.on_stop[100] = exec_runner
    out = W.withdraw(_args(rd), procs)
    assert out == {"outcome": "started", "state": "running", "moved_on": [100]}
    assert procs.sent == [(100, signal.SIGSTOP), (100, signal.SIGCONT)]
    assert not (rd / "refused.json").exists() and not (rd / "events.jsonl").exists()
    assert (req / "custody" / "cred.json").exists(), "a run that started keeps its custody key"


def test_a_launch_still_being_answered_is_busy_and_nothing_is_signalled(tmp_path):
    rd, req = _machine(tmp_path)
    procs = Table(tmp_path / "proc")
    procs.add(100, _hold("disk-hold", req / ".launch-ack"))
    with open(req / ".launch.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = W.withdraw(_args(rd), procs)
    assert out["outcome"] == "busy" and procs.sent == [] and not (rd / "refused.json").exists()


def test_a_run_with_records_of_its_own_is_left_to_them(tmp_path):
    rd, req = _machine(tmp_path)
    procs = Table(tmp_path / "proc")
    procs.add(100, _hold("cpu-hold", req / ".launch-ack"))
    (rd / "refused.json").write_text(json.dumps({"reason": "no memory", "fault": "MemoryNotAvailable"}))
    assert W.withdraw(_args(rd), procs) == {"outcome": "refused", "reason": "no memory", "fault": "MemoryNotAvailable"}
    (rd / "refused.json").unlink()
    (rd / "status.json").write_text(json.dumps({"state": "running"}))
    assert W.withdraw(_args(rd), procs) == {"outcome": "started", "state": "running"}
    assert procs.sent == []


def test_a_run_with_no_hold_is_not_marked_withdrawn(tmp_path):
    """No hold names the request: the runner is starting (the hold released it), or it died without a record, which is a crash
    for launcher.log to explain and not a withdrawal."""
    rd, _ = _machine(tmp_path)
    procs = Table(tmp_path / "proc")
    procs.add(100, ["python3", "-m", "research", "run", "--id", RUN])
    assert W.withdraw(_args(rd), procs) == {"outcome": "no-hold", "runner_pid": 100, "runner_alive": True}
    (tmp_path / "proc" / "100" / "cmdline").unlink()
    (tmp_path / "proc" / "100" / "stat").unlink()
    (tmp_path / "proc" / "100").rmdir()
    assert W.withdraw(_args(rd), procs) == {"outcome": "no-hold", "runner_pid": 100, "runner_alive": False}
    assert procs.sent == [] and not (rd / "refused.json").exists()


def test_a_hold_that_outlives_its_sigterm_leaves_the_run_unrecorded(tmp_path):
    rd, req = _machine(tmp_path)
    procs = Table(tmp_path / "proc", survives=True)
    procs.add(100, _hold("cpu-hold", req / ".launch-ack"))
    out = W.withdraw(_args(rd), procs)
    assert out["outcome"] == "survived" and out["pids"] == [100]
    assert not (rd / "refused.json").exists() and (req / "custody" / "cred.json").exists()
    assert json.loads((rd / "events.jsonl").read_text().splitlines()[-1])["target_still_present"] == [100]
