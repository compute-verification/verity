"""`research disk-hold`: a queued run's disk admission on the machine, ahead of its allocation (research.disk)."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from research import disk, remote

SRC = str(Path(remote.__file__).resolve().parents[1])
TB = 1000 * disk.GB
S = int(1e9)


def _disk(used_tb: list[float]):
    """A 10 TB disk whose use is read from the front of `used_tb` at each poll."""
    def usage(path):
        return int(10 * TB), int(used_tb.pop(0) * TB) if len(used_tb) > 1 else int(used_tb[0] * TB), 0
    return usage


class Node:
    """One filesystem's queue, shared by the runs a test steps one poll at a time: a 10 TB disk at `used_tb`, a clock in ns, and
    the pids that are alive."""

    def __init__(self, tmp_path: Path, used_tb: float):
        self.path, self.used_tb, self.now, self.live, self.runs = tmp_path, used_tb, 1000 * S, set(), []

    def usage(self, path):
        return int(10 * TB), int(self.used_tb * TB), 0

    def run(self, pid: int, gb: float = 100) -> Run:
        self.live.add(pid)
        self.runs.append(r := Run(self, pid, gb))
        self.now += S
        return r

    def tickets(self) -> list[str]:
        return sorted(p.name for p in (self.path / disk.QUEUE_DIR).iterdir() if disk.TICKET.fullmatch(p.name))


class Stopped(Exception):
    pass


@pytest.fixture
def new_node(tmp_path):
    """Node(tmp_path, used_tb); at teardown each run still held is stopped, so no thread outlives its test."""
    made = []
    yield lambda used_tb: made.append(Node(tmp_path, used_tb)) or made[-1]
    for r in (r for n in made for r in n.runs if r.state[0] == "held"):
        r.stopping = True
        assert r.step() == "raised" and isinstance(r.state[1], Stopped)


class Run:
    """A `disk.hold` in a thread whose sleep blocks until the test steps it, so only one run acts at a time."""

    def __init__(self, node: Node, pid: int, gb: float):
        self.go, self.at, self.lines, self.stopping = threading.Semaphore(0), queue.Queue(), [], False

        def body():
            try:
                self.at.put(("released", disk.hold(gb, node.path, usage=node.usage, sleep=self._sleep, log=self.lines.append,
                                                   clock=lambda: node.now, is_alive=lambda p, s: p in node.live, pid=pid)))
            except BaseException as e:  # noqa: BLE001
                self.at.put(("raised", e))

        threading.Thread(target=body, daemon=True).start()
        self.state = self.at.get()

    def _sleep(self, s):
        self.at.put(("held", s))
        self.go.acquire()
        if self.stopping:
            raise Stopped

    def step(self) -> str:
        self.go.release()
        self.state = self.at.get()
        return self.state[0]


def test_a_run_fits_at_once_up_to_80_percent(tmp_path):
    sleeps, lines = [], []
    assert disk.hold(1000, tmp_path, usage=_disk([6.9]), sleep=sleeps.append, log=lines.append) == 0
    assert sleeps == [] and lines == []
    assert (tmp_path / disk.QUEUE_DIR).stat().st_mode & 0o7777 == 0o1777, "runs of other users share the queue"


def test_a_held_run_is_released_at_80_percent_with_no_lower_mark(tmp_path):
    """vy-nebius-2, 18 Oct: retention gc stops at 75% and the disk sat at 76-80%; a release mark of 75% held runs forever."""
    sleeps, lines = [], []
    held = disk.hold(1000, tmp_path, usage=_disk([7.5, 7.2, 7.0]), sleep=sleeps.append, log=lines.append)
    assert held == 2 and sleeps == [disk.POLL_S] * 2, "8.5 and 8.2 TB are over 80%; 8.0 is not"
    assert lines[0].startswith("research disk-hold: holding: 1000 GB more would fill")
    assert "to 85.0% (over 80%)" in lines[0]
    assert lines[-1].startswith("research disk-hold: released after 2 poll(s): 80.0%")
    assert not [p for p in (tmp_path / disk.QUEUE_DIR).iterdir() if disk.TICKET.fullmatch(p.name)], "the ticket goes with the release"


def test_a_new_run_waits_behind_an_older_held_one_even_when_it_fits(new_node):
    node = new_node(7.95)
    a = node.run(101)
    assert a.state[0] == "held"
    node.used_tb = 7.0
    b = node.run(102)
    assert b.state[0] == "held" and "1 held run(s) ahead" in b.lines[0], "under 80%, but a held run was there first"
    assert a.step() == "released" and b.step() == "held", "a went in now; b waits out the pacing window"
    node.now += 2 * disk.POLL_S * S
    assert b.step() == "released"
    assert node.tickets() == []


def test_held_runs_go_in_first_come_first_served_one_per_window(new_node):
    node = new_node(7.95)
    a, b, c = node.run(103), node.run(102), node.run(101)
    assert [r.state[0] for r in (a, b, c)] == ["held"] * 3
    node.used_tb = 6.0
    order = []
    for _ in range(3):
        for name, r in (("c", c), ("b", b), ("a", a)):
            if r.state[0] == "held" and r.step() == "released":
                order.append(name)
        node.now += 2 * disk.POLL_S * S
    assert order == ["a", "b", "c"], "by ticket age, not pid or poll order"


def test_two_eligible_held_runs_are_not_released_in_one_window(new_node):
    node = new_node(7.95)
    a, b = node.run(101), node.run(102)
    node.used_tb = 6.0
    assert a.step() == "released"
    node.now += 2 * disk.POLL_S * S - 1
    assert b.step() == "held", "a has not had its window to start writing"
    node.now += 1
    assert b.step() == "released"


def test_a_clock_stepped_back_does_not_stall_the_queue(new_node):
    node = new_node(7.95)
    a, b = node.run(101), node.run(102)
    node.used_tb = 6.0
    assert a.step() == "released"
    node.now -= 3600 * S
    assert b.step() == "released", "a's release reads as an hour ahead: not a window to wait out"


def test_the_head_waits_for_its_own_gb_to_fit(new_node):
    node = new_node(7.95)
    a = node.run(101, gb=1000)
    b = node.run(102)
    node.used_tb = 7.5
    assert b.step() == "held" and a.step() == "held", "a's 1000 GB would reach 85%; b is behind it"
    node.used_tb = 6.9
    assert a.step() == "released"


def test_a_dead_runs_ticket_is_ignored_and_removed(new_node):
    node = new_node(7.0)
    a = node.run(101)
    a2 = node.run(102)
    assert a.state[0] == "released" and a2.state[0] == "released", "an empty queue admits at once"
    node.used_tb = 7.95
    assert node.run(103).state[0] == "held"
    node.live.discard(103)
    node.used_tb = 7.0
    assert node.tickets() == [f"{1002 * S}.103"]
    assert node.run(104).state == ("released", 0), "103 died holding: its ticket doesn't hold 104"
    assert node.tickets() == []


def test_alive_names_the_process_its_ticket_recorded():
    me = os.getpid()
    assert disk.alive(me, disk.started(me)) and disk.alive(me, "")
    assert not disk.alive(me, "1"), "the pid now names a process started at another time: reused"
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not disk.alive(child.pid, "")


def test_the_ticket_goes_when_the_hold_fails(tmp_path):
    node = Node(tmp_path, 7.95)

    def boom(s):
        assert node.tickets() == [f"{1000 * S}.101"]
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        disk.hold(100, tmp_path, usage=node.usage, sleep=boom, log=lambda s: None, clock=lambda: node.now,
                  is_alive=lambda p, s: True, pid=101)
    assert node.tickets() == []


def test_the_command_runs_in_its_place_once_admitted(tmp_path, monkeypatch):
    events = []
    rc = disk.main(["--gb", "1000", "--path", str(tmp_path), "--", "echo", "ran"], usage=_disk([7.5, 7.5, 7.5, 7.0]),
                   sleep=lambda s: events.append("poll"), execvp=lambda f, a: events.append(("exec", f, a)))
    assert events == ["poll"] * 3 + [("exec", "echo", ["echo", "ran"])], "held for 3 polls, then exec'd in place"
    assert rc == 127, "execvp returns only when it fails"
    events.clear()
    monkeypatch.setenv("RESEARCH_DISK_HOLD_PCT", "90")
    disk.main(["--gb", "1000", "--path", str(tmp_path), "--", "echo"], usage=_disk([7.5]),
              sleep=lambda s: events.append("poll"), execvp=lambda f, a: events.append("exec"))
    assert events == ["exec"], "8.0 TB of 10 is under a 90% hold"
    assert subprocess.run([sys.executable, "-m", "research", "disk-hold", "--gb", "0", "--path", str(tmp_path)],
                          capture_output=True, text=True, env={"PYTHONPATH": SRC}).returncode == 2
