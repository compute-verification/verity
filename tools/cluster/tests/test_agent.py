"""The node agent, driven by an injected clock over a fake vy-nebius-2 (a lease directory and the /proc entries of its
holders): what it reads and when, what it writes, when it stops itself, and what live mode carries out."""
from __future__ import annotations

import errno
import fcntl
import json
import os
import signal
from pathlib import Path

import pytest

from cluster import agent, ledger, model, nebius2, shadow
from cluster.ledger import epoch
from cluster.plan import Start

DESC = Path(__file__).parents[1] / "descriptions" / "nebius.toml"
T0 = epoch("2026-09-30T20:00:00Z")


@pytest.fixture(scope="module")
def c() -> model.Cluster:
    return model.load(DESC)


class Node:
    """A lease directory whose locks are "held" by fake /proc entries, and the fill runner's status file."""

    def __init__(self, root: Path):
        self.gl, self.proc, self.status = root / "gl", root / "proc", root / "status.txt"
        self.gl.mkdir()
        self.proc.mkdir()
        self.set_status(False)

    def set_status(self, timed: bool) -> None:
        self.status.write_text(f"2026-09-30T20:00:00Z free GPUs 0/8, timed {timed}\n")

    def _hold(self, pid: int, path: Path) -> None:
        fd, info = self.proc / str(pid) / "fd", self.proc / str(pid) / "fdinfo"
        fd.mkdir(parents=True, exist_ok=True)
        info.mkdir(exist_ok=True)
        n = str(10 + len(os.listdir(fd)))
        (fd / n).symlink_to(path)
        (info / n).write_text("pos:\t0\nlock:\t1: FLOCK  ADVISORY  WRITE 0 fe:20:1 0 EOF\n")

    def lease(self, pid: int, gpus, since: str, extra: str = "") -> None:
        for i in gpus:
            (self.gl / f"{i}.lock").touch()
            (self.gl / f"{i}.owner").write_text(f"pid={pid} run=- who=w{pid} since={since} {extra} cmd=x\n")
            self._hold(pid, self.gl / f"{i}.lock")

    def waiter(self, pid: int, n: int, since: str) -> None:
        f = self.gl / f"wait.1.{pid}"
        f.write_text(f"pid={pid} n={n} run=- who=w{pid} since={since}\n")
        self._hold(pid, f)

    def end(self, pid: int) -> None:
        for p in (self.proc / str(pid) / "fd").iterdir():
            p.unlink()


class Clock:
    """Time that moves only when the agent sleeps; `at[t]` runs when the clock reaches t."""

    def __init__(self, t: float, at: dict | None = None):
        self.t, self.at, self.sleeps = t, dict(at or {}), []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s
        for when in sorted(k for k in self.at if k <= self.t):
            self.at.pop(when)()


def run(c, node, out, clock, **kw):
    kw.setdefault("until", clock.t + 5)
    return agent.run(c, out, clock=clock, sleep=clock.sleep, lease_dir=node.gl, proc=node.proc, status=node.status, **kw)


def test_shadow_mode_mirrors_the_node_into_the_ledger_and_state_json_and_never_takes_the_agent_lock(c, tmp_path):
    node = Node(tmp_path)
    node.lease(7, [0], "2026-09-30T19:59:00Z", "until=2026-09-30T20:20:00Z mem=193G preempt=1")
    node.waiter(20, 1, "2026-09-30T19:59:30Z")
    out = tmp_path / "out"
    s = run(c, node, out, Clock(T0))
    assert s["stopped"] == "until" and s["exception"] is None and s["ticks_read"] == 5
    records = ledger.read(out / "ledger.jsonl")
    assert ledger.verify(records) == [] and [r["kind"] for r in records] == ["submit", "start"]
    st = ledger.state(records)
    assert ledger.state_from_dict(json.loads((out / "state.json").read_text())) == st
    assert list(st.allocations) == [f"gl-7@{epoch('2026-09-30T19:59:00Z')}"] and list(st.queue) == ["gl-20"]
    assert not (node.gl / "agent.lock").exists() and not list(node.gl.glob("grant.*"))
    assert s["footprint"]["nvml_calls"] == 0 and json.loads((out / "summary.json").read_text())["mode"] == "shadow"


def test_during_a_timed_window_it_reads_only_the_status_file_every_five_seconds(c, tmp_path, monkeypatch):
    node = Node(tmp_path)
    reads = []
    real = nebius2.read
    monkeypatch.setattr(nebius2, "read", lambda t, *a: reads.append(t) or real(t, *a))
    clock = Clock(T0)

    def begin():
        node.lease(30, range(8), "2026-09-30T20:00:02Z", "timed=1")

    def over():
        node.end(30)
        node.set_status(False)
        os.utime(node.status, (clock.t, clock.t))
    clock.at = {T0 + 2: begin, T0 + 5: lambda: node.set_status(True), T0 + 40: over}
    s = run(c, node, tmp_path / "out", clock, until=T0 + 50)
    in_window = [t for t in reads if T0 + 2 < t <= T0 + 40]
    assert in_window == [], "nothing but status.txt is read inside a window"
    assert s["windows_seen"] == 1 and s["windows_from_status_only"] == 0 and s["window_polls"] == 7
    assert clock.sleeps[:2] == [1.0, 1.0] and set(clock.sleeps[2:10]) == {5.0} and clock.sleeps[10] == 1.0
    kinds = [r["kind"] for r in ledger.read(tmp_path / "out" / "ledger.jsonl")]
    assert kinds == ["start", "end"]


def test_the_stop_file_an_exception_and_a_plan_that_breaks_a_rule_each_stop_it_with_a_summary(c, tmp_path, monkeypatch):
    node = Node(tmp_path)
    stop = tmp_path / "stop"
    clock = Clock(T0, {T0 + 3: stop.touch})
    assert run(c, node, tmp_path / "a", clock, stop_file=stop, until=None)["stopped"] == f"stop file {stop}"

    def boom(*a):
        raise OSError(errno.EIO, "boom")
    monkeypatch.setattr(nebius2, "read", boom)
    s = run(c, node, tmp_path / "b", Clock(T0))
    assert s["stopped"] == "exception" and "OSError" in s["exception"]
    monkeypatch.undo()

    node.lease(7, [0], "2026-09-30T19:59:00Z")
    node.waiter(20, 1, "2026-09-30T19:59:30Z")
    monkeypatch.setattr(shadow, "plan", lambda cl, n, allocs, queue, now: [
        Start(model.Allocation(id=f"{j.id}@{now}", job=j, node=n, gpus=(0,), cpus=frozenset(), memory_gib=1, started=now))
        for j in queue])
    s = run(c, node, tmp_path / "c", Clock(T0))
    assert s["stopped"].startswith("invariant: the plan breaks a rule") and s["ticks_read"] == 1


def test_live_mode_holds_the_agent_lock_evicts_fill_for_a_window_then_grants_it(c, tmp_path, monkeypatch):
    node = Node(tmp_path)
    node.lease(7, [0], "2026-09-30T19:59:00Z", "until=2026-09-30T20:20:00Z mem=193G preempt=1 scope=gpu-lease-7")
    node.waiter(20, 8, "2026-09-30T20:00:00Z")
    signals = []
    held = []

    def probe():
        fd = os.open(node.gl / "agent.lock", os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            held.append(False)
        except BlockingIOError:
            held.append(True)
        finally:
            os.close(fd)
    clock = Clock(T0, {T0 + 1: probe, T0 + 2: lambda: node.end(7)})
    ex = agent.Executor(node.gl, kill=lambda pid, sig: signals.append((pid, sig)))
    s = run(c, node, tmp_path / "out", clock, mode="live", executor=ex)
    assert s["mode"] == "live" and s["stopped"] == "until" and held == [True]
    assert signals == [(7, signal.SIGTERM)]
    [p] = [nebius2.preemption(x) for x in (node.gl / "preempted.log").read_text().splitlines()]
    assert (p["signal"], p["gpu"], p["by"], p["victim"]["pid"]) == ("SIGTERM", 0, 20, "7")
    assert (node.gl / "grant.20").read_text() == "0,1,2,3,4,5,6,7\n"
    fd = os.open(node.gl / "agent.lock", os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)


def test_live_mode_kills_a_lease_still_there_thirty_seconds_after_its_sigterm(c, tmp_path):
    ex = agent.Executor(tmp_path, kill=lambda pid, sig: sent.append((pid, sig)))
    sent: list = []
    for t in (0, 1, 29, 30, 31):
        ex.evict(T0 + t, "gl-7@1", (0,), "pid=7 since=2026-09-30T19:59:00Z", 20, 8, "w")
    assert sent == [(7, signal.SIGTERM), (7, signal.SIGKILL), (7, signal.SIGKILL)]


def test_the_footprint_counts_anything_that_would_touch_nvml():
    fp = agent.Footprint()
    fp.hook("subprocess.Popen", ("nvidia-smi", ["nvidia-smi", "-L"], None, None))
    fp.hook("ctypes.dlopen", ("libnvidia-ml.so.1",))
    fp.hook("subprocess.Popen", ("flock", ["flock"], None, None))
    fp.hook("open", ("/proc/locks", "r", 0))
    r = fp.report(100.0)
    assert (r["execs"], r["nvml_calls"]) == (3, 2) and r["cpu_share"] is not None
