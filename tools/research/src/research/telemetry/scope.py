"""Ending a run's processes by its cgroup scope (INF-93): what `research cancel` and a stage timeout stop.

A run's workload is its own process group, but its processes leave the group: vLLM's TP rank workers and suites.py's pytest-xdist
workers start groups and sessions of their own, and a worker whose parent died is reparented out of the tree (vy-glm53-eu, 19 Oct
01:20Z: eight rank workers kept ~94 GB per GPU after `research cancel`).  What they cannot leave is their cgroup.  A run sits in
systemd scopes:

  cluster-<run id>.scope   a queued run (`research run --queue`): its runner and workload (infra/cluster submit.wrappers)
  gpu-lease-<pid>.scope    the workload under `gpu-lease` (`--gpus N`, a queued GPU phase), or a lease the workload takes itself
  session-<n>.scope        a direct run's ssh login (one per launch); a login of a person's holds a tmux and anything else

The run's *own* scopes are its queue scope (named for its run id) and every gpu-lease scope whose gpu-lease process is the run's.
A Sweep's targets are every process in those scopes and, wherever they are, the run's processes: the workload leader's descendants,
its process group and session, and every process whose environment names the run dir (RESEARCH_RUN_DIR, which the runner exports to
its workload).  Never a target: the runner and what it runs besides the workload (its sampler), and the ancestors of the runner and of
this process.  A process is the runner's when its parent chain reaches, before the leader, this process, the leader's parent or a
`research run ... --id <run>` process; an ancestor of theirs claims nothing (systemd --user, the subreaper the run's orphaned workers
are reparented to, is one).  A session scope is not the run's own, so there only the run's processes are targets.

`end`: SIGTERM to every target, a grace (GRACE_S) for them to exit, SIGKILL to the rest (re-read first, so a reused pid is never
signalled), then the run's own scopes are confirmed empty (CONFIRM_S).  The record names the scopes, what each signal reached and what
was left.  A run in no scope of its own (a direct run without gpu-lease) is ended by the same steps over its group, session and
run-dir processes, and the record says so (`fallback`).
"""
from __future__ import annotations

import os
import re
import signal
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .cgroup import mounts
from .procs import list_pids, stat_fields

GRACE_S = 20.0
CONFIRM_S = 5.0
POLL_S = 0.2
KEPT = 16                     # pids a record lists per field (events.MAX_RECORD_BYTES)
CGROUP_ROOT = Path("/sys/fs/cgroup")
CONTROL_CANCEL = Path("control") / "cancel"      # attempt-dir relative: the cancel's grace and the run's scopes, read by the runner
OWN_SCOPE_RE = re.compile(r"^(?:cluster-(?P<run>r\d{8}-\d{6}-[0-9a-f]{4})|gpu-lease-(?P<pid>\d+))\.scope$")
#: The time a Sweep waits on; a test replaces both.
clock = time.monotonic
sleep = time.sleep


def run_dir_of(pid: int) -> str | None:
    """RESEARCH_RUN_DIR in the environment `pid` was exec'd with, or None (absent, gone, or not ours to read)."""
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for kv in env.split(b"\0"):
        if kv.startswith(b"RESEARCH_RUN_DIR="):
            return kv.split(b"=", 1)[1].decode(errors="replace")
    return None


def cgroup_of(pid: int) -> str | None:
    """`pid`'s cgroup v2 path (`/user.slice/.../cluster-r1.scope`), or None."""
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:].strip() or None
    return None


def cmdline_of(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return []


def cgroup_procs(root: Path, path: str) -> list[int]:
    """Every pid in cgroup `path` and the cgroups below it."""
    out: list[int] = []
    base = root / path.lstrip("/")
    for d, _, files in os.walk(base):
        if "cgroup.procs" in files:
            try:
                out += [int(x) for x in (Path(d) / "cgroup.procs").read_text().split() if x.isdigit()]
            except OSError:
                pass
    return out


def _brief(st: dict[str, Any]) -> str:
    return f"{st['pid']} {st['comm']}"


class Sweep:
    """The run whose workload leader is `leader` (start time `leader_start`) and whose run dir is `run_dir`.  `scopes`: cgroup paths
    already known to be the run's own (a cancel names them for the runner), kept only when their name says so."""

    def __init__(self, leader: int, run_dir: str, *, leader_start: int | None = None, scopes: Iterable[str] = (),
                 stat: Callable[[int], dict[str, Any] | None] = stat_fields, pids: Callable[[], list[int]] = list_pids,
                 cgroup: Callable[[int], str | None] = cgroup_of, env_of: Callable[[int], str | None] = run_dir_of,
                 cmdline: Callable[[int], list[str]] = cmdline_of, root: Path | None = None,
                 kill: Callable[[int, int], None] = os.kill):
        self.leader, self.run_dir, self.run_id = leader, str(run_dir), Path(run_dir).name
        self.stat, self.pids, self.cgroup, self.env_of, self.cmdline, self.kill = stat, pids, cgroup, env_of, cmdline, kill
        self.root = root or Path(mounts()["v2"] or CGROUP_ROOT)
        if leader_start is None and (st := stat(leader)) is not None:
            leader_start = st["starttime_ticks"]
        self.leader_start = leader_start
        self.own: dict[str, str] = {s: "named by the cancel" for s in scopes if OWN_SCOPE_RE.match(s.rsplit("/", 1)[-1])}
        self.seen: set[str] = set()                      # every cgroup a run's process was found in
        self.sent: dict[str, list[str]] = {}             # signal name -> the processes it reached
        self.denied: list[str] = []
        self.left: list[str] = []
        self.ended = False
        self._runner: dict[int, bool] = {}

    def _is_runner(self, pid: int) -> bool:
        if pid not in self._runner:
            argv = self.cmdline(pid)
            self._runner[pid] = "run" in argv and "--id" in argv[:-1] and argv[argv.index("--id") + 1] == self.run_id
        return self._runner[pid]

    def plan(self) -> dict[int, dict[str, Any]]:
        """The live targets now, pid -> /proc stat."""
        st = {p: s for p in self.pids() if (s := self.stat(p)) is not None and s["state"] not in ("Z", "X")}
        lead = self.stat(self.leader)
        leader = self.leader if lead is not None and lead["starttime_ticks"] == self.leader_start else None
        # the runner (the leader's parent) and this process claim their descendants; their ancestors are only spared themselves, since
        # one of them (systemd --user, a subreaper) is where the run's orphaned workers are reparented
        roots = {os.getpid(), *([lead["ppid"]] if leader is not None and lead else [])}
        spare: set[int] = set()
        for p in roots:
            while p > 1 and p not in spare:
                spare.add(p)
                p = (self.stat(p) or {}).get("ppid", 0)

        def side(p: int) -> str:
            """`workload` (reaches the leader), `runner` (reaches the runner first) or `orphan` (neither)."""
            hops = 0
            while p > 1 and hops < 4096:
                if p == leader:
                    return "workload"
                if p in roots or self._is_runner(p):
                    return "runner"
                s = st.get(p) or self.stat(p)
                if s is None:
                    break
                p, hops = s["ppid"], hops + 1
            return "orphan"

        ours: dict[int, str] = {}
        for p, s in st.items():
            if p in spare:
                continue
            if leader is not None and p == leader:
                ours[p] = "workload"
                continue
            k = side(p)
            if k == "workload" or (k == "orphan" and (s["pgrp"] == self.leader or s["session"] == self.leader
                                                      or self.env_of(p) == self.run_dir)):
                ours[p] = k
        for p in ours:
            if (cg := self.cgroup(p)) is None:
                continue
            self.seen.add(cg)
            m = OWN_SCOPE_RE.match(cg.rsplit("/", 1)[-1])
            if m and cg not in self.own:
                if m["run"] == self.run_id:
                    self.own[cg] = f"the queue scope of run {self.run_id}"
                elif m["pid"] and (int(m["pid"]) in ours or int(m["pid"]) == self.leader):
                    self.own[cg] = f"gpu-lease {m['pid']}'s scope, the run's lease"
        for cg in self.own:
            for p in cgroup_procs(self.root, cg):
                if p not in ours and p in st and p not in spare and side(p) != "runner":
                    ours[p] = "scope"
        return {p: st[p] for p in ours}

    def signal(self, sig: int, skip: Iterable[int] = ()) -> list[int]:
        """Send `sig` to every target but `skip`; the pids it reached."""
        name, hit = signal.Signals(sig).name, []
        skip = set(skip)
        for p, s in self.plan().items():
            if p in skip:
                continue
            try:
                self.kill(p, sig)
            except ProcessLookupError:
                continue
            except PermissionError:
                self.denied.append(_brief(s))
                continue
            hit.append(p)
            self.sent.setdefault(name, []).append(_brief(s))
        return hit

    def wait(self, grace_s: float) -> list[int]:
        """Until every target planned now has exited or `grace_s` has passed; those still alive."""
        targets = self.plan()
        t_end = clock() + grace_s

        def alive() -> list[int]:
            return [p for p, s in targets.items() if (n := self.stat(p)) is not None
                    and n["starttime_ticks"] == s["starttime_ticks"] and n["state"] not in ("Z", "X")]
        while (left := alive()) and clock() < t_end:
            sleep(POLL_S)
        return left

    def end(self, grace_s: float = GRACE_S, *, termed: bool = False) -> dict[str, Any]:
        """SIGTERM (unless `termed`: it was sent already), the grace, SIGKILL to whatever is left then (planned again, so what was
        started during the grace too), and the confirmation that nothing is; the record."""
        if not termed:
            self.signal(signal.SIGTERM)
        self.wait(grace_s)
        self.signal(signal.SIGKILL)
        self.wait(CONFIRM_S)
        self.left = [_brief(s) for s in self.plan().values()]
        self.ended = True
        return self.record()

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"scopes": dict(self.own), "fallback": None}
        if not self.own:
            out["fallback"] = ("no scope of the run's own (its processes were in " + (", ".join(sorted(self.seen)) or "no cgroup read")
                               + "): its process group, session and the processes naming its run dir only")
        for name, who in self.sent.items():
            out[name] = {"n": len(who), "pids": who[:KEPT]}
        if self.denied:
            out["denied"] = self.denied[:KEPT]
        if self.ended:
            out["left"] = self.left[:KEPT]
            out["empty"] = not self.left
        return out
