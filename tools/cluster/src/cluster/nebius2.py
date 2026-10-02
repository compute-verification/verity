"""vy-nebius-2's `gpu-lease` as the model sees it: the adapter between the node's lock files and the planner.

`gpu-lease` keeps its state in `/run/gpu-lease`: `<i>.lock`, held with flock(1) for as long as a lease runs, beside
`<i>.owner`, one line naming the holder (`pid=… run=… who=… uuid=… since=… [until=…] [mem=…G] [preempt=1] [timed=1]
[scope=…] cmd=…`, left behind when the lease ends); and `wait.<ns>.<pid>`, a `--wait` request's line (`pid=… n=… [on=…]
run=… who=… since=…`), locked while it waits. The node's sampler records the same lines every 10 s
(`/workspace/pouw/infra/util/*.jsonl`: `gpus[].lease`, `wait_files`), which is what a replay reads.

`read` takes the node's state without touching a lock: which files are locked comes from the holders' `/proc/<pid>/fdinfo`,
never from a `flock -n` probe, which would itself hold the lock for an instant and could make a concurrent take fail.

The mapping (the cutover plan's): a lease is a CPU-unmanaged allocation (`cpus = 0`) of its GPUs, pinned to them, with job
id `gl-<pid>` (a waiter's job id too, so a start takes its request out of the queue) and allocation id
`gl-<pid>@<since>`. `--preemptible` is class `fill` and `requeue`; `--timed`, or every GPU held (or asked for) by one
request, is a quiet job of class `timed`; anything else is a `session` nobody preempts. `--max-min` is `max_s`; without it
a lease is a session expected to take `GPU_LEASE_MAX_MIN` (30 minutes, `gpu-lease`'s own backfill rule), never enforced.
A wait file doesn't say `--max-min`, `--preemptible` or `--timed`, so a waiter is read as a session, or a timed window when
it asks for the whole node. A wait file's `on=` naming exactly the GPUs it asks for pins them (`gpu_ids`); naming
more is `gpu_pool`, the only GPUs it may be granted, as `gpu-lease --on` means.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .ledger import epoch
from .model import Allocation, Cluster, Job

NODE = "vy-nebius-2"
LEASE_DIR = Path("/run/gpu-lease")
EXPECTED_S = 30 * 60


def fields(line: str) -> dict[str, str]:
    """An owner or wait line's `key=value` fields; `cmd` is the rest of the line after `cmd=`."""
    head, sep, cmd = line.strip().partition(" cmd=")
    if not sep and head.startswith("cmd="):
        head, cmd = "", head[4:]
    out = dict(tok.split("=", 1) for tok in head.split() if "=" in tok)
    if sep or cmd:
        out["cmd"] = cmd
    return out


@dataclass(frozen=True)
class Observation:
    """The node's leases and waiters at `t`: each held GPU's owner line, and each live waiter's line with its file name.
    `held` lists the GPUs whose lock is held by no owner line's pid (a lease being taken, or a dead holder's child)."""
    t: int
    leases: dict[int, str]
    waits: tuple[tuple[str, str], ...] = ()
    waits_known: bool = True
    held: tuple[int, ...] = ()


def from_sampler(rec: dict[str, Any]) -> Observation:
    """One sampler line (`gpu_util_sampler.py`). Lines before the sampler recorded wait files have `waits_known` False."""
    leases = {int(g["i"]): g["lease"] for g in rec["gpus"] if g.get("lease")}
    waits = tuple((f"wait.{fields(w).get('pid', '?')}", w) for w in rec.get("wait_files") or ())
    return Observation(epoch(rec["t"]), leases, waits, "wait_files" in rec)


def held_by(pid: int, paths: set[Path], proc: Path = Path("/proc")) -> set[Path] | None:
    """Which of `paths` process `pid` holds a lock on: its open files among them whose `fdinfo` lists a `lock:` line.
    Empty when the process is gone; None when its descriptors can't be read (another user's process).

    `gpu-lease` takes each lock with a short-lived `flock -n -x FD` child on a descriptor its own shell keeps open, so the
    shell (the owner line's `pid`, the wait file's last name part) holds it; `/proc/locks` names the taker, that child, so
    the holder can't be read from there (whether anyone holds it can: `locked`)."""
    fd_dir = proc / str(pid) / "fd"
    want = {str(p) for p in paths}
    out: set[Path] = set()
    try:
        fds = os.listdir(fd_dir)
    except FileNotFoundError:
        return set()
    except PermissionError:
        return None
    for fd in fds:
        try:
            target = os.readlink(fd_dir / fd)
            if target in want and "\nlock:" in (proc / str(pid) / "fdinfo" / fd).read_text():
                out.add(Path(target))
        except (FileNotFoundError, PermissionError):
            continue
    return out


def locked(paths: set[Path], proc: Path = Path("/proc")) -> set[Path] | None:
    """Which of `paths` anyone holds a flock on, from `/proc/locks` by device and inode (its pid is the taker's, which may be
    long gone); None when it can't be read."""
    try:
        text = (proc / "locks").read_text()
    except OSError:
        return None
    held = {p[5] for p in (line.split() for line in text.splitlines()) if len(p) > 5 and p[1] == "FLOCK"}
    out = set()
    for path in paths:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        if f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}" in held:
            out.add(path)
    return out


def read(t: int, lease_dir: Path = LEASE_DIR, proc: Path = Path("/proc")) -> Observation:
    """The live state of `lease_dir`, read-only and without taking a lock: each GPU whose owner line's pid holds its lock,
    and each wait file its waiter holds. A lock held under a line that names another pid (a lease being taken, or a holder
    that died while a child kept its descriptor) is `held`: busy, and nobody's to move."""
    lease_dir = Path(os.path.realpath(lease_dir))
    leases: dict[int, str] = {}
    waits: list[tuple[str, str]] = []
    try:
        names = sorted(os.listdir(lease_dir))
    except FileNotFoundError:
        return Observation(t, {})
    owners: dict[int, list[tuple[int, str]]] = {}
    for name in names:
        try:
            if name.endswith(".owner") and name[:-6].isdigit():
                line = (lease_dir / name).read_text().strip()
                pid = fields(line).get("pid", "")
                if pid.isdigit():
                    owners.setdefault(int(pid), []).append((int(name[:-6]), line))
            elif name.startswith("wait.") and name.rsplit(".", 1)[-1].isdigit():
                pid = int(name.rsplit(".", 1)[-1])
                if held_by(pid, {lease_dir / name}, proc):
                    waits.append((name, (lease_dir / name).read_text().strip()))
        except FileNotFoundError:
            continue
    for pid, lines in owners.items():
        held = held_by(pid, {lease_dir / f"{i}.lock" for i, _ in lines}, proc)
        for i, line in lines:
            if held is None or lease_dir / f"{i}.lock" in held:
                leases[i] = line
    gpus = {int(name[:-5]) for name in names if name.endswith(".lock") and name[:-5].isdigit()}
    busy = locked({lease_dir / f"{i}.lock" for i in gpus - set(leases)}, proc) or set()
    return Observation(t, leases, tuple(sorted(waits)), held=tuple(sorted(int(p.name[:-5]) for p in busy)))


def timed_now(cluster: Cluster, obs: Observation, node: str = NODE) -> bool:
    """A timed window holds the node: a lease tagged `timed=1`, or every GPU held by one pid (the sampler's and the fill
    runner's rule)."""
    if any(" timed=1 " in f" {line} " for line in obs.leases.values()):
        return True
    n = len(cluster.nodes[node].gpus)
    pids = {fields(line).get("pid") for line in obs.leases.values()}
    return len(obs.leases) == n and len(pids) == 1


@dataclass(frozen=True)
class Lease:
    pid: int
    gpus: tuple[int, ...]
    f: dict[str, str] = field(hash=False)


def leases(obs: Observation) -> list[Lease]:
    by: dict[int, list[int]] = {}
    lines: dict[int, dict[str, str]] = {}
    for i, line in obs.leases.items():
        f = fields(line)
        if "pid" not in f or "since" not in f:
            continue
        pid = int(f["pid"])
        by.setdefault(pid, []).append(i)
        lines.setdefault(pid, f)
    return [Lease(pid, tuple(sorted(gs)), lines[pid]) for pid, gs in sorted(by.items())]


def job_of(cluster: Cluster, f: dict[str, str], gpus: int, gpu_ids: tuple[int, ...], node: str = NODE,
           whole_node: bool = False) -> Job:
    """The job a lease or a request is, by the module's mapping."""
    since = epoch(f["since"])
    until = epoch(f["until"]) if f.get("until") else None
    timed = f.get("timed") == "1" or whole_node
    pre = f.get("preempt") == "1" and not timed
    max_s = until - since if until is not None else 60 * int(f["max"]) if f.get("max", "").isdigit() else None
    mem = f.get("mem", "")
    return Job(id=f"gl-{f['pid']}", project=f.get("project", cluster.nodes[node].owner), submitter=f.get("who", "-"),
               priority="timed" if timed else "fill" if pre else "session",
               kind="batch" if max_s is not None else "session", gpus=gpus, cpus=0,
               memory_gib=int(mem[:-1]) if mem.endswith("G") and mem[:-1].isdigit() and int(mem[:-1]) > 0 else 1,
               max_s=max_s, expected_s=None if max_s is not None else EXPECTED_S,
               preempt="requeue" if pre else "never", quiet=timed, node=node, gpu_ids=gpu_ids, submitted=since,
               run=f["run"] if f.get("run", "-") != "-" else None)


def allocations(cluster: Cluster, obs: Observation, node: str = NODE) -> list[Allocation]:
    n = len(cluster.nodes[node].gpus)
    out = []
    for ls in leases(obs):
        job = job_of(cluster, ls.f, len(ls.gpus), ls.gpus, node, whole_node=len(ls.gpus) == n)
        since = epoch(ls.f["since"])
        out.append(Allocation(id=f"{job.id}@{since}", job=job, node=node, gpus=ls.gpus, cpus=frozenset(),
                              memory_gib=job.memory_gib, started=since))
    return out


def unowned(cluster: Cluster, gpus: tuple[int, ...], node: str = NODE) -> Allocation:
    """GPUs whose lock is held under a stale owner line (a lease being taken): held, by nothing the planner may move."""
    job = Job(id=f"gl-unowned-{'-'.join(map(str, gpus))}", project=cluster.nodes[node].owner, submitter="-",
              priority="session", kind="session", gpus=len(gpus), cpus=0, expected_s=60, preempt="never", node=node,
              gpu_ids=gpus)
    return Allocation(id=f"{job.id}@0", job=job, node=node, gpus=gpus, cpus=frozenset(), memory_gib=1, started=0)


def queue(cluster: Cluster, obs: Observation, node: str = NODE) -> list[tuple[Job, str]]:
    """The live waiters as jobs, each with its handle (the wait file). A waiter already holding GPUs isn't one."""
    n = len(cluster.nodes[node].gpus)
    holding = {ls.pid for ls in leases(obs)}
    out = []
    for name, line in obs.waits:
        f = fields(line)
        if "pid" not in f or "since" not in f or not f.get("n", "").isdigit() or int(f["pid"]) in holding:
            continue
        want = int(f["n"])
        on = tuple(int(x) for x in f.get("on", "").split(",") if x.isdigit())
        job = job_of(cluster, f, want, on if len(on) == want else (), node, whole_node=want == n)
        out.append((replace(job, gpu_pool=on) if on and len(on) != want else job, name))
    return out


def preemption(line: str) -> dict[str, Any] | None:
    """A `preempted.log` line: when, the signal, the GPU, the requester's pid and the stopped lease's owner line."""
    parts = line.strip().split(" ", 7)
    if len(parts) < 8 or parts[2] != "GPU" or parts[4:6] != ["for", "pid"] or not (parts[3] + parts[6]).isdigit():
        return None
    head, _, owner = parts[7].partition("): ")
    return {"t": epoch(parts[0]), "signal": parts[1], "gpu": int(parts[3]), "by": int(parts[6]),
            "requester": head.lstrip("("), "victim": fields(owner)}


__all__ = ["NODE", "LEASE_DIR", "Observation", "Lease", "fields", "from_sampler", "held_by", "locked", "read", "timed_now",
           "leases", "job_of", "allocations", "unowned", "queue", "preemption"]
