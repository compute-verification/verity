"""The cluster's description, its jobs and their allocations.

A **cluster** is a set of nodes, each owned by one project, plus the priority classes and the **shares** under which a
project runs on a node it doesn't own. Nothing reaches a node except as a **job**: a batch job (GPU, CPU or Lean work),
a long-lived service, a time-boxed interactive session, or a data-staging job. A job that runs holds an **allocation**:
exact GPUs, CPUs and memory on one node from a start time, running or frozen.

Standing decides displacement before priority does: on its own node a project's job outranks every guest's, and a guest
runs only on idle capacity, only preemptibly, and never beside a quiet job. Inside an owner, a **workstream** owns some of
its node's GPUs: its jobs within those GPUs outrank every job running on GPUs it borrowed, whatever their priority, so
lent capacity comes back when the lender's work arrives. Times are integer seconds since the epoch, always passed in;
nothing here reads a clock.
"""
from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import cpuset

KINDS = ("batch", "service", "session", "stage")
PREEMPT = ("never", "freeze", "requeue")
OWNER, GUEST = "owner", "guest"


class ModelError(ValueError):
    pass


def instant(value: str | int | None) -> int | None:
    """An instant as epoch seconds, from an int or an ISO 8601 UTC string (`2026-10-07T15:00:00Z`)."""
    if value is None or isinstance(value, int):
        return value
    t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ModelError(f"instant {value!r} has no time zone")
    return int(t.timestamp())


@dataclass(frozen=True)
class Gpu:
    index: int
    numa: int
    uuid: str = ""


@dataclass(frozen=True)
class Node:
    name: str
    owner: str
    gpus: tuple[Gpu, ...]
    numa: dict[int, frozenset[int]]
    system_cpus: frozenset[int]
    memory_gib: int
    stops_at: int | None = None
    drain_s: int = 0
    smt_pairs: bool = True
    pools: dict[str, frozenset[int]] = field(default_factory=dict)
    gpu_model: str = ""
    gpu_executor: str = ""  # what runs a GPU job here besides gpu-lease's own node: "n1-lease" (node 1's Kueue-lent leases)

    @property
    def cpus(self) -> frozenset[int]:
        """The CPUs jobs may be given: every NUMA node's, less the system's."""
        return frozenset().union(*self.numa.values()) - self.system_cpus

    @property
    def pooled_cpus(self) -> frozenset[int]:
        """CPUs reserved to named pools: only a job naming its pool gets them, even while they sit idle."""
        return frozenset().union(*self.pools.values())

    @property
    def admission_ends(self) -> int | None:
        """No allocation may run past this instant: the stop less the drain (final backups, custody)."""
        return None if self.stops_at is None else self.stops_at - self.drain_s

    def numa_of_cpu(self, cpu: int) -> int:
        return next(n for n, cs in self.numa.items() if cpu in cs)

    def gpu(self, index: int) -> Gpu:
        return next(g for g in self.gpus if g.index == index)


@dataclass(frozen=True)
class Share:
    """What a project may hold on a node it doesn't own. Every limit is a cap; `None` leaves it to the node's size."""
    project: str
    node: str
    cpus: frozenset[int] | None = None
    max_gpus: int | None = None
    max_cpus: int | None = None
    max_memory_gib: int | None = None
    max_jobs: int | None = None
    max_job_memory_gib: int | None = None
    until: int | None = None


@dataclass(frozen=True)
class Workstream:
    """A part of a node's owner that owns `gpus` of the node's GPUs (Kueue's nominal quota). Its jobs within them run in
    share; beyond them they borrow idle GPUs, at most `borrow_gpus` more (`None`: any; 0: it never borrows), and a job
    running in anyone's share outranks every borrower."""
    name: str
    node: str
    gpus: int
    borrow_gpus: int | None = None


@dataclass(frozen=True)
class Cluster:
    name: str
    nodes: dict[str, Node]
    priorities: dict[str, int]
    shares: tuple[Share, ...] = ()
    workstreams: tuple[Workstream, ...] = ()

    def share(self, project: str, node: str) -> Share | None:
        return next((s for s in self.shares if s.project == project and s.node == node), None)

    def workstream(self, name: str | None, node: str) -> Workstream | None:
        return next((w for w in self.workstreams if w.name == name and w.node == node), None)

    def standing(self, project: str, node: str) -> str | None:
        if self.nodes[node].owner == project:
            return OWNER
        return GUEST if self.share(project, node) else None

    def rank(self, job: "Job", node: str, in_share: bool = False) -> tuple[int, int, int]:
        """Who displaces whom on `node`: standing first, then whether the job runs within its workstream's own GPUs
        (`in_share`, which only the planner knows, from what else runs), then priority. Only a strictly higher rank displaces."""
        return (1 if self.standing(job.project, node) == OWNER else 0, 1 if in_share else 0, self.priorities[job.priority])


@dataclass(frozen=True)
class Job:
    id: str
    project: str
    submitter: str
    priority: str
    kind: str = "batch"
    gpus: int = 0
    gpu_model: str | None = None
    cpus: int = 1
    memory_gib: int = 1
    max_s: int | None = None
    expected_s: int | None = None
    preempt: str = "requeue"
    quiet: bool = False
    quiet_safe: bool = False
    node: str | None = None
    gpu_ids: tuple[int, ...] = ()
    gpu_pool: tuple[int, ...] = ()
    locks: frozenset[str] = frozenset()
    inputs: frozenset[str] = frozenset()
    after: tuple[str, ...] = ()
    pool: str | None = None
    workstream: str | None = None
    submitted: int = 0
    run: str | None = None

    @property
    def length_s(self) -> int | None:
        """How long it is planned to run: `max_s`, else a session's `expected_s` (an estimate for backfill, never enforced)."""
        return self.max_s if self.max_s is not None else self.expected_s

    def problems(self, cluster: Cluster) -> list[str]:
        """Why this job can never run anywhere in `cluster`, by rule; empty when it is well formed.

        `cpus = 0` is a CPU-unmanaged job: the planner gives it no CPUs and its processes run wherever they are placed
        outside it (node 2's `gpu-lease` leases, which aren't pinned). A session may have no `max_s` if it has an
        `expected_s`: nothing ends it but itself and the node's admission end."""
        out = []
        if self.priority not in cluster.priorities:
            out.append(f"unknown priority class {self.priority!r}")
        if self.kind not in KINDS:
            out.append(f"unknown kind {self.kind!r}")
        if self.preempt not in PREEMPT:
            out.append(f"unknown preempt mode {self.preempt!r}")
        if self.kind == "service":
            if self.max_s is not None:
                out.append("a service has no max_s: it runs until its node drains")
        elif self.kind == "session" and self.max_s is None:
            if self.expected_s is None or self.expected_s <= 0:
                out.append("a session without max_s declares expected_s, the length backfill plans around")
        elif self.max_s is None or self.max_s <= 0:
            out.append("every job but a service declares max_s, the time it is killed at")
        if self.expected_s is not None and self.kind != "session":
            out.append("only a session has an expected_s instead of a hard max_s")
        if self.cpus < 0 or (self.cpus == 0 and self.gpus == 0):
            out.append("a job needs at least one CPU, or GPUs with cpus = 0 (CPU-unmanaged)")
        if self.cpus == 0 and self.pool is not None:
            out.append("a CPU-unmanaged job names no CPU pool")
        if self.gpus < 0 or self.memory_gib < 1:
            out.append("gpus must be >= 0 and memory_gib >= 1")
        if self.quiet_safe and self.kind != "service":
            out.append("only a service may be quiet_safe (it runs beside quiet jobs)")
        if self.quiet and self.kind not in ("batch", "session"):
            out.append("only a batch job or a session may be quiet")
        if self.gpu_ids and len(self.gpu_ids) != self.gpus:
            out.append(f"gpu_ids names {len(self.gpu_ids)} GPUs for a job of {self.gpus}")
        if self.gpu_pool and len(set(self.gpu_pool)) < self.gpus:
            out.append(f"gpu_pool offers {len(set(self.gpu_pool))} GPUs for a job of {self.gpus}")
        if self.node is not None and self.node not in cluster.nodes:
            out.append(f"unknown node {self.node!r}")
        if self.workstream is not None and not any(w.name == self.workstream for w in cluster.workstreams):
            out.append(f"unknown workstream {self.workstream!r}")
        return out


@dataclass(frozen=True)
class Allocation:
    id: str
    job: Job
    node: str
    gpus: tuple[int, ...]
    cpus: frozenset[int]
    memory_gib: int
    started: int
    state: str = "running"
    frozen_by: str | None = None
    frozen_s: int = 0
    frozen_at: int | None = None

    def running_s(self, now: int) -> int:
        """Time it has run: frozen time doesn't count against max_s, since a quiet window, not the job, stopped it."""
        return now - self.started - self.frozen_s - (now - self.frozen_at if self.frozen_at is not None else 0)

    @property
    def ends_by(self) -> int | None:
        """When it is sure to be gone if it isn't frozen again: its start plus max_s plus the time it has been frozen. For a
        session with only an `expected_s`, when it is expected to be gone (a plan for backfill, not a promise)."""
        n = self.job.length_s
        return None if n is None else self.started + self.frozen_s + n


def load(path: str | Path) -> Cluster:
    return from_dict(tomllib.loads(Path(path).read_text()))


def from_dict(d: dict[str, Any]) -> Cluster:
    meta = d.get("cluster", {})
    priorities = {k: int(v) for k, v in d.get("priorities", {}).items()}
    nodes = {}
    for name, nd in d.get("nodes", {}).items():
        numa = {int(k): cpuset.parse(v) for k, v in nd["numa"].items()}
        nodes[name] = Node(
            name=name, owner=nd["owner"],
            gpus=tuple(Gpu(int(g["index"]), int(g["numa"]), g.get("uuid", "")) for g in nd.get("gpus", [])),
            numa=numa, system_cpus=cpuset.parse(nd.get("system_cpus", "")), memory_gib=int(nd["memory_gib"]),
            stops_at=instant(nd.get("stops_at")), drain_s=int(nd.get("drain_s", meta.get("drain_s", 0))),
            smt_pairs=bool(nd.get("smt_pairs", True)),
            pools={k: cpuset.parse(v) for k, v in nd.get("pools", {}).items()}, gpu_model=str(nd.get("gpu_model", "")),
            gpu_executor=str(nd.get("gpu_executor", "")))
    shares = tuple(
        Share(project=s["project"], node=s["node"], cpus=cpuset.parse(s["cpus"]) if "cpus" in s else None,
              max_gpus=s.get("max_gpus"), max_cpus=s.get("max_cpus"), max_memory_gib=s.get("max_memory_gib"),
              max_jobs=s.get("max_jobs"), max_job_memory_gib=s.get("max_job_memory_gib"), until=instant(s.get("until")))
        for s in d.get("shares", []))
    workstreams = tuple(Workstream(name=w["name"], node=w["node"], gpus=int(w["gpus"]), borrow_gpus=w.get("borrow_gpus"))
                        for w in d.get("workstreams", []))
    cluster = Cluster(name=meta.get("name", "cluster"), nodes=nodes, priorities=priorities, shares=shares,
                      workstreams=workstreams)
    problems = validate(cluster)
    if problems:
        raise ModelError("; ".join(problems))
    return cluster


def validate(cluster: Cluster) -> list[str]:
    out = []
    if not cluster.priorities:
        out.append("no priority classes")
    for n in cluster.nodes.values():
        seen: set[int] = set()
        for k, cs in n.numa.items():
            if seen & cs:
                out.append(f"{n.name}: NUMA node {k} overlaps another ({cpuset.format(seen & cs)})")
            seen |= cs
        if not n.system_cpus <= seen:
            out.append(f"{n.name}: system CPUs {cpuset.format(n.system_cpus - seen)} are on no NUMA node")
        idx = [g.index for g in n.gpus]
        if len(set(idx)) != len(idx):
            out.append(f"{n.name}: duplicate GPU index")
        for g in n.gpus:
            if g.numa not in n.numa:
                out.append(f"{n.name}: GPU {g.index} is on unknown NUMA node {g.numa}")
        if n.memory_gib < 1:
            out.append(f"{n.name}: memory_gib must be >= 1")
        pooled: set[int] = set()
        for k, cs in sorted(n.pools.items()):
            if not cs <= n.cpus:
                out.append(f"{n.name}: pool {k} holds CPUs {cpuset.format(cs - n.cpus)} that aren't the node's to give")
            if pooled & cs:
                out.append(f"{n.name}: pool {k} overlaps another pool ({cpuset.format(pooled & cs)})")
            pooled |= cs
    for s in cluster.shares:
        if s.node not in cluster.nodes:
            out.append(f"share for {s.project} names unknown node {s.node!r}")
            continue
        node = cluster.nodes[s.node]
        if node.owner == s.project:
            out.append(f"share for {s.project} on {s.node}: the project owns that node")
        if s.cpus is not None and not s.cpus <= node.cpus:
            out.append(f"share for {s.project} on {s.node}: CPUs {cpuset.format(s.cpus - node.cpus)} aren't the node's to give")
    pairs = [(s.project, s.node) for s in cluster.shares]
    if len(set(pairs)) != len(pairs):
        out.append("two shares for one project on one node")
    for w in cluster.workstreams:
        if w.node not in cluster.nodes:
            out.append(f"workstream {w.name} names unknown node {w.node!r}")
        elif w.gpus < 0 or (w.borrow_gpus is not None and w.borrow_gpus < 0):
            out.append(f"workstream {w.name}: gpus and borrow_gpus must be >= 0")
    for name in {w.node for w in cluster.workstreams if w.node in cluster.nodes}:
        own = sum(w.gpus for w in cluster.workstreams if w.node == name)
        if own > len(cluster.nodes[name].gpus):
            out.append(f"{name}: its workstreams own {own} GPUs of {len(cluster.nodes[name].gpus)}")
    names = [(w.name, w.node) for w in cluster.workstreams]
    if len(set(names)) != len(names):
        out.append("two workstreams of one name on one node")
    return out


def job_from_dict(d: dict[str, Any]) -> Job:
    known = {f.name for f in fields(Job)}
    extra = set(d) - known
    if extra:
        raise ModelError(f"unknown job fields {sorted(extra)}")
    d = dict(d)
    for k in ("locks", "inputs"):
        if k in d:
            d[k] = frozenset(d[k])
    for k in ("gpu_ids", "gpu_pool", "after"):
        if k in d:
            d[k] = tuple(d[k])
    if "submitted" in d:
        d["submitted"] = instant(d["submitted"])
    return Job(**d)


def job_to_dict(job: Job) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(Job):
        v = getattr(job, f.name)
        if v == f.default:
            continue
        out[f.name] = sorted(v) if isinstance(v, frozenset) else list(v) if isinstance(v, tuple) else v
    return out


def allocation_from_dict(d: dict[str, Any]) -> Allocation:
    return Allocation(id=d["id"], job=job_from_dict(d["job"]), node=d["node"], gpus=tuple(d.get("gpus", ())),
                      cpus=cpuset.parse(d.get("cpus", "")), memory_gib=int(d["memory_gib"]), started=instant(d["started"]),
                      state=d.get("state", "running"), frozen_by=d.get("frozen_by"), frozen_s=int(d.get("frozen_s", 0)),
                      frozen_at=d.get("frozen_at"))


def allocation_to_dict(a: Allocation) -> dict[str, Any]:
    out = {"id": a.id, "job": job_to_dict(a.job), "node": a.node, "gpus": list(a.gpus), "cpus": cpuset.format(a.cpus),
           "memory_gib": a.memory_gib, "started": a.started, "state": a.state}
    for k in ("frozen_by", "frozen_s", "frozen_at"):
        if getattr(a, k):
            out[k] = getattr(a, k)
    return out
