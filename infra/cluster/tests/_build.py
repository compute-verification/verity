"""Small clusters and jobs for the tests: two NUMA nodes of 16 CPUs, GPUs 0-3 on the first and 4-7 on the second."""
from __future__ import annotations

from cluster import cpuset
from cluster.model import Allocation, Cluster, Gpu, Job, Node, Share, Workstream

T0 = 1_790_000_000
PRIO = {"service": 1000, "timed": 900, "session": 700, "capture": 600, "work": 500, "bench": 300, "dev": 100, "fill": 10}


def node(name: str, owner: str, *, gpus: int = 8, memory: int = 256, stops_at: int | None = None, drain_s: int = 0,
         system: str = "", pools: dict[str, str] | None = None) -> Node:
    return Node(name=name, owner=owner, gpus=tuple(Gpu(i, 0 if i < gpus // 2 or gpus == 1 else 1) for i in range(gpus)),
                numa={0: cpuset.parse("0-15"), 1: cpuset.parse("16-31")}, system_cpus=cpuset.parse(system),
                memory_gib=memory, stops_at=stops_at, drain_s=drain_s,
                pools={k: cpuset.parse(v) for k, v in (pools or {}).items()})


def cluster(*nodes: Node, shares: tuple[Share, ...] = (), workstreams: tuple[Workstream, ...] = ()) -> Cluster:
    return Cluster(name="test", nodes={n.name: n for n in nodes}, priorities=dict(PRIO), shares=shares, workstreams=workstreams)


def one(owner: str = "p", **kw) -> Cluster:
    return cluster(node("n1", owner, **kw))


def job(id: str, project: str = "p", **kw) -> Job:
    kw.setdefault("submitter", id.split("-")[0])
    kw.setdefault("priority", "work")
    kw.setdefault("cpus", 2)
    kw.setdefault("memory_gib", 8)
    if kw.get("kind") != "service":
        kw.setdefault("max_s", 600)
    kw.setdefault("submitted", T0)
    return Job(id=id, project=project, **kw)


def alloc(j: Job, gpus: tuple[int, ...] = (), cpus: str = "", *, node: str = "n1", started: int = T0,
          state: str = "running") -> Allocation:
    return Allocation(id=f"{j.id}@{started}", job=j, node=node, gpus=gpus, cpus=cpuset.parse(cpus), memory_gib=j.memory_gib,
                      started=started, state=state)
