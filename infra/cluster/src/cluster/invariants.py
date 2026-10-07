"""The planner's rules as checks on a node's state: what the simulation asserts on every tick, and what the node agent checks
before it carries out a plan (it stops itself on a violation the plan would introduce).

`check(cluster, node, allocations, now, actions=(), before=())` returns one message per violation, each naming what broke,
so two states' violations compare as sets: the agent acts only when the planned state adds none to the observed one (a
node run by today's tools may already break a rule the planner keeps, such as memory the leases overcommit).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .model import Allocation, Cluster
from .plan import Action, Evict, Start, in_share, whole_quiet


def check(cluster: Cluster, node: str, allocations: Iterable[Allocation], now: int, *, actions: Sequence[Action] = (),
          before: Iterable[Allocation] = ()) -> list[str]:
    """Every rule `allocations` (the node's state after `actions`, carried out at `now` on `before`) breaks."""
    nd = cluster.nodes[node]
    mine = sorted((a for a in allocations if a.node == node), key=lambda a: a.id)
    out: list[str] = []
    for what, items in (("GPU", [g for a in mine for g in a.gpus]), ("CPU", [c for a in mine for c in a.cpus]),
                        ("lock", [lk for a in mine for lk in a.job.locks])):
        out += [f"{node}: {what} {k} is in {n} allocations" for k, n in sorted(Counter(items).items()) if n > 1]
    if not {g for a in mine for g in a.gpus} <= {g.index for g in nd.gpus}:
        out.append(f"{node}: an allocation holds a GPU the node lacks")
    if not {c for a in mine for c in a.cpus} <= nd.cpus:
        out.append(f"{node}: an allocation holds a CPU that isn't the node's to give")
    if sum(a.memory_gib for a in mine) > nd.memory_gib:
        out.append(f"{node}: memory overcommitted")
    for a in mine:
        if a.job.pool:
            if not a.cpus <= nd.pools.get(a.job.pool, frozenset()):
                out.append(f"{node}: {a.id} holds CPUs outside its pool {a.job.pool}")
        elif a.cpus & nd.pooled_cpus:
            out.append(f"{node}: {a.id} holds a reserved pool's CPUs")
    quiet = [a for a in mine if a.job.quiet and a.state == "running"]
    if len(quiet) > 1:
        out.append(f"{node}: two quiet jobs run at once")
    if quiet:
        q, whole, shared_now = quiet[0], whole_quiet(cluster, node, quiet[0].job), in_share(cluster, node, mine)
        out += [f"{node}: {a.id} runs beside quiet {q.id}" for a in mine
                if a is not q and a.state == "running" and not a.job.quiet_safe
                and (whole or _displaceable(cluster, node, q, a, shared_now))]
    else:
        out += [f"{node}: {a.id} is frozen with no quiet job" for a in mine if a.state == "frozen"]
    ends = nd.admission_ends
    if ends is not None and now >= ends:
        out += [f"{node}: {a.id} outlives the admission end" for a in mine if a.job.kind != "service"]
    starts = [s.alloc for s in actions if isinstance(s, Start)]
    for a in starts:
        n = a.job.length_s
        if ends is not None and a.job.kind != "service" and (n is None or a.started + n > ends):
            out.append(f"{node}: {a.id} admitted past the admission end")
        if any(b.job.quiet and b.state == "running" for b in before if b.node == node) and not a.job.quiet:
            out.append(f"{node}: {a.id} started inside a quiet window")
    for a in mine:
        if nd.owner != a.job.project:
            share = cluster.share(a.job.project, node)
            if share is None:
                out.append(f"{node}: {a.id} is {a.job.project}'s, which has no share here")
                continue
            if share.cpus is not None and not a.cpus <= share.cpus:
                out.append(f"{node}: guest {a.id} holds CPUs outside its share")
            if a.job.quiet or a.job.preempt == "never":
                out.append(f"{node}: guest {a.id} is quiet or not preemptible")
    for s in (s for s in cluster.shares if s.node == node):
        theirs = [a for a in mine if a.job.project == s.project]
        for cap, used, what in ((s.max_gpus, sum(len(a.gpus) for a in theirs), "GPUs"), (s.max_jobs, len(theirs), "jobs"),
                                (s.max_cpus, sum(len(a.cpus) for a in theirs), "CPUs"),
                                (s.max_memory_gib, sum(a.memory_gib for a in theirs), "memory")):
            if cap is not None and used > cap:
                out.append(f"{node}: {s.project}'s share of {what} exceeded")
    for w in (w for w in cluster.workstreams if w.node == node and w.borrow_gpus is not None):
        if sum(len(a.gpus) for a in mine if a.job.workstream == w.name) > w.gpus + w.borrow_gpus:
            out.append(f"{node}: workstream {w.name} holds more GPUs than it owns and may borrow")
    prior = sorted((b for b in before if b.node == node), key=lambda b: b.id)
    by_id = {b.id: b for b in prior}
    started_by = {a.job.id: a for a in starts}
    shared = in_share(cluster, node, prior)
    for e in (x for x in actions if isinstance(x, Evict)):
        victim, by = by_id.get(e.alloc), started_by.get(e.by)
        if victim is None or by is None:
            out.append(f"{node}: eviction of {e.alloc} by {e.by} names no allocation or no start")
            continue
        if victim.job.preempt != "requeue":
            out.append(f"{node}: {victim.id} evicted though it isn't preemptible")
        # the starter's own share status depends on the order it was planned in; granting it the higher one never
        # reports a sound eviction
        if cluster.rank(by.job, node, cluster.workstream(by.job.workstream, node) is not None) <= \
                cluster.rank(victim.job, node, victim.id in shared):
            out.append(f"{node}: {victim.id} evicted by an equal or lower rank ({e.by})")
    return out


def _displaceable(cluster: Cluster, node: str, q: Allocation, a: Allocation, shared: frozenset[str]) -> bool:
    """Whether quiet `q` may evict (GPU work) or freeze (CPU work) `a`: what a partial window clears (plan rule 3)."""
    lower = cluster.rank(a.job, node, a.id in shared) < cluster.rank(q.job, node, q.id in shared)
    return lower and (a.job.preempt == "requeue" if a.gpus else a.job.preempt in ("freeze", "requeue"))
