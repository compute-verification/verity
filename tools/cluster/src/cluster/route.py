"""Placement: which node's queue a job enters.

Each node's planner is authoritative for its node (`plan.plan`); routing only picks the queue, so a node keeps scheduling its
own queue when the router is down. The order of preference:

1. a node where the job starts now without displacing anything, its owner's node first;
2. then its owner's node, where it may displace guests or wait with a reservation;
3. then any node that holds its inputs, then by name.

A pinned job goes to its node or nowhere. A node that could never run it (`plan.static_refusal`) is skipped, so a job that
would cross every node's admission end is refused here, with each node's reason.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import OWNER, Allocation, Cluster, Job
from .plan import fit, static_refusal


@dataclass(frozen=True)
class View:
    """What routing reads of a node: its live allocations and the content staged there."""
    allocations: tuple[Allocation, ...] = ()
    has: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Placement:
    node: str | None
    why: str


def place(cluster: Cluster, job: Job, views: dict[str, View], now: int,
          held_locks: frozenset[str] = frozenset()) -> Placement:
    refusals = []
    scored = []
    for name in sorted(cluster.nodes):
        why = static_refusal(cluster, name, job, now)
        if why:
            refusals.append(f"{name}: {why}")
            continue
        view = views.get(name, View())
        quiet = any(a.job.quiet and a.state == "running" for a in view.allocations)
        fits_now = not quiet and not isinstance(fit(cluster, name, job, view.allocations, held_locks), str)
        owner = cluster.standing(job.project, name) == OWNER
        local = job.inputs <= view.has
        scored.append(((not fits_now, not owner, not local, name), name, fits_now, owner, local))
    if not scored:
        return Placement(None, "; ".join(refusals) or "no nodes")
    _, name, fits_now, owner, local = min(scored)
    why = ", ".join(w for w, on in (("starts now", fits_now), ("owner's node", owner), ("inputs staged", local and bool(job.inputs))) if on)
    return Placement(name, why or "queues behind current work")
