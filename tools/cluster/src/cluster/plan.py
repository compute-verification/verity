"""The node planner: one pure function from a node's allocations and queue to the actions that keep its rules.

`plan(cluster, node, allocations, queue, now)` returns actions in the order an executor applies them: expiries, evictions,
freezes, starts, thaws, then the jobs that wait or are refused, each with its reason. It reads no clock and no files, so
the same call decides on the node, in a simulation, or in a shadow run against today's nodes.

The rules, each enforced here and checked by `tests/test_simulation.py` on random workloads:

1. **Exclusive resources.** A GPU, a CPU or a named lock is in at most one allocation; memory is never overcommitted; a
   guest's CPUs come from its share; a node's reserved CPU pools go only to jobs naming them, even while idle.
2. **Standing, then share, then priority.** A job displaces an allocation only when its rank (`Cluster.rank`: owner before
   guest, then running within its workstream's own GPUs before borrowing, then priority class) is strictly higher, and only
   a `requeue` allocation is evicted. A guest must be preemptible and may not be quiet, so an owner's work always gets its
   node back; a workstream's work within its own GPUs gets them back from any borrower, whatever its priority. A
   workstream never runs beyond its GPUs plus its `borrow_gpus`.
3. **Quiet jobs** (timed windows) run with nothing else active on their node: GPU-holding allocations are evicted (a frozen
   CUDA context still holds its GPU), CPU-only ones frozen and thawed afterwards; only a `quiet_safe` service keeps running.
   A quiet job blocked by an allocation it may not displace waits, and holds the node's reservation. A quiet job asking
   for some but not all of its node's GPUs (a 1-GPU `gpu-lease --timed`) still clears the rest of the node, but starts
   beside what it may not displace when it fits, as `gpu-lease` starts it: a design divergence kept until the 7 Oct clamp
   (infra's and PoUW's rulings).
4. **Reservation and backfill.** The highest-ranked blocked job with a known start (the ends of the allocations it may not
   displace) reserves its node from then. Lower jobs start ahead of it only if they end before then or it may evict them,
   so a whole-node request is never starved and GPUs never idle behind it.
5. **Deadlines.** Every job but a service declares `max_s`, and none is admitted to run past its node's admission end
   (stop less drain) or its share's `until`. A job expires once it has run `max_s` (time frozen by a quiet window doesn't
   count), and every job but a service expires at the admission end, frozen or not.
6. **Fair turns.** Among equals, the submitter with the fewest allocations on the node goes first, then the oldest job.
7. **Idle leases** (`idle`): GPUs held without computing are named to their holder and recorded, never killed or refused
   for it; the fixes are structural (preemptible fill where GPUs are free, CPU preparation in its own GPU-less job).

A lease taken by hand over SSH (a kernel inner loop, a debugger) is a `session` allocation like any other, so the same
rules see it: it holds its GPUs until it ends, a quiet job waits for it unless it is preemptible, and it shows when idle.
A session without a `max_s` plans with its `expected_s`: a reservation behind it expects it gone then (or now, once it
has overrun), and only jobs with a hard `max_s` backfill ahead of a reservation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Union

from .model import GUEST, OWNER, Allocation, Cluster, Job, Node


@dataclass(frozen=True)
class Start:
    alloc: Allocation


@dataclass(frozen=True)
class Evict:
    alloc: str
    by: str


@dataclass(frozen=True)
class Freeze:
    alloc: str
    by: str


@dataclass(frozen=True)
class Thaw:
    alloc: str


@dataclass(frozen=True)
class Expire:
    alloc: str


@dataclass(frozen=True)
class Wait:
    job: str
    reason: str
    start_by: int | None = None


@dataclass(frozen=True)
class Refuse:
    job: str
    reason: str


@dataclass(frozen=True)
class Idle:
    """An allocation holding GPUs that computed nothing for `idle_s`: named, so its owner and everyone else sees it."""
    alloc: str
    submitter: str
    idle_s: int


Action = Union[Start, Evict, Freeze, Thaw, Expire, Wait, Refuse, Idle]


def static_refusal(cluster: Cluster, node: str, job: Job, now: int) -> str | None:
    """Why `job` can never start on `node` from `now` on, whatever else runs there; None if it might."""
    problems = job.problems(cluster)
    if problems:
        return "; ".join(problems)
    if job.node is not None and job.node != node:
        return f"pinned to {job.node}"
    n = cluster.nodes[node]
    if job.gpus and job.gpu_model and job.gpu_model != n.gpu_model:
        return f"its GPUs are {n.gpu_model or 'undescribed'}, not {job.gpu_model}"
    standing = cluster.standing(job.project, node)
    if standing is None:
        return f"{job.project} has no share on {node}"
    length = job.length_s
    if standing == GUEST:
        share = cluster.share(job.project, node)
        if job.quiet:
            return f"only {n.owner} runs quiet jobs on {node}"
        if job.preempt == "never":
            return f"a guest on {node} must be preemptible"
        if share.max_job_memory_gib is not None and job.memory_gib > share.max_job_memory_gib:
            return f"{job.project}'s jobs on {node} get at most {share.max_job_memory_gib} GiB each"
        if share.until is not None and (now >= share.until or (length is not None and now + length > share.until)):
            return f"{job.project}'s share on {node} ends at {share.until}"
    if job.workstream is not None:
        w = cluster.workstream(job.workstream, node)
        if w is None or standing != OWNER:
            return f"workstream {job.workstream} is not {job.project}'s on {node}"
        if w.borrow_gpus is not None and job.gpus > w.gpus + w.borrow_gpus:
            return f"larger than workstream {w.name} may hold on {node} ({w.gpus} GPUs + {w.borrow_gpus} borrowed)"
    ends = n.admission_ends
    if ends is not None and (now >= ends or (length is not None and now + length > ends)):
        return f"would run past {node}'s admission end {ends}"
    if job.pool is not None and job.pool not in n.pools:
        return f"{node} has no CPU pool {job.pool!r}"
    allowed = allowed_cpus(cluster, n, job)
    if job.gpus > len(n.gpus) or job.cpus > len(allowed) or job.memory_gib > n.memory_gib:
        return f"larger than {node} allows ({job.gpus} GPUs, {job.cpus} CPUs, {job.memory_gib} GiB)"
    if set(job.gpu_ids) - {g.index for g in n.gpus}:
        return f"pins GPUs {sorted(set(job.gpu_ids) - {g.index for g in n.gpus})} that {node} lacks"
    if job.gpu_pool and len(set(job.gpu_pool) & {g.index for g in n.gpus}) < job.gpus:
        return f"its GPU pool {sorted(job.gpu_pool)} has fewer than {job.gpus} of {node}'s GPUs"
    return None


def fit(cluster: Cluster, node: str, job: Job, occupants: Iterable[Allocation],
        held_locks: frozenset[str] = frozenset()) -> tuple[tuple[int, ...], frozenset[int]] | str:
    """The GPUs and CPUs `job` would get beside `occupants`, or why it doesn't fit."""
    n = cluster.nodes[node]
    occ = list(occupants)
    used_gpus = {g for a in occ for g in a.gpus}
    used_cpus = frozenset().union(*(a.cpus for a in occ))
    if job.memory_gib > n.memory_gib - sum(a.memory_gib for a in occ):
        return "no memory free"
    taken = held_locks | {lk for a in occ for lk in a.job.locks}
    if job.locks & taken:
        return f"lock {sorted(job.locks & taken)[0]} is held"
    if cluster.standing(job.project, node) == GUEST:
        share = cluster.share(job.project, node)
        mine = [a for a in occ if a.job.project == job.project]
        caps = ((share.max_gpus, sum(len(a.gpus) for a in mine) + job.gpus, "GPUs"),
                (share.max_cpus, sum(len(a.cpus) for a in mine) + job.cpus, "CPUs"),
                (share.max_memory_gib, sum(a.memory_gib for a in mine) + job.memory_gib, "memory"),
                (share.max_jobs, len(mine) + 1, "jobs"))
        for cap, want, what in caps:
            if cap is not None and want > cap:
                return f"at {job.project}'s share of {what} on {node}"
    w = cluster.workstream(job.workstream, node)
    if w is not None and w.borrow_gpus is not None and job.gpus:
        mine = sum(len(a.gpus) for a in occ if a.job.workstream == w.name)
        if mine + job.gpus > w.gpus + w.borrow_gpus:
            return f"at workstream {w.name}'s GPUs on {node}"
    free_gpus = sorted(({g.index for g in n.gpus} & set(job.gpu_pool or (g.index for g in n.gpus))) - used_gpus)
    if job.gpu_ids:
        if set(job.gpu_ids) & used_gpus:
            return "pinned GPUs are held"
        gpus = tuple(job.gpu_ids)
    else:
        if len(free_gpus) < job.gpus:
            return "no GPUs free"
        gpus = _pick_gpus(n, free_gpus, job.gpus)
    free_cpus = allowed_cpus(cluster, n, job) - used_cpus
    if len(free_cpus) < job.cpus:
        return "no CPUs free"
    return gpus, _pick_cpus(n, free_cpus, job.cpus, gpus)


def plan(cluster: Cluster, node: str, allocations: Iterable[Allocation], queue: Iterable[Job], now: int, *,
         done: frozenset[str] = frozenset(), has: frozenset[str] = frozenset(),
         held_locks: frozenset[str] = frozenset()) -> list[Action]:
    """The actions for `node` at `now` (module docstring). `done` holds the ids of jobs that ended well, `has` the content
    ids staged on the node, and `held_locks` the locks allocations elsewhere in the cluster hold."""
    live = {a.id: a for a in sorted(allocations, key=lambda a: a.id) if a.node == node}
    ends = cluster.nodes[node].admission_ends
    expires = [Expire(a.id) for a in live.values()
               if (a.job.max_s is not None and a.running_s(now) >= a.job.max_s)
               or (ends is not None and now >= ends and a.job.kind != "service")]
    for e in expires:
        del live[e.alloc]
    evicts: list[Evict] = []
    freezes: list[Freeze] = []
    starts: list[Start] = []
    later: list[Action] = []

    def occupants() -> list[Allocation]:
        gone = {e.alloc for e in evicts}
        return [a for a in live.values() if a.id not in gone] + [s.alloc for s in starts]

    def quiet_now() -> bool:
        return any(a.job.quiet and a.state == "running" for a in occupants())

    def arank(a: Allocation) -> tuple:
        return cluster.rank(a.job, node, a.id in in_share(cluster, node, occupants()))

    def jrank(job: Job) -> tuple:
        return cluster.rank(job, node, job_in_share(cluster, node, job, occupants()))

    pending = []
    for job in sorted(queue, key=lambda j: j.id):
        why = static_refusal(cluster, node, job, now)
        if why:
            later.append(Refuse(job.id, why))
        elif set(job.after) - done:
            later.append(Wait(job.id, f"after {sorted(set(job.after) - done)[0]}"))
        elif job.inputs - has:
            later.append(Wait(job.id, f"inputs not staged: {', '.join(sorted(job.inputs - has))}"))
        else:
            pending.append(job)

    reservation: tuple[Job, int | None, tuple] | None = None

    def backfill_ok(job: Job) -> bool:
        if reservation is None:
            return True
        holder, start_by, holder_rank = reservation
        if start_by is not None and job.max_s is not None and now + job.max_s <= start_by:
            return True
        return jrank(job) < holder_rank and job.preempt == "requeue"

    def turns(job: Job) -> tuple:
        mine = sum(1 for a in occupants() if a.job.submitter == job.submitter)
        return (*(-x for x in jrank(job)), mine, job.submitted, job.id)

    while pending:
        job = min(pending, key=turns)
        pending.remove(job)
        if quiet_now():
            later.append(Wait(job.id, "node is quiet"))
            continue
        if not backfill_ok(job):
            later.append(Wait(job.id, f"{node} is reserved for {reservation[0].id}", reservation[1]))
            continue
        occ = occupants()
        if job.quiet:
            outcome = _quiet(cluster, node, job, occ, held_locks, arank, jrank(job), now)
        else:
            outcome = _ordinary(cluster, node, job, occ, held_locks, arank, jrank(job), now)
        if isinstance(outcome, Wait):
            if reservation is None and outcome.start_by is not None:
                reservation = (job, outcome.start_by, jrank(job))
            later.append(outcome)
            continue
        victims, frozen, (gpus, cpus) = outcome
        evicts += [Evict(v, job.id) for v in victims]
        freezes += [Freeze(f, job.id) for f in frozen]
        starts.append(Start(Allocation(id=f"{job.id}@{now}", job=job, node=node, gpus=gpus, cpus=cpus,
                                       memory_gib=job.memory_gib, started=now)))

    frozen_now = {f.alloc for f in freezes}
    thaws = [] if quiet_now() else [Thaw(a.id) for a in live.values()
                                    if a.state == "frozen" and a.id not in frozen_now and a.id not in {e.alloc for e in evicts}]
    return [*expires, *evicts, *freezes, *starts, *thaws, *later]


def idle(cluster: Cluster, node: str, allocations: Iterable[Allocation], last_busy: dict[str, int], now: int, *,
         idle_s: dict[str, int]) -> list[Action]:
    """Idle leases: GPUs held but not computing, reported and never acted on (interactive and debugging leases are bursty
    by nature). `last_busy[alloc]` is the last instant the node's sampler saw any of the allocation's GPUs busy (absent:
    never), and `idle_s[kind]` how long a kind may hold GPUs idle before it is named (a kind absent from it never is)."""
    out: list[Action] = []
    for a in sorted((a for a in allocations if a.node == node), key=lambda a: a.id):
        limit = idle_s.get(a.job.kind)
        if not a.gpus or a.state != "running" or limit is None:
            continue
        quiet_for = now - max(last_busy.get(a.id, a.started), a.started)
        if quiet_for >= limit:
            out.append(Idle(a.id, a.job.submitter, quiet_for))
    return out


def apply(allocations: Iterable[Allocation], actions: Iterable[Action], now: int) -> list[Allocation]:
    """The allocations after an executor carries out `actions` at `now` (a Wait, Refuse or Idle changes nothing)."""
    live = {a.id: a for a in allocations}
    for act in actions:
        if isinstance(act, (Evict, Expire)):
            live.pop(act.alloc, None)
        elif isinstance(act, Freeze):
            a = live[act.alloc]
            live[act.alloc] = replace(a, state="frozen", frozen_by=act.by, frozen_at=a.frozen_at if a.frozen_at is not None else now)
        elif isinstance(act, Thaw):
            a = live[act.alloc]
            live[act.alloc] = replace(a, state="running", frozen_by=None, frozen_at=None,
                                      frozen_s=a.frozen_s + now - (a.frozen_at if a.frozen_at is not None else now))
        elif isinstance(act, Start):
            live[act.alloc.id] = act.alloc
    return sorted(live.values(), key=lambda a: a.id)


def in_share(cluster: Cluster, node: str, allocations: Iterable[Allocation]) -> frozenset[str]:
    """The allocations running within their workstream's own GPUs: each workstream's oldest first, while their GPUs fit.
    The rest of a workstream's allocations borrow."""
    used: dict[str, int] = {}
    out = set()
    for a in sorted(allocations, key=lambda a: (a.started, a.id)):
        w = cluster.workstream(a.job.workstream, node)
        if w is not None and a.gpus and used.get(w.name, 0) + len(a.gpus) <= w.gpus:
            used[w.name] = used.get(w.name, 0) + len(a.gpus)
            out.add(a.id)
    return frozenset(out)


def job_in_share(cluster: Cluster, node: str, job: Job, occupants: Iterable[Allocation]) -> bool:
    """Whether `job` would start within its workstream's own GPUs beside `occupants`. A share is GPUs, so a job asking none
    has no standing from it."""
    w = cluster.workstream(job.workstream, node)
    if w is None or job.gpus == 0:
        return False
    occ = list(occupants)
    shared = in_share(cluster, node, occ)
    return sum(len(a.gpus) for a in occ if a.id in shared and a.job.workstream == w.name) + job.gpus <= w.gpus


def whole_quiet(cluster: Cluster, node: str, job: Job) -> bool:
    """A quiet job that waits for what it may not displace: one of every GPU, or of none (a CPU window); rule 3."""
    return job.quiet and not 0 < job.gpus < len(cluster.nodes[node].gpus)


def _ordinary(cluster, node, job, occ, held_locks, arank, mine, now):
    got = fit(cluster, node, job, occ, held_locks)
    if not isinstance(got, str):
        return [], [], got
    movable = sorted((a for a in occ if a.job.preempt == "requeue" and arank(a) < mine),
                     key=lambda a: (arank(a), -a.started, a.id))
    chosen: list[Allocation] = []
    for a in movable:
        chosen.append(a)
        if not isinstance(fit(cluster, node, job, _without(occ, chosen), held_locks), str):
            break
    else:
        return Wait(job.id, got, _start_by(cluster, node, job, occ, movable, held_locks, now))
    for a in list(reversed(chosen)):
        trial = [c for c in chosen if c is not a]
        if not isinstance(fit(cluster, node, job, _without(occ, trial), held_locks), str):
            chosen = trial
    return [a.id for a in chosen], [], fit(cluster, node, job, _without(occ, chosen), held_locks)


def _quiet(cluster, node, job, occ, held_locks, arank, mine, now):
    evict, freeze, blockers = [], [], []
    for a in occ:
        lower = arank(a) < mine
        if a.job.quiet_safe or a.state == "frozen":
            continue
        if a.gpus:
            (evict if lower and a.job.preempt == "requeue" else blockers).append(a)
        else:
            (freeze if lower and a.job.preempt in ("freeze", "requeue") else blockers).append(a)
    if blockers and not whole_quiet(cluster, node, job) and not isinstance(fit(cluster, node, job, _without(occ, evict),
                                                                                  held_locks), str):
        blockers = []
    if blockers:
        ends = [a.ends_by for a in blockers]
        start_by = None if None in ends else max(now, *ends)
        return Wait(job.id, f"quiet job waits for {', '.join(a.id for a in blockers)}", start_by)
    extra: list[Allocation] = []
    got = fit(cluster, node, job, _without(occ, evict), held_locks)
    for a in sorted((a for a in freeze if a.job.preempt == "requeue"), key=lambda a: (arank(a), -a.started, a.id)):
        if not isinstance(got, str):
            break
        extra.append(a)
        got = fit(cluster, node, job, _without(occ, evict + extra), held_locks)
    if isinstance(got, str):
        return Wait(job.id, f"quiet job: {got} beside frozen jobs", None)
    return [a.id for a in evict + extra], [a.id for a in freeze if a not in extra], got


def _start_by(cluster, node, job, occ, movable, held_locks, now) -> int | None:
    """The earliest instant `job` fits once the allocations it may not evict have ended in order of their ends (a session
    past its expected length is expected gone now, as `gpu-lease` counts a lease past its rule)."""
    staying = sorted((a for a in occ if a not in movable), key=lambda a: (a.ends_by is None, a.ends_by or 0))
    for i, a in enumerate(staying):
        if a.ends_by is None:
            return None
        if not isinstance(fit(cluster, node, job, staying[i + 1:], held_locks), str):
            return max(now, a.ends_by)
    return None


def _without(occ, gone):
    ids = {a.id for a in gone}
    return [a for a in occ if a.id not in ids]


def allowed_cpus(cluster: Cluster, n: Node, job: Job) -> frozenset[int]:
    """A job naming a pool gets that pool's CPUs; any other job gets the CPUs outside every pool. A guest's share narrows
    either."""
    base = n.pools.get(job.pool, frozenset()) if job.pool is not None else n.cpus - n.pooled_cpus
    share = cluster.share(job.project, n.name) if n.owner != job.project else None
    return base & share.cpus if share is not None and share.cpus is not None else base


def _pick_gpus(n: Node, free: list[int], count: int) -> tuple[int, ...]:
    """Lowest free indices, from the NUMA node with the most free GPUs first (so a job's GPUs share a host)."""
    by_numa: dict[int, list[int]] = {}
    for i in free:
        by_numa.setdefault(n.gpu(i).numa, []).append(i)
    fits = [k for k, v in by_numa.items() if len(v) >= count]
    order = sorted(fits, key=lambda k: (len(by_numa[k]), k)) if fits else sorted(by_numa, key=lambda k: (-len(by_numa[k]), k))
    out: list[int] = []
    for k in order:
        out += by_numa[k][: count - len(out)]
    return tuple(sorted(out))


def _pick_cpus(n: Node, free: frozenset[int], count: int, gpus: tuple[int, ...]) -> frozenset[int]:
    """Whole cores first, on the job's GPUs' NUMA node first, else on the NUMA node with the most free CPUs."""
    per_numa = {k: free & cs for k, cs in n.numa.items()}
    gpu_numa = [n.gpu(g).numa for g in gpus]
    order = sorted(per_numa, key=lambda k: (-gpu_numa.count(k), -len(per_numa[k]), k))
    out: list[int] = []
    for k in order:
        cs = per_numa[k]
        whole = [c for c in sorted(cs) if n.smt_pairs and (c ^ 1) in cs]
        single = [c for c in sorted(cs) if c not in whole]
        out += (whole + single)[: count - len(out)]
        if len(out) == count:
            break
    return frozenset(out)


__all__ = ["Action", "Evict", "Expire", "Freeze", "Idle", "OWNER", "Refuse", "Start", "Thaw", "Wait", "apply", "fit", "idle",
           "in_share", "job_in_share", "plan", "static_refusal", "allowed_cpus", "whole_quiet"]
