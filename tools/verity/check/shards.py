"""The shard planner: an auto plan (`check.py --shard auto`, `--shard-plan`) splits a check over as many runs as its critical path
allows, where ci.toml's `[[plan]]`s are written by hand (they stay, as fallbacks).

Its tasks are the check's parts with the test steps split by suite: each suite of pytest and of lean-suites (`pytest:SUITE`,
`lean-suites:SUITE`), each Lake package's audit (`lean-audit:PKG`), circuit-check's `circuit-check:K/M` (its ci.toml
`shards`), and every other job whole (the builds, lean-changed, ratchet-gate).  Each costs its measured seconds (`priced`, from
recorded verdicts and runs); one with none costs the median of its kind's measured tasks, else of every measured task, else
`DEFAULT_S`.

A part whose job `needs` or runs `after` another (lean-unit-cut, lean-agreement and lean-suites read lean-build's binaries,
flock-rust-tests flock-circuit-build's crates) runs on a shard that runs that build, and so does a spec package's audit, which
audits the packages its `proved_in` names beside it (`lean-audit:Security` and `lean-audit:Security/Properties` audit
`Security/Proofs`).  A cheap build (at most `CHEAP_S`) may run again, beside its parts, on each shard that holds a part reading
it and not the build's own part (`builds`), until a shared artifact cache exists; the plan says so and charges it.  A costlier
one holds its readers on its own shard (`units`).  The build's part is in exactly one shard, so a combine takes each part once.

Units are packed longest first (own cost and builds) onto the shard where they end soonest, the builds a shard lacks counted
(LPT); a shard's predicted time is the sum of its parts' and builds' costs (a shard runs its groups at once, so this is an upper
bound).  The floor is the longest unit with its builds: no N beats it.  N is the smallest N up to ``max_shards`` whose packed
makespan is within `SLACK` of max(floor, total / max_shards), the best bound the cap allows (else the N with the least
makespan).  Stdlib only, no I/O: check.py gathers the costs and the tree's parts.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass

KIND = "check-shard-plan/v1"
#: how far over the best bound a packing may be and still be taken
SLACK = 0.05
#: the most shards an auto plan uses unless told otherwise (`check.py --shard-plan --max-shards M`)
MAX_SHARDS = 6
#: a build costing at most this is run again on each shard that reads it; a costlier one holds its readers on its own shard
CHEAP_S = 300.0
#: a task's cost when nothing of any kind was measured
DEFAULT_S = 60.0
#: the jobs split by suite
SUITE_JOBS = ("pytest", "lean-suites")


@dataclass(frozen=True)
class Task:
    part: str
    seconds: float
    #: "measured", "median" (of its kind), "median-all" (of every measured task) or "default"
    source: str


def kind(part: str) -> str:
    """A part's kind, whose measured median prices an unmeasured part: a suite of either test step is one kind."""
    job = part.partition(":")[0]
    return "suite" if job in SUITE_JOBS else job


def parts(all_parts: list[str], suites: list[str], after_lean: list[str], lean: bool) -> list[str]:
    """The auto plan's parts from a check's (`declaration.all_parts`): `pytest` as `pytest:SUITE` for each suite (but those of
    ``after_lean`` when the tree has the Lean verifier, ``lean``), `lean-suites` as `lean-suites:SUITE` for those."""
    out = []
    for p in all_parts:
        if p == "pytest":
            out += [f"pytest:{s}" for s in suites if not (lean and s in after_lean)]
        elif p == "lean-suites":
            out += [f"lean-suites:{s}" for s in suites if lean and s in after_lean]
        else:
            out.append(p)
    return out


def priced(names: list[str], suites: dict[str, float], audits: dict[str, float], steps: dict[str, float]) -> dict[str, float]:
    """The measured seconds of each part of ``names`` that has some: a suite's (``suites``, by name), a package's audit
    (``audits``), a part a recorded step ran alone (``steps``, by part), else its whole step's (``steps``, by job), spread
    evenly over a split job's parts (circuit-check's shards)."""
    split: dict[str, int] = {}
    for p in names:
        if ":" in p:
            split[p.partition(":")[0]] = split.get(p.partition(":")[0], 0) + 1
    out = {}
    for p in names:
        job, _, x = p.partition(":")
        if job in SUITE_JOBS and x:
            got = suites.get(x)
        elif job == "lean-audit" and x:
            got = audits.get(x)
        else:
            got = steps.get(p, steps[job] / split.get(job, 1) if x and job in steps else None)
        if got is not None:
            out[p] = float(got)
    return out


def tasks(names: list[str], measured: dict[str, float]) -> list[Task]:
    """Each part of ``names`` at its measured seconds, else its kind's median, else every measured part's, else `DEFAULT_S`."""
    by: dict[str, list[float]] = {}
    for p, s in measured.items():
        by.setdefault(kind(p), []).append(s)
    every = list(measured.values())
    out = []
    for p in names:
        if p in measured:
            out.append(Task(p, float(measured[p]), "measured"))
        elif by.get(kind(p)):
            out.append(Task(p, float(statistics.median(by[kind(p)])), "median"))
        elif every:
            out.append(Task(p, float(statistics.median(every)), "median-all"))
        else:
            out.append(Task(p, DEFAULT_S, "default"))
    return out


def builds(names: list[str], needs: dict[str, list[str]]) -> dict[str, list[str]]:
    """Each part's builds: the parts among ``names`` that it or its job needs run beside it (``needs``, by part or by job
    name: a job's `needs` and `after`, a package's audit its provers'), transitively, deepest first."""
    present = set(names)

    def chain(p: str, seen: set[str], out: list[str]) -> list[str]:
        for n in [*needs.get(p, []), *(needs.get(p.partition(":")[0], []) if ":" in p else [])]:
            if n not in seen:
                seen.add(n)
                chain(n, seen, out)
                out.append(n)
        return out

    return {p: [b for b in chain(p, {p}, []) if b in present] for p in names}


def units(ts: list[Task], needs: dict[str, list[str]]) -> list[tuple[list[str], list[str]]]:
    """The tasks as the units LPT places: each part with the builds it needs that cost more than `CHEAP_S`, and their readers,
    held on one shard (in ``ts``'s order); beside each unit, the cheap builds its parts need from outside it."""
    cost = {t.part: t.seconds for t in ts}
    need = builds(list(cost), needs)
    root = {p: p for p in cost}

    def find(p: str) -> str:
        while root[p] != p:
            p = root[p]
        return p

    for p, bs in need.items():
        for b in bs:
            if cost[b] > CHEAP_S:
                root[find(p)] = find(b)
    held: dict[str, list[str]] = {}
    for p in cost:
        held.setdefault(find(p), []).append(p)
    return [(u, [b for b in dict.fromkeys(b for p in u for b in need[p]) if b not in u]) for u in held.values()]


def lpt(ts: list[Task], needs: dict[str, list[str]], n: int) -> list[dict]:
    """``ts`` packed onto ``n`` shards: its `units` longest first (own cost and builds), each onto the shard where it ends
    soonest with the builds that shard doesn't run yet (the lowest index on a tie).  Each shard's parts (in ``ts``'s order),
    the builds it runs beside them (`builds` not among its parts), and its predicted seconds."""
    cost = {t.part: t.seconds for t in ts}
    order = {t.part: i for i, t in enumerate(ts)}
    load, runs, held = [0.0] * n, [set() for _ in range(n)], [[] for _ in range(n)]
    for u, bs in sorted(units(ts, needs), key=lambda x: (-sum(cost[p] for p in [*x[0], *x[1]]), order[x[0][0]])):
        def extra(k: int) -> float:
            return sum(cost[b] for b in [*bs, *u] if b not in runs[k])

        k = min(range(n), key=lambda i: (load[i] + extra(i), i))
        load[k] += extra(k)
        runs[k] |= {*bs, *u}
        held[k] += u
    return [{"parts": sorted(held[k], key=order.__getitem__),
             "builds": sorted(runs[k] - set(held[k]), key=order.__getitem__), "seconds": round(load[k], 1)} for k in range(n)]


def floor(ts: list[Task], needs: dict[str, list[str]]) -> tuple[float, list[str]]:
    """The longest unit with its builds (`units`): (its seconds, the builds and its parts)."""
    cost = {t.part: t.seconds for t in ts}
    chain = max(([*bs, *u] for u, bs in units(ts, needs)), key=lambda c: (sum(cost[p] for p in c), c))
    return sum(cost[p] for p in chain), chain


def choose(ts: list[Task], needs: dict[str, list[str]], max_shards: int) -> int:
    """The smallest N up to ``max_shards`` whose packed makespan is within `SLACK` of max(floor, total / max_shards), else the
    smallest N with the least makespan."""
    best = max(floor(ts, needs)[0], sum(t.seconds for t in ts) / max_shards)
    spans = {n: max(s["seconds"] for s in lpt(ts, needs, n)) for n in range(1, max_shards + 1)}
    return next((n for n, s in spans.items() if s <= best * (1 + SLACK)), min(spans, key=lambda n: (spans[n], n)))


def plan(ts: list[Task], needs: dict[str, list[str]], max_shards: int = MAX_SHARDS, n: int | None = None) -> dict:
    """The auto plan (`KIND`) of ``ts``: N (``n``, else `choose`'s), each shard's parts, builds and predicted seconds, the
    floor and its chain, the predicted wall time (the makespan), the total, and a digest of the shards' parts and builds,
    which every shard of one plan names."""
    if not ts:
        raise ValueError("an auto plan of no parts")
    if max_shards < 1 or (n is not None and n < 1):
        raise ValueError(f"max_shards {max_shards}, n {n}: at least 1")
    n = n or choose(ts, needs, max_shards)
    shards = lpt(ts, needs, n)
    low, chain = floor(ts, needs)
    return {"kind": KIND, "n": n, "max_shards": max_shards, "floor_s": round(low, 1), "floor": chain,
            "total_s": round(sum(t.seconds for t in ts), 1), "predicted_s": max(s["seconds"] for s in shards),
            "all": [t.part for t in ts], "shards": shards,
            "tasks": {t.part: [round(t.seconds, 1), t.source] for t in ts}, "digest": digest(shards)}


def digest(shards: list[dict]) -> str:
    return hashlib.sha256(json.dumps([[s["parts"], s["builds"]] for s in shards], separators=(",", ":")).encode()).hexdigest()


def check(doc: dict, n: int, want: list[str]) -> str | None:
    """Why ``doc`` is not an auto plan of N = ``n`` shards whose parts are exactly ``want`` (this tree's), each in one shard,
    or None."""
    if not isinstance(doc, dict) or doc.get("kind") != KIND:
        return f"not a {KIND}"
    shards = doc.get("shards")
    if not (isinstance(shards, list) and all(isinstance(s, dict) and isinstance(s.get("parts"), list)
                                              and isinstance(s.get("builds"), list) for s in shards)):
        return "it lists no shards' parts and builds"
    if doc.get("n") != n or len(shards) != n:
        return f"it has {len(shards)} shards, not {n}"
    flat = [p for s in shards for p in s["parts"]]
    if len(set(flat)) != len(flat):
        return "it runs a part twice"
    if sorted(flat) != sorted(want) or sorted(doc.get("all") or []) != sorted(want):
        missing, extra = sorted(set(want) - set(flat)), sorted(set(flat) - set(want))
        return ("its parts aren't this tree's: " + (f"missing {', '.join(missing[:5])}" if missing else "")
                + ("; " if missing and extra else "") + (f"unknown {', '.join(extra[:5])}" if extra else ""))
    if doc.get("digest") != digest(shards):
        return "its digest isn't its shards'"
    return None


def summary(doc: dict) -> dict:
    """What a shard's result records of its plan: N, the floor, the predicted wall time, each shard's seconds, the digest."""
    return {k: doc[k] for k in ("n", "max_shards", "floor_s", "floor", "total_s", "predicted_s", "digest")} | {
        "seconds": [s["seconds"] for s in doc["shards"]]}


def lines(doc: dict) -> list[str]:
    """The plan as text: N, the floor and the predicted wall time, then each shard's seconds, parts and builds."""
    def m(s: float) -> str:
        return f"{s / 60:.1f} min"

    priced_by: dict[str, int] = {}
    for _, src in doc["tasks"].values():
        priced_by[src] = priced_by.get(src, 0) + 1
    out = [f"shard plan: N = {doc['n']} (max {doc['max_shards']}), floor {m(doc['floor_s'])} ({' + '.join(doc['floor'])}), "
           f"predicted wall {m(doc['predicted_s'])}, total {m(doc['total_s'])} over {len(doc['all'])} parts ("
           + ", ".join(f"{v} {k}" for k, v in sorted(priced_by.items())) + f" costs); digest {doc['digest'][:12]}"]
    for k, s in enumerate(doc["shards"], 1):
        jobs: dict[str, list[str]] = {}
        for p in s["parts"]:
            jobs.setdefault(p.partition(":")[0], []).append(p)
        out.append(f"shard plan: {k}/{doc['n']}: {m(s['seconds'])}: "
                   + ", ".join(ps[0] if len(ps) == 1 else f"{j} x{len(ps)}" for j, ps in jobs.items())
                   + (f"; builds {', '.join(s['builds'])} beside them" if s["builds"] else ""))
    return out
