"""verity_vllm/check/replay/tier_a_pool.py — tier A's strata, evaluated in the parent or on `jobs` forked workers.

`replay.tier_a` hands its specializations here, with its `Replayer`.  The parent consumes the challenge stream exactly as
a sequential loop does: `plan_child` once per oversize specialization, `_distinct_step_order` once per stratum, and a
child unit's members (one `randrange` per level of its path) when its candidate is dispatched.  A stratum dispatches the
next candidate of its order while its counted (pass / fail / error) plus pending evaluations are fewer than it needs, so
each skip is refilled by one more candidate, and no candidate past the point where a sequential loop stops is ever
evaluated.  An instance-granular stratum draws nothing after its order, so with workers the parent goes on drawing the
next strata while it runs; a child-granular stratum settles first, since its refills draw members.  The report is
assembled in specialization, regime and candidate order: at one seed it is the same for every `jobs`, apart from `jobs`,
`seconds` and `max_rss_gb`.  `jobs=1` evaluates in the parent, at dispatch, in the sequential loop's order.

Workers (`jobs` > 1) are `fork.fork_pool` children of the parent's Replayer, forked after `twins.preload()` (the MUFU
tables the gates read, loaded once and shared) and `gc.freeze()` (a child's first collection would otherwise copy the
parent's whole heap, `driver._parallel_replay`).  A worker drops its value caches after each task, so it holds one task's
working set.  A worker that dies mid-task raises `fork.WorkerDied`: tier A then writes no report, so the stage fails and
its gate is not evaluated, rather than a partial report or a failing tier.
"""
from __future__ import annotations

import gc
import queue
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from verity.ml.circuits.layout import FunctionCall
from verity_vllm.check.replay import fork as FK

Outcome = tuple[str, Any]          # `Replayer.outcome`: ("ok", entry) | ("skipped", reason) | ("error", "<type>: <message>")
_REP: Any = None                   # the `replay.Replayer` the forked workers evaluate on, set in the parent before the fork


def _regime(m: dict) -> str:
    """The execution regime of an instance: `prefill` (step 0) or `decode`."""
    return "prefill" if int(m["step"]) == 0 else "decode"


def _distinct_step_order(metas: list[dict], rng: random.Random) -> list[dict]:
    """A seeded order that visits distinct steps first (round-robin over steps, each step's instances shuffled)."""
    by_step: dict[int, list[dict]] = defaultdict(list)
    for m in metas:
        by_step[m["step"]].append(m)
    steps = sorted(by_step)
    rng.shuffle(steps)
    for st in steps:
        rng.shuffle(by_step[st])
    out: list[dict] = []
    i = 0
    while len(out) < len(metas):
        for st in steps:
            if i < len(by_step[st]):
                out.append(by_step[st][i])
        i += 1
    return out


def _reason_class(reason: str) -> str:
    """Short class of a skip reason for the per-spec summary."""
    if "no snapshot" in reason or "has no snapshot" in reason:
        return "operand/output not snapshotted"
    if "weight source" in reason:
        return "no weight source"
    return reason[:80]


@dataclass
class _Stratum:
    regime: str
    order: list[dict]
    need: int
    path: list[tuple[int, int]] | None
    out: dict[int, Outcome] = field(default_factory=dict)    # position in `order` -> outcome
    sent: int = 0
    counted: int = 0
    pending: int = 0

    def settled(self) -> bool:
        return self.pending == 0 and (self.counted >= self.need or self.sent == len(self.order))


def _evaluate(rep: Any, task: tuple) -> Outcome:
    node, path, members = task
    return rep.outcome(rep.by_node[node], path, members)


def _work(task: tuple) -> Outcome:
    try:
        return _evaluate(_REP, task)
    finally:
        _drop_caches(_REP)


def _drop_caches(rep: Any) -> None:
    """A worker's caches are private copies that only its next tasks could read; dropping them bounds it to one task."""
    rep._field_cache.clear()
    rep._out_cache.clear()
    rep._ret_index.clear()
    if isinstance(cache := getattr(rep.weights, "_cache", None), dict):     # `WeightSource`'s checkpoint fields
        cache.clear()
    for s in rep.store.entries:
        if s._words is not None or s._index is not None:
            s._words = s._index = None


class _Inline:
    forked = False

    def __init__(self, rep: Any) -> None:
        self.rep, self.done = rep, []

    def submit(self, key: tuple, task: tuple) -> None:
        self.done.append((key, _evaluate(self.rep, task)))

    def next(self) -> tuple[tuple, Outcome]:
        return self.done.pop(0)


class _Forked:
    """Tasks on the pool's workers.  `next` returns results as they complete, and raises `WorkerDied` once a worker has
    exited mid-task, since its task would never complete."""

    forked = True

    def __init__(self, pool: Any, jobs: int) -> None:
        self.pool, self.jobs = pool, jobs
        self.pids, self.oom0 = {p.pid for p in pool._pool}, FK.oom_kills()
        self.q: queue.SimpleQueue = queue.SimpleQueue()

    def submit(self, key: tuple, task: tuple) -> None:
        self.pool.apply_async(_work, (task,), callback=lambda out: self.q.put((key, out)), error_callback=lambda e: self.q.put((key, e)))

    def next(self) -> tuple[tuple, Outcome]:
        while True:
            if (why := FK.dead_worker(self.pool, self.pids, self.oom0, self.jobs, "--jobs")) is not None:
                print(f"[replay] tier a: {why}", file=sys.stderr, flush=True)
                raise FK.WorkerDied(why)
            try:
                key, out = self.q.get(timeout=FK.POOL_POLL_S)
            except queue.Empty:
                continue
            if isinstance(out, BaseException):
                raise out
            return key, out


def evaluate_strata(rep: Any, report: dict, by_spec: dict[str, list[dict]], snap_steps: list[int], *, k: int,
                    max_arg_leaves: int, rng: random.Random, jobs: int) -> None:
    """Tier A's evaluations on `rep` (a `replay.Replayer`) into `report`: `strata`, `coverage`, `instances` and
    `uncovered_specs` (the module docstring)."""
    if jobs <= 1:
        _run(rep, report, by_spec, snap_steps, k, max_arg_leaves, rng, _Inline(rep))
        return
    global _REP
    from verity_vllm.program.kernels.twins import preload
    preload()
    _REP = rep
    collected = gc.collect()
    gc.freeze()
    print(f"[replay] tier a: {jobs} forked workers, {gc.get_freeze_count()} objects frozen ({collected} collected) before the fork",
          file=sys.stderr, flush=True)
    try:
        with FK.fork_pool(jobs) as pool:
            _run(rep, report, by_spec, snap_steps, k, max_arg_leaves, rng, _Forked(pool, jobs))
    finally:
        gc.unfreeze()
        _REP = None


def _run(rep: Any, report: dict, by_spec: dict[str, list[dict]], snap_steps: list[int], k: int, max_arg_leaves: int,
         rng: random.Random, ex: _Inline | _Forked) -> None:
    report["strata"] = 0
    specs = []
    for spec in sorted(by_spec):
        cov, uncovered, path, regimes = _plan(rep, report, spec, by_spec[spec], snap_steps, k, max_arg_leaves, rng)
        strata = []
        for rg in sorted(regimes):
            st = _Stratum(rg, _distinct_step_order(regimes[rg], rng), min(k, len(regimes[rg])), path)
            strata.append(st)
            _top_up(st, ex, rng)
            if path is not None or not ex.forked:
                _drain(ex, rng, st.settled)
        specs.append((spec, cov, uncovered, strata))
    every = [st for *_, strata in specs for st in strata]
    _drain(ex, rng, lambda: all(st.settled() for st in every))
    for spec, cov, uncovered, strata in specs:
        _assemble(report, spec, cov, uncovered, strata)


def _plan(rep: Any, report: dict, spec: str, metas: list[dict], snap_steps: list[int], k: int, max_arg_leaves: int,
          rng: random.Random) -> tuple[dict, list[dict], list[tuple[int, int]] | None, dict[str, list[dict]]]:
    """A specialization's coverage record, child path and strata, before any of its evaluations.  No strata, and an
    `uncovered_specs` entry, when none of its instances is evaluable."""
    cov = {"instances": len(metas), "evaluable": 0, "evaluated": 0, "pass": 0, "fail": 0, "skipped": 0, "error": 0,
           "granularity": None, "regimes": {}}
    report["coverage"][spec] = cov
    cands = [m for m in metas if m["step"] in snap_steps]
    # Strata (D82): the specialization x the execution regime it ran in.  `k` per stratum; within a stratum the
    # sample prefers distinct steps (early/late KV lengths) before repeating a step.
    regimes: dict[str, list[dict]] = defaultdict(list)
    for m in cands:
        regimes[_regime(m)].append(m)
    for rg in sorted(regimes):
        cov["regimes"][rg] = {"instances": len(regimes[rg]), "pass": 0, "fail": 0, "error": 0, "skipped": 0}
    report["strata"] += len(regimes)
    if not cands:
        present = sorted({m["step"] for m in metas})
        return cov, [{"spec": spec, "reason": f"no snapshot step contains it (present at steps {present[:6]}"
                                              f"{'...' if len(present) > 6 else ''}; snapshot steps {snap_steps})"}], None, {}
    leaves = rep.arg_leaves(cands[0]["node"])
    path: list[tuple[int, int]] | None = None
    if leaves <= max_arg_leaves:
        cov["granularity"] = "instance"
    else:
        try:
            path = rep.plan_child(cands[0]["node"], max_arg_leaves, rng)
        except Exception as e:  # noqa: BLE001 - reported per spec
            cov["plan_error"] = f"{type(e).__name__}: {e}"
        if path is None:
            return cov, [{"spec": spec, "reason": f"{leaves} argument leaves exceed max_arg_leaves={max_arg_leaves} "
                                                  "and no child unit fits (or its interface is interior)"}], None, {}
        probe = rep.call(cands[0]["node"])
        for j, _ in path:
            probe = FunctionCall(rep.circuit, probe.scope(), j, 0)
        cov["granularity"] = f"child:{probe.id}"
        cov["child_arg_leaves"] = sum(t.leaves for t in probe.arg_types())
    cov["evaluable"] = len(cands)
    cov["required"] = sum(min(k, len(ms)) for ms in regimes.values())
    return cov, [], path, regimes


def _top_up(st: _Stratum, ex: _Inline | _Forked, rng: random.Random) -> None:
    while st.counted + st.pending < st.need and st.sent < len(st.order):
        i, m = st.sent, st.order[st.sent]
        members = [rng.randrange(nm) for _, nm in st.path] if st.path is not None else None
        st.sent += 1
        st.pending += 1
        ex.submit((st, i), (m["node"], st.path, members))


def _drain(ex: _Inline | _Forked, rng: random.Random, done: Callable[[], bool]) -> None:
    while not done():
        (st, i), out = ex.next()
        st.out[i] = out
        st.pending -= 1
        st.counted += out[0] != "skipped"
        _top_up(st, ex, rng)


def _assemble(report: dict, spec: str, cov: dict, uncovered: list[dict], strata: list[_Stratum]) -> None:
    report["uncovered_specs"] += uncovered
    reasons: Counter = Counter()
    for st in strata:
        rc = cov["regimes"][st.regime]
        for i in sorted(st.out):
            (kind, val), m = st.out[i], st.order[i]
            e = val if kind == "ok" else {"node": m["node"], "name": m["name"], "spec": spec, "step": m["step"], "status": kind,
                                          "reason": val, "granularity": cov["granularity"]}
            if kind == "skipped":
                reasons[_reason_class(val)] += 1
            e["regime"] = st.regime
            report["instances"].append(e)
            cov[e["status"]] += 1
            rc[e["status"]] += 1
            if e["status"] != "skipped":
                cov["evaluated"] += 1
    need = cov.get("required", 0)
    if strata and cov["pass"] < need and not (cov["fail"] or cov["error"]):
        why = "; ".join(f"{r} (x{n})" for r, n in reasons.most_common(3)) or "no evaluation completed"
        report["uncovered_specs"].append({"spec": spec, "reason": f"{cov['pass']} of {need} required evaluations passed; skipped: {why}"})
