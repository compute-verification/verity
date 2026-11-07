"""A drained edge in a vLLM V1 engine core, and a tensor-parallel engine's per-rank boundary state taken there.

Under async scheduling the engine core keeps up to two steps in flight (``EngineCore.batch_queue``). Their sampled
tokens are still on the GPU and the scheduler holds placeholders for them. `Edges` replaces the engine core's
``step_fn``. Once the scheduler has scheduled the step an edge names (``scheduler.current_step`` reaching it), it
schedules nothing more: each call only pops the oldest in-flight step and processes its output, as vLLM's own
``step_with_batch_queue`` does when its queue is full. When the queue is empty, nothing is in flight:
``processed_step_seq`` has caught up with ``sched_step_seq``, ``deferred_frees`` is empty and no request has
placeholders. `Edges.snapshot` then takes the boundary state, and scheduling resumes with the step after the edge's.
Under synchronous scheduling every step ends drained, and the snapshot follows the edge's step at once.

The snapshot, through ``collective_rpc`` on every rank (`tp_state`), runs in five parts:
- **The block set:** `live_blocks`: every block a request holds, every free block still in the prefix cache, and the
  null block.
- **Digests:** each rank's per-layer digests of those blocks, its RNG state and its rank record.
- **The cross-rank check (`agree`):** GLM-5.3's MLA latent and DSA indexer keys are computed by replicated projections,
  so all ranks should hold the same bytes. The check finds the first layer and block where a rank differs from rank 0.
- **The dump:** rank 0's blocks if the ranks agree, every rank's if they don't (only at the edges in ``dump``).
- **Savers:** each one in ``savers`` (the scheduler's state, the runner's rows) gets the engine core and the edge's
  directory.

``edge.json`` records the step, the block ids, the pool's counts, every rank's record and the check.

`install` puts `Edges` on an engine core in this process (the restored engine's, in-process). `register`, a vLLM general
plugin, puts it on every engine core a ``vllm serve`` process builds, configured by ``VERITY_TPEDGE`` (JSON: ``out``;
``steps`` or ``at_s``; ``dump``, how many of the first edges write their blocks, 1 by default; ``savers``, each a
``module:function`` the snapshot calls with the engine core and the edge's directory).
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import tp_state

ENV = "VERITY_TPEDGE"
_PLUGIN: list[Any] = []


def live_blocks(pool: Any) -> list[int]:
    """The blocks a restore must copy, ascending: the null block, every block a request holds (``ref_cnt`` > 0) and
    every free block still in the prefix cache (a ``block_hash``), which a request after the edge can hit."""
    keep = {pool.null_block.block_id}
    keep.update(b.block_id for b in pool.blocks if b.ref_cnt > 0 or b.block_hash is not None)
    return sorted(keep)


def pool_counts(pool: Any) -> dict[str, int]:
    """The pool's blocks: all of them, those requests hold, the free ones still cached, and the free ones."""
    held = sum(1 for b in pool.blocks if b.ref_cnt > 0 and not b.is_null)
    cached = sum(1 for b in pool.blocks if b.ref_cnt == 0 and b.block_hash is not None)
    return {"blocks": len(pool.blocks), "held": held, "cached_free": cached, "free": pool.get_num_free_blocks()}


def agree(recs: Sequence[Mapping[str, Any]], out: Path | None = None, ids: Sequence[int] = ()) -> dict[str, Any]:
    """Whether every rank's `tp_state.digest` matches rank 0's: its layers, its KV digests and its RNG's. With ``out``
    (the ranks' digest files), say where each differing rank first differs from rank 0."""
    recs = sorted(recs, key=lambda r: r["rank"]["rank"])
    base = recs[0]
    res: dict[str, Any] = {"ranks": [r["rank"]["rank"] for r in recs],
                           "layers_equal": all(r["layers"] == base["layers"] for r in recs),
                           "kv_equal": all(r["layer_digests"] == base["layer_digests"] for r in recs),
                           "rng_equal": all(r["rng_digest"] == base["rng_digest"] for r in recs)}
    if not res["layers_equal"]:
        res["layers_differ"] = [r["rank"]["rank"] for r in recs if r["layers"] != base["layers"]]
        return res
    if not res["kv_equal"]:
        firsts = {}
        for r in recs[1:]:
            if r["layer_digests"] == base["layer_digests"]:
                continue
            rank = r["rank"]["rank"]
            li = next(i for i, (x, y) in enumerate(zip(r["layer_digests"], base["layer_digests"])) if x != y)
            firsts[rank] = {"layer": li, "name": base["layers"][li]["name"]}
            if out is not None:
                a = (out / f"digests.rank{base['rank']['rank']}.bin").read_bytes()
                b = (out / f"digests.rank{rank}.bin").read_bytes()
                firsts[rank] = tp_state.first_difference(a, b, base["layers"], ids)
        res["kv_first_difference"] = firsts
    if not res["rng_equal"]:
        res["rng_differ"] = [r["rank"]["rank"] for r in recs if r["rng_digest"] != base["rng_digest"]]
    return res


def rank_map(recs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each rank's place, by rank: the host, the device's UUID and PCI address, ``CUDA_VISIBLE_DEVICES`` and the
    all-reduce backends; a restore on another map is refused unless asked (`check_rank_map`)."""
    keys = ("rank", "tp_rank", "local_rank", "host", "visible", "device", "allreduce")
    return [{k: r["rank"].get(k) for k in keys} for r in sorted(recs, key=lambda r: r["rank"]["rank"])]


def check_rank_map(served: Sequence[Mapping[str, Any]], restored: Sequence[Mapping[str, Any]]) -> list[str]:
    """How the restored engine's rank map differs from the served one's: one line per differing rank and field."""
    diffs = []
    if len(served) != len(restored):
        return [f"{len(served)} served ranks, {len(restored)} restored"]
    for s, r in zip(served, restored):
        for k in ("host", "device", "allreduce", "visible"):
            if s.get(k) != r.get(k):
                diffs.append(f"rank {s.get('rank')}: {k} {s.get(k)!r} served, {r.get(k)!r} restored")
    return diffs


def _rpc(core: Any) -> Callable[..., list[Any]]:
    return core.model_executor.collective_rpc


def trace_step(sched: Any) -> int | None:
    """The scheduler's step trace's count of steps scheduled with tokens (`StepTrace.scheduled`, the index its next
    such step gets), from serve_tax's installs or a replay's trace, or None without one."""
    for obj in (*getattr(sched, "_verity_servetax", ()), getattr(sched, "_verity_trace", None)):
        if obj is not None and hasattr(obj, "scheduled"):
            return int(obj.scheduled)
    return None


def in_flight(sched: Any) -> dict[str, int]:
    """The running requests by phase at the edge: still prefilling (a chunked prefill part done) or decoding."""
    pre = sum(1 for r in sched.running if r.num_computed_tokens < r.num_prompt_tokens)
    return {"prefilling": pre, "decoding": len(sched.running) - pre}


def scheduler_step(core: Any) -> int:
    return core.scheduler.current_step


class Edges:
    """The drained edges of one engine core, each snapshotted under ``out/edge-<step>``: at scheduler steps ``steps``
    (a replay, which knows the served edges' steps), or at the first step ``at_s`` seconds after the engine's first
    scheduled step (serving, which doesn't). Steps count as ``counter`` counts them: the scheduler's ``current_step``,
    or a replay's step trace (`trace_step`). The first ``dump`` edges taken also write their blocks."""

    def __init__(self, out: str | Path, steps: Iterable[int] = (), at_s: Iterable[float] = (), dump: int = 1,
                 savers: Sequence[Callable[[Any, Path], Any]] = (), clock: Callable[[], float] = time.monotonic,
                 counter: Callable[[Any], int] = scheduler_step):
        self.out = Path(out)
        self.pending: list[int] = sorted(set(int(s) for s in steps))
        self.times: list[float] = sorted(float(t) for t in at_s)
        if not self.pending and not self.times:
            raise ValueError("tp_edge.Edges: no edge (give steps or at_s)")
        self.dump = int(dump)
        self.savers = list(savers)
        self.clock = clock
        self.counter = counter
        self.t_first: float | None = None
        self.draining: int | None = None
        self.taken: list[dict[str, Any]] = []

    def due(self, current_step: int) -> int | None:
        """The edge the scheduler has reached (its step scheduled), if any is pending: its step."""
        if current_step >= 1 and self.t_first is None:
            self.t_first = self.clock()
        if self.pending and current_step >= self.pending[0]:
            self.pending.pop(0)
            return current_step
        if self.times and self.t_first is not None and self.clock() - self.t_first >= self.times[0]:
            self.times.pop(0)
            return current_step
        return None

    def wrap(self, core: Any, step_fn: Callable[[], Any]) -> Callable[[], Any]:
        """``step_fn`` with the edges: after the call that schedules an edge's step, scheduling stops until the batch
        queue is empty; the call that empties it snapshots."""
        def step() -> Any:
            if self.draining is not None:
                sched = core.scheduler
                sched.has_requests = lambda: False
                try:
                    got = step_fn()
                finally:
                    del sched.has_requests
                if not core.batch_queue:
                    edge, self.draining = self.draining, None
                    self.snapshot(core, edge)
                return got
            got = step_fn()
            edge = self.due(self.counter(core))
            if edge is not None:
                if core.batch_queue:
                    self.draining = edge
                else:
                    self.snapshot(core, edge)
            return got
        return step

    def snapshot(self, core: Any, edge: int) -> dict[str, Any]:
        """The boundary state at a drained edge (module docstring), into ``out/edge-<edge>``; returns ``edge.json``."""
        sched = core.scheduler
        pool = sched.kv_cache_manager.block_pool
        if getattr(sched, "deferred_frees", None) or sched.sched_step_seq != sched.processed_step_seq:
            raise RuntimeError(f"tp_edge: edge {edge} is not drained: sched_step_seq {sched.sched_step_seq}, "
                               f"processed_step_seq {sched.processed_step_seq}, deferred frees "
                               f"{len(getattr(sched, 'deferred_frees', ()))}")
        d = self.out / f"edge-{edge:06d}"
        d.mkdir(parents=True, exist_ok=True)
        ids = live_blocks(pool)
        t0 = self.clock()
        recs = _rpc(core)(tp_state.digest, args=(ids, str(d)))
        t1 = self.clock()
        check = agree(recs, d, ids)
        wrote: list[Any] = []
        if len(self.taken) < self.dump:
            ranks = [0] if check["kv_equal"] else check["ranks"]
            wrote = [w for w in _rpc(core)(tp_state.dump, args=(str(d), ids, ranks)) if w]
        t2 = self.clock()
        saved = [save(core, d) for save in self.savers]
        t3 = self.clock()
        rec = {"edge": edge, "current_step": sched.current_step, "trace_step": trace_step(sched),
               "sched_step_seq": sched.sched_step_seq,
               "running": len(sched.running), "in_flight": in_flight(sched), "waiting": len(sched.waiting),
               "pool": pool_counts(pool), "ids": ids,
               "check": check, "rank_map": rank_map(recs), "layers": recs[0]["layers"],
               "ranks": [{k: r[k] for k in ("layer_digests", "rng_digest", "rng")} | {"rank": r["rank"]["rank"]}
                         for r in sorted(recs, key=lambda r: r["rank"]["rank"])],
               "dump": wrote, "saved": saved, "seconds": {"digest": round(t1 - t0, 3), "dump": round(t2 - t1, 3),
                                                          "savers": round(t3 - t2, 3)}}
        (d / "edge.json").write_text(json.dumps(rec, default=repr))
        self.taken.append({k: rec[k] for k in ("edge", "running", "waiting", "pool", "check", "seconds")})
        return rec


def install(core: Any, edges: Edges) -> Edges:
    """Put ``edges`` on an engine core: its ``step_fn`` (vLLM picks ``step`` or ``step_with_batch_queue`` at
    construction) gains the drained edges."""
    core.step_fn = edges.wrap(core, core.step_fn)
    core._verity_edges = edges
    return edges


def config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """``VERITY_TPEDGE``'s JSON, or None when unset."""
    raw = (os.environ if environ is None else environ).get(ENV)
    if not raw:
        return None
    cfg = json.loads(raw)
    if "out" not in cfg or not (cfg.get("steps") or cfg.get("at_s")):
        raise ValueError(f"{ENV} needs out, and steps or at_s; got {sorted(cfg)}")
    return cfg


def resolve(name: str) -> Callable[..., Any]:
    """A ``module:function`` name's function (a saver ``VERITY_TPEDGE`` names, run in the engine core)."""
    import importlib

    mod, _, fn = name.partition(":")
    if not fn:
        raise ValueError(f"{ENV}: a saver is module:function, not {name!r}")
    return getattr(importlib.import_module(mod), fn)


def register() -> None:
    """vLLM general plugin: every ``EngineCore`` this process builds gets `Edges` from ``VERITY_TPEDGE``. Idempotent."""
    cfg = config()
    if cfg is None or _PLUGIN:
        return
    from vllm.v1.engine.core import EngineCore

    orig = EngineCore.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        orig(self, *args, **kwargs)
        install(self, Edges(cfg["out"], cfg.get("steps", ()), cfg.get("at_s", ()), cfg.get("dump", 1),
                            [resolve(n) for n in cfg.get("savers", ())]))

    EngineCore.__init__ = __init__
    _PLUGIN.append(orig)
