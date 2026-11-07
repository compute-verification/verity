"""Rebuild a replay block's batch at its first step from the served K/V pages alone, as lifetime replay starts a sampled
request's replay block, replay to a far edge, and say how close the replay came.

    rebuild.py --arm-dir OUT/restore [--from-edge 1] [--to-edge 2] --kv-dir DIR [--checkpoint DIR] [--tail copy] --out DIR

A restore (`restore.py`) puts the served engine's whole state back: the block pool, the scheduler and the model runner's
rows. Lifetime replay has less: a host copy of each K/V page, written when the page was, and the step record. So here a
fresh engine gets each request the served engine held at the edge as a new request, in the served running then waiting
order, with its prompt, its outputs so far, its served id and sampling; and its computed prefix's pages are copied from
the edge's K/V file into blocks the fresh engine allocates, as a KV connector's load does (`allocate_slots` with the
prefix as external tokens, then `cache_blocks`). Block ids, runner slots, the prefix cache and every other piece of
engine state are the fresh engine's own. ``--tail`` sets what a re-added request's last, partial block holds past its
prefix: the served bytes (``copy``), zeros (``zero``), or NaN, with every other slot of the fresh pool NaN too
(``nan``: then nothing outside a computed prefix may be read). The served arm must have kept the K/V file of both edges
(serve_tax's ``--save-kv FROM,TO``).

The fresh engine's prefix cache holds only the re-added requests' pages, so a request the served engine served from a
finished request's cached pages would miss. ``--hits forced`` (the default) takes each request's prefix-cache hit from
the step record instead, as lifetime replay's schedule does: its lookup is cut to the served hit, and a hit page the
fresh cache lacks is loaded from the edge's K/V, the block the served pool cached under its hash (`Hits`).
``--hits engine`` leaves the lookups to the fresh engine.

rebuild.json compares, from edge to edge: the schedule step by step (at the first step the re-added requests are new to
the fresh engine, so they are left out of its ``new`` and ``cache_hit``), every sampled token and its logprob's bits, and
at both edges each request's K/V over its computed tokens, a SHA-512 root per request against the served edge's K/V file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "config"))
sys.path.append(str(ROOT))
from verity_config.settings import SHARED  # noqa: E402 (from the tree's tools/config, put on sys.path above)
import boundary as B  # noqa: E402 (beside this file)
import restore as R  # noqa: E402
import serve_tax as S  # noqa: E402

TAILS = ("copy", "zero", "nan")


def layer_offsets(layout: dict[str, Any]) -> list[int]:
    """Each layer's byte offset in a K/V file `boundary.write_host` wrote (the pool's storages, concatenated in order);
    refused unless every layer lays its blocks out first and contiguous (`boundary.KvPool.rows`)."""
    if not layout.get("block_row_bytes"):
        raise SystemExit("the pool's layers are not block-major: no block rows to copy")
    starts = [0]
    for n in layout["storages"][:-1]:
        starts.append(starts[-1] + n)
    return [starts[v["storage"]] + v["offset"] for v in layout["views"]]


def slot_major(shape: list[int], block_size: int) -> bool:
    """A layer's block is its slots one after another: past the block axis and any axes of size 1, the next is the
    block's slots (MLA's ``[blocks, 1, block_size, 576]``)."""
    rest = list(shape[1:])
    while rest and rest[0] == 1:
        rest.pop(0)
    return bool(rest) and rest[0] == block_size


def kv_file(path: Path, layout: dict[str, Any]) -> list[Any]:
    """The K/V file's layers as read-only (blocks, row bytes) int8 maps."""
    import numpy as np

    n, row = layout["num_blocks"], layout["block_row_bytes"]
    return [np.memmap(path, dtype=np.int8, mode="r", offset=off, shape=(n, row)) for off in layer_offsets(layout)]


def roots(layers: list[Any], spans: dict[str, tuple[list[int], int]], block_size: int) -> dict[str, str]:
    """Each request's K/V root: SHA-512 over every layer's bytes of its first ``n`` tokens, layer by layer, from its block
    ids; ``layers`` as (blocks, row bytes) arrays, numpy maps or device tensors."""
    import numpy as np

    out = {}
    if not spans:
        return out
    row = int(layers[0].shape[1])
    slot = row // block_size
    order = sorted(spans)
    ids = [b for r in order for b in spans[r][0][:-(-spans[r][1] // block_size)]]
    got = []
    for x in layers:
        if isinstance(x, np.ndarray):
            got.append(np.asarray(x[ids]))
        else:
            import torch

            got.append(x.index_select(0, torch.tensor(ids, device=x.device)).cpu().numpy())
    at = 0
    for r in order:
        blocks, n = spans[r]
        k = -(-n // block_size)
        h = hashlib.sha512()
        for g in got:
            h.update(g[at:at + k].reshape(-1)[: n * slot].tobytes())
        out[r] = h.hexdigest()
        at += k
    return out


def roots_diff(served: dict[str, str], rebuilt: dict[str, str], limit: int = 20) -> dict[str, Any]:
    both = sorted(set(served) & set(rebuilt))
    bad = [r for r in both if served[r] != rebuilt[r]]
    total = hashlib.sha512("".join(f"{r}:{served[r]};" for r in sorted(served)).encode()).hexdigest()
    mine = hashlib.sha512("".join(f"{r}:{rebuilt[r]};" for r in sorted(rebuilt)).encode()).hexdigest()
    return {"requests_served": len(served), "requests_rebuilt": len(rebuilt), "compared": len(both), "differing": len(bad),
            "only_served": sorted(set(served) - set(rebuilt))[:limit], "only_rebuilt": sorted(set(rebuilt) - set(served))[:limit],
            "first": bad[:limit], "root_served": total, "root_rebuilt": mine, "roots_equal": total == mine}


def served_spans(st: dict[str, Any]) -> dict[str, tuple[list[int], int]]:
    """Each request a served edge's state holds with a computed prefix: its block ids and computed tokens."""
    reqs = B.unpickle_requests(st["requests"], {"block_hasher": None})
    blocks = st["managers"][0]["req_to_blocks"]
    return {r.request_id: ([b.key for b in blocks[r.request_id]], r.num_computed_tokens)
            for r in reqs if r.num_computed_tokens > 0 and r.request_id in blocks}


def live_spans(sched: Any, kvm: Any) -> dict[str, tuple[list[int], int]]:
    return {rid: (kvm.get_block_ids(rid)[0], r.num_computed_tokens) for rid, r in sched.requests.items() if r.num_computed_tokens > 0}


def recorded_hits(steps: dict[int, dict]) -> dict[str, int]:
    """Each request's prefix-cache hit, in tokens, at the step the served engine first scheduled it (0 for none)."""
    out: dict[str, int] = {}
    for s in sorted(steps):
        for rid in steps[s].get("new") or ():
            out.setdefault(rid, (steps[s].get("cache_hit") or {}).get(rid, 0))
    return out


class Hits:
    """The served engine's prefix-cache hits, forced on the fresh engine as lifetime replay's schedule takes them from
    the step record: a request's lookup (`lookup`, around `get_computed_blocks`) is cut to the hit the served engine took,
    and before a request is scheduled, each page of its served hit the fresh cache lacks is loaded (`seed`) from the
    edge's K/V, the block the served pool cached under that hash, into a block held for the replay (never evicted)."""

    def __init__(self, kvm: Any, rows: list[Any], layers: list[Any], served_pool: dict[str, Any], want: dict[str, int],
                 block_size: int) -> None:
        self.kvm, self.pool, self.rows, self.layers = kvm, kvm.block_pool, rows, layers
        self.served, self.want, self.bs = served_pool, want, block_size
        self.held: list[Any] = []
        self.cut, self.short, self.missing = 0, [], 0

    def seed(self, q: Any) -> None:
        import numpy as np
        import torch
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        for i in range(self.want.get(q.request_id, 0) // self.bs):
            key = make_block_hash_with_group_id(q.block_hashes[i], 0)
            if self.pool.cached_block_hash_to_block.get_one_block(key) is not None:
                continue
            src = self.served["cache"].get(key)
            if src is None:
                self.missing += 1
                continue
            src = src if isinstance(src, int) else next(iter(src.values()))
            (blk,) = self.pool.get_new_blocks(1)
            for f, dst in zip(self.layers, self.rows, strict=True):
                dst[blk.block_id].copy_(torch.from_numpy(np.asarray(f[src])).to(dst.device))
            self.pool._insert_block_hash(key, blk, self.served["hash_tokens"][src])
            self.held.append(blk)

    def lookup(self, orig: Any) -> Any:
        def wrapped(request: Any, *a: Any, **kw: Any) -> Any:
            blocks, n, *rest = orig(request, *a, **kw)
            want = self.want.get(request.request_id)
            if want is not None and n > want:
                blocks = self.kvm.create_kv_cache_blocks(tuple(g[: want // self.bs] for g in blocks.blocks))
                n, self.cut = want, self.cut + 1
            elif want is not None and n < want:
                self.short.append([request.request_id, n, want])
            return (blocks, n, *rest)
        return wrapped

    def report(self) -> dict[str, Any]:
        return {"requests_with_hits": sum(1 for v in self.want.values() if v), "pages_loaded": len(self.held),
                "pages_not_cached_at_edge": self.missing, "lookups_cut": self.cut, "lookups_short": len(self.short),
                "short_first": self.short[:20]}


def readd(eng: Any, core: Any, st: dict[str, Any], arrived: dict[str, dict], layers: list[Any], block_size: int,
          tail: str, fresh: Any = None) -> dict[str, Any]:
    """Every request ``st`` (the served edge's state) holds, added to the fresh engine as `rebuild`'s docstring says;
    what it loaded."""
    import numpy as np
    import torch

    sched, kvm, pool, runner = B.parts(core)
    if getattr(sched, "needs_kv_cache_zeroing", False):
        raise SystemExit("this pool zeroes newly allocated blocks, which would clear the loaded pages")
    if len(st["managers"]) != 1:
        raise SystemExit(f"{len(st['managers'])} KV cache groups: the rebuild copies one")
    if not all(slot_major(v["shape"], block_size) for v in st["kv_layout"]["views"]):
        raise SystemExit("a layer's blocks are not laid out slot by slot: the tail's slots can't be told apart")
    reqs = {r.request_id: r for r in B.unpickle_requests(st["requests"], {"block_hasher": None})}
    s = st["scheduler"]
    order = [x.key for q in ("running", "waiting", "skipped_waiting") for x in (s.get(q) or [])]
    if sorted(order) != sorted(reqs):
        raise SystemExit(f"the served queues hold {len(order)} requests, the scheduler {len(reqs)}")
    served_blocks = {rid: [b.key for b in ids] for rid, ids in st["managers"][0]["req_to_blocks"].items()}
    rows = B.KvPool(runner.kv_caches, len(pool.blocks)).rows
    dtypes = [getattr(torch, v["dtype"].split(".")[-1]) for v in st["kv_layout"]["views"]]
    slot = int(rows[0].shape[1]) // block_size
    if tail == "nan":
        for r, dt in zip(rows, dtypes, strict=True):
            r.view(dt).fill_(float("nan"))
    t = time.monotonic()
    pairs: list[tuple[int, int]] = []
    partial: list[tuple[int, int]] = []
    statuses: dict[str, int] = {}
    for rid in order:
        r = reqs[rid]
        statuses[r.status.name] = statuses.get(r.status.name, 0) + 1
        a = arrived[rid]
        eng.add_request(rid, {"prompt_token_ids": list(r.prompt_token_ids)},
                        R.params({"sampling": a["sampling"], "max_tokens": a["sampling"].get("max_tokens")}))
        q = sched.requests[rid]
        outs = list(r.output_token_ids)
        if outs:
            q.append_output_token_ids(outs)
        n = min(r.num_computed_tokens, q.num_tokens - 1)
        if n <= 0:
            if fresh is not None:
                fresh(q)
            continue
        if kvm.allocate_slots(q, 0, num_external_computed_tokens=n, delay_cache_blocks=True) is None:
            raise SystemExit(f"no room in the fresh pool for {rid}'s {n} tokens")
        q.num_computed_tokens = n
        kvm.cache_blocks(q, n)
        k = -(-n // block_size)
        new, old = kvm.get_block_ids(rid)[0][:k], served_blocks[rid][:k]
        if len(new) < k or len(old) < k:
            raise SystemExit(f"{rid}: {n} tokens in {len(old)} served and {len(new)} fresh blocks")
        pairs += zip(old, new, strict=True)
        if n % block_size:
            partial.append((new[-1], n % block_size))
    alloc_s = time.monotonic() - t
    t = time.monotonic()
    if pairs:
        old_ids = np.array([p[0] for p in pairs])
        new_ids = torch.tensor([p[1] for p in pairs], device=rows[0].device)
        for src, dst in zip(layers, rows, strict=True):
            dst.index_copy_(0, new_ids, torch.from_numpy(np.ascontiguousarray(src[old_ids])).to(dst.device))
    for b, j in partial:
        for dst, dt in zip(rows, dtypes, strict=True):
            if tail == "zero":
                dst[b, j * slot:].zero_()
            elif tail == "nan":
                dst[b, j * slot:].view(dt).fill_(float("nan"))
    torch.cuda.synchronize()
    return {"requests": len(order), "statuses": statuses, "with_prefix": len({p for p in order if reqs[p].num_computed_tokens > 0}),
            "blocks_loaded": len(pairs), "bytes_loaded": len(pairs) * int(rows[0].shape[1]) * len(rows),
            "partial_blocks": len(partial), "alloc_s": alloc_s, "load_s": time.monotonic() - t, "tail": tail,
            "order": order}


def step_diff(want: dict[int, dict], got: dict[int, dict], first: int, readded: set[str]) -> dict[str, Any]:
    """`restore.step_diff`, with the re-added requests left out of the first step's ``new``, ``resumed`` and ``cache_hit``
    on both sides: they are new to the fresh engine by construction, and a request the served engine had preempted
    resumes there."""
    def drop(steps: dict[int, dict]) -> dict[int, dict]:
        if first not in steps:
            return steps
        r = dict(steps[first])
        r["new"] = [x for x in r.get("new") or () if x not in readded]
        r["resumed"] = [x for x in r.get("resumed") or () if x not in readded]
        r["cache_hit"] = {k: v for k, v in (r.get("cache_hit") or {}).items() if k not in readded}
        return steps | {first: r}
    return R.step_diff(drop(want), drop(got))


def split_outputs(diff_of: Any, served: dict[int, dict], rebuilt: dict[int, dict], readded: set[str]) -> dict[str, Any]:
    """`boundary.outputs_diff` over the re-added requests (the batch-mates) and over the rest (the requests whose lifetime
    starts inside the block, as a sampled request's does) apart."""
    def only(outs: dict[int, dict], keep: bool) -> dict[int, dict]:
        return {s: {r: v for r, v in o.items() if (r in readded) == keep} for s, o in outs.items()}
    return {"readded": diff_of(only(served, True), only(rebuilt, True)),
            "arrived": diff_of(only(served, False), only(rebuilt, False))}


def run(a: argparse.Namespace) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(ROOT / "integrations/vllm"))
    sys.path.insert(0, str(HERE))
    from servetax_plugin import step_outputs_hook
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    meta = json.loads((a.arm_dir / "meta.json").read_text())
    w = R.served(a.arm_dir, a.kv_dir, a.from_edge, a.to_edge)
    st0, st1 = w["state"][a.from_edge], w["state"][a.to_edge]
    files = {k: a.kv_dir / f"edge-{k}.kv" for k in (a.from_edge, a.to_edge)}
    for f in files.values():
        if not f.is_file():
            raise SystemExit(f"{f}: the served arm kept no K/V file at this edge (serve_tax --save-kv)")
    checkpoint = a.checkpoint or Path(meta["checkpoint"])
    log_stats = "--disable-log-stats" not in meta["server_argv"]
    bs = int(w["header"]["config"]["block_size"])
    arrived = {r["request"]: r for r in S.trace_records(S.step_traces(a.arm_dir)[0]) if r["kind"] == "arrive"}

    t = time.monotonic()
    eng = R.engine(checkpoint, w["header"], False, log_stats)
    out: dict[str, Any] = {"engine_ready_s": time.monotonic() - t}
    core = eng.engine_core.engine_core
    sched, kvm, pool, runner = B.parts(core)
    trace = StepTrace().install(eng)
    kv = B.KvPool(runner.kv_caches, len(pool.blocks))
    if kv.layout() != st0["kv_layout"]:
        raise SystemExit("the fresh engine's KV pool is laid out unlike the served one's")
    near_file = kv_file(files[a.from_edge], st0["kv_layout"])
    hits = Hits(kvm, kv.rows, near_file, st0["pool"], recorded_hits(w["steps"]), bs) if a.hits == "forced" else None
    out["rebuilt"] = readd(eng, core, st0, arrived, near_file, bs, a.tail, hits.seed if hits else None)
    readded = set(out["rebuilt"].pop("order"))
    trace.scheduled = st0["next_step"]
    t = time.monotonic()
    want0 = roots(near_file, served_spans(st0), bs)
    got0 = roots(kv.rows, live_spans(sched, kvm), bs)
    out["near_edge"] = roots_diff(want0, got0) | {"check_s": time.monotonic() - t}

    busy = B.Busy(lambda: torch.cuda.Event(enable_timing=True), first=st0["next_step"])
    runner._verity_busy = busy
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    B.busy_runner(GPUModelRunner, lambda r: getattr(r, "_verity_busy", None))
    outs: dict[int, dict] = {}
    hooks = Hooks(tag="serve-tax-rebuild")
    hooks.wrap(sched, "update_from_output", step_outputs_hook(trace, lambda rec: outs.__setitem__(rec["step"], rec["out"]),
                                                              lambda: True))
    if hits:
        hooks.wrap(kvm, "get_computed_blocks", hits.lookup)
    pending = sorted(w["arrive"], key=lambda r: r["before_step"])

    def add_due() -> None:
        while pending and pending[0]["before_step"] == trace.scheduled:
            r = pending.pop(0)
            eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]}, R.params(r))
            if hits:
                hits.seed(sched.requests[r["engine_id"]])
        if pending and pending[0]["before_step"] < trace.scheduled:
            raise SystemExit(f"the replay passed step {pending[0]['before_step']} before {pending[0]['engine_id']} arrived")

    def schedule(orig):
        def wrapped(*args, **kw):
            add_due()
            return orig(*args, **kw)
        return wrapped

    hooks.wrap(sched, "schedule", schedule)
    taken: dict[str, Any] = {}

    def take(e: B.Edges, info: dict) -> None:
        torch.cuda.synchronize()
        taken["drained_t"] = info["drained_t"]
        busy.reap(wait=True)
        taken["spans"] = live_spans(sched, kvm)
        taken["roots"] = roots(kv.rows, taken["spans"], bs)

    edges = B.Edges(core, B.at_steps([w["hi"]], lambda: trace.scheduled), take, time.monotonic)
    stepper = type(core).step_with_batch_queue
    core.step_fn = lambda: edges.step(stepper)
    start = time.monotonic()
    while not taken:
        if not (sched.has_requests() or core.batch_queue):
            add_due()
            if not sched.has_requests():
                raise SystemExit(f"the replay is idle at step {trace.scheduled} short of {w['hi']}")
        eng.step()
    out["replay_wall_s"] = taken["drained_t"] - start
    busy_s, busy_steps = busy.total(w["lo"], w["hi"])
    srv_edge = w["edges"][a.to_edge]
    out |= {"replay_busy_s": busy_s, "replay_busy_steps": busy_steps, "served_busy_s": srv_edge["since_last"]["busy_s"],
            "served_wall_s": srv_edge["since_last"]["wall_s"]}
    out["replay_over_served_busy"] = busy_s / out["served_busy_s"] if out["served_busy_s"] else None
    steps = {r["step"]: r for r in trace.records if r["kind"] == "step" and w["lo"] <= r["step"] < w["hi"]}
    t = time.monotonic()
    want1 = roots(kv_file(files[a.to_edge], st1["kv_layout"]), served_spans(st1), bs)
    out |= {"schedule": step_diff(w["steps"], steps, w["lo"], readded),
            "outputs": B.outputs_diff(w["outputs"], outs),
            "outputs_by_kind": split_outputs(B.outputs_diff, w["outputs"], outs, readded),
            "far_edge": roots_diff(want1, taken["roots"]) | {"check_s": time.monotonic() - t}}
    out["bit_exact"] = (out["near_edge"]["differing"] == 0 and out["near_edge"]["compared"] == out["near_edge"]["requests_served"]
                        and out["schedule"]["steps_differing"] == 0 and out["outputs"]["first"] is None
                        and out["far_edge"]["roots_equal"])
    out["hits"] = hits.report() if hits else {"mode": "engine"}
    hooks.uninstall()
    with open(a.out / "rebuild_outputs.jsonl", "w") as f:
        for s in sorted(outs):
            f.write(json.dumps({"step": s, "out": outs[s]}, separators=(",", ":")) + "\n")
    with open(a.out / "rebuild_steps.jsonl", "w") as f:
        for s in sorted(steps):
            f.write(json.dumps(steps[s], separators=(",", ":")) + "\n")
    return out | {"replay_block": {"from_edge": a.from_edge, "to_edge": a.to_edge, "from_step": w["lo"], "to_step": w["hi"],
                                   "arrivals": len(w["arrive"]), "requests_at_edge": len(readded)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--from-edge", type=int, default=1)
    ap.add_argument("--to-edge", type=int, default=2)
    ap.add_argument("--kv-dir", type=Path, required=True, help="where the served arm kept both edges' K/V files and states")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tail", choices=TAILS, default="copy")
    ap.add_argument("--hits", choices=("forced", "engine"), default="forced",
                    help="the served prefix-cache hits forced, their pages from the edge's K/V (`Hits`), or the fresh "
                         "engine's own lookups")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did: the scheduler iterates sets of request ids")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    a.out.mkdir(parents=True, exist_ok=True)
    got = run(a)
    rec = {"schema": "verity-bench/serve-tax/rebuild/v0", "arm_dir": str(a.arm_dir), "run_id": SHARED.RESEARCH_RUN_ID, **got}
    (a.out / "rebuild.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("bit_exact", "replay_wall_s", "replay_busy_s", "served_busy_s")} |
                     {"tail": a.tail, "near_differing": rec["near_edge"]["differing"],
                      "steps_differing": rec["schedule"]["steps_differing"], "outputs_first": rec["outputs"]["first"],
                      "readded_tokens_differing": rec["outputs_by_kind"]["readded"]["request_steps_tokens_differing"],
                      "arrived_tokens_differing": rec["outputs_by_kind"]["arrived"]["request_steps_tokens_differing"],
                      "far_roots_equal": rec["far_edge"]["roots_equal"], "far_differing": rec["far_edge"]["differing"],
                      "hits": {k: v for k, v in rec["hits"].items() if k != "short_first"}}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
