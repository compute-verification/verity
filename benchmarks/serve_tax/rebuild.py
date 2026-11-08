"""Rebuild a replay block's batch at its first step from the served K/V pages alone, as lifetime replay starts a sampled
request's replay block, replay to a far edge, and say how close the replay came.

    rebuild.py --arm-dir OUT/restore [--from-edge 1] [--to-edge 2] --kv-dir DIR [--checkpoint DIR] [--tail copy] --out DIR
    rebuild.py ... --tail nan --at-step S --mirror-out MDIR --out DIR    # then
    rebuild.py ... --tail nan --from-mirror MDIR --out DIR

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

A real replay starts at a sampled request's first step, inside the replay block, with its batch-mates mid-request and
their last pages partly filled. ``--at-step S --mirror-out MDIR`` replays from the edge as above, keeping lifetime
replay's host copy as it goes (`Mirror`: each step's written slots, copied once the GPU has finished the step), and right
before the engine schedules step S writes what a replay starts from (`start_state`: the scheduler's queues and each
request's counters, the mirror's rows, and each request's K/V root on the GPU), then stops; mirror.json says whether that
replay matched serving up to S. ``--from-mirror MDIR`` starts a fresh engine at S from that alone (`readd_mirror`):
outputs through S-1 are those processed by then and the record's for S-1, still in flight under async scheduling, and
nothing past a computed prefix is filled in. It refuses a step whose rows run a request the step before finished (async
scheduling's extra row: the fresh engine would need the finished request back to run it), and compares from S to the
far edge as rebuild.json does, with the roots at S against the GPU's.
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


def alloc_prefix(kvm: Any, pool: Any, q: Any, n: int, old: list[int], fresh_of: dict[int, int], bs: int) -> tuple[list[int], int]:
    """``q``'s ``n`` computed tokens allocated in the fresh pool as a KV connector's load does, its leading full blocks
    shared with a request allocated before wherever the served engine held them under one block id (``old``, the
    request's served block ids; ``fresh_of``, served id to fresh id so far), so the fresh pool holds as many blocks as
    the served one and preempts where it did. Returns the request's fresh block ids and how many lead ones are shared,
    which hold their bytes already."""
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    full = n // bs
    j = 0
    while j < full and old[j] in fresh_of:
        j += 1
    if j * bs == n:
        j -= 1  # allocate_slots places at least one token
    hit = KVCacheBlocks(([pool.blocks[fresh_of[b]] for b in old[:j]],)) if j else None
    if kvm.allocate_slots(q, 0, num_new_computed_tokens=j * bs, new_computed_blocks=hit, num_external_computed_tokens=n - j * bs,
                          delay_cache_blocks=True, has_scheduled_reqs=False) is None:
        raise SystemExit(f"no room in the fresh pool for {q.request_id}'s {n} tokens")
    q.num_computed_tokens = n
    kvm.cache_blocks(q, n)
    new = kvm.get_block_ids(q.request_id)[0][: -(-n // bs)]
    for a, b in zip(old[:full], new, strict=False):
        fresh_of.setdefault(a, b)
    return new, j


def recorded_starts(steps: dict[int, dict], first: int) -> dict[str, list[tuple[int, int]]]:
    """Each request's first position at every step from ``first`` on where the served engine scheduled it new or resumed
    it (its row's start: the prefix-cache hit, in tokens), in step order."""
    out: dict[str, list[tuple[int, int]]] = {}
    for s in sorted(t for t in steps if t >= first):
        rec = steps[s]
        start = {r[0]: r[1] for r in rec["rows"]}
        for rid in [*(rec.get("new") or ()), *(rec.get("resumed") or ())]:
            out.setdefault(rid, []).append((s, start[rid]))
    return out


class Hits:
    """The served engine's prefix-cache hits, forced on the fresh engine as lifetime replay's schedule takes them from
    the step record. A request's lookup (`lookup`, around `get_computed_blocks`) is cut to the hit the served engine took
    at the step being scheduled, and right before the step where the record has it scheduled new or resumed (`due`), each
    page of that hit the fresh cache lacks is loaded from ``pages`` (its block hash to each layer's row, or None) into a
    free block, cached under its hash and freed again, so the pool counts it free as the served pool counted its cached
    copy."""

    def __init__(self, kvm: Any, rows: list[Any], pages: Any, starts: dict[str, list[tuple[int, int]]], block_size: int,
                 now: Any) -> None:
        self.kvm, self.pool, self.rows, self.pages = kvm, kvm.block_pool, rows, pages
        self.starts, self.bs, self.now = starts, block_size, now
        self.by_step: dict[int, list[str]] = {}
        for rid, occ in starts.items():
            for s, _ in occ:
                self.by_step.setdefault(s, []).append(rid)
        self.loaded, self.cut, self.short, self.missing = 0, 0, [], 0

    def want_at(self, rid: str, step: int) -> int | None:
        return next((n for s, n in self.starts.get(rid, ()) if s >= step), None)

    def due(self, requests: dict[str, Any], step: int) -> None:
        for rid in self.by_step.get(step, ()):
            if rid in requests:
                self.seed(requests[rid], self.want_at(rid, step) or 0)

    def seed(self, q: Any, n: int) -> None:
        import torch
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        for i in range(n // self.bs):
            key = make_block_hash_with_group_id(q.block_hashes[i], 0)
            if self.pool.cached_block_hash_to_block.get_one_block(key) is not None:
                continue
            src = self.pages(q.block_hashes[i])
            if src is None:
                self.missing += 1
                continue
            (blk,) = self.pool.get_new_blocks(1)
            for row, dst in zip(src, self.rows, strict=True):
                dst[blk.block_id].copy_(torch.from_numpy(row).to(dst.device))
            self.pool._insert_block_hash(key, blk, self.bs)
            self.pool.free_blocks([blk])
            self.loaded += 1

    def lookup(self, orig: Any) -> Any:
        def wrapped(request: Any, *a: Any, **kw: Any) -> Any:
            blocks, n, *rest = orig(request, *a, **kw)
            want = self.want_at(request.request_id, self.now())
            if want is not None and n > want:
                blocks = self.kvm.create_kv_cache_blocks(tuple(g[: want // self.bs] for g in blocks.blocks))
                n, self.cut = want, self.cut + 1
            elif want is not None and n < want:
                self.short.append([request.request_id, n, want])
            return (blocks, n, *rest)
        return wrapped

    def report(self) -> dict[str, Any]:
        return {"requests_with_hits": sum(1 for occ in self.starts.values() if any(n for _, n in occ)),
                "pages_loaded": self.loaded, "pages_not_found": self.missing, "lookups_cut": self.cut,
                "lookups_short": len(self.short), "short_first": self.short[:20]}


def edge_pages(layers: list[Any], served_pool: dict[str, Any]) -> Any:
    """Pages from a served edge's K/V file, by block hash: the block the served pool cached under it there (the first, as
    its lookup takes)."""
    import numpy as np
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    def get(h: bytes) -> list[Any] | None:
        src = served_pool["cache"].get(make_block_hash_with_group_id(h, 0))
        if src is None:
            return None
        src = src if isinstance(src, int) else next(iter(src.values()))
        return [np.array(f[src]) for f in layers]
    return get


def first_of(*sources: Any) -> Any:
    def get(h: bytes) -> list[Any] | None:
        return next((p for p in (s(h) for s in sources) if p is not None), None)
    return get


class Mirror:
    """Lifetime replay's host copy of the K/V, slot by slot. `scheduled` notes the slots each step will write: each
    scheduled request's positions this step, from its start (and, for a request new or resumed this step, its hit's
    positions before it), at its block ids then. `flush`, called before the next schedule (which may free and reuse
    blocks) once the GPU has finished every step so far, copies them to each request's host rows, layer by layer, and
    records each full page under its block hash, the first writer's (the copy a served lookup takes first)."""

    def __init__(self, sched: Any, kvm: Any, rows: list[Any], block_size: int) -> None:
        self.sched, self.kvm, self.bs = sched, kvm, block_size
        self.slot = int(rows[0].shape[1]) // block_size
        self.flat = [r.view(r.shape[0] * block_size, self.slot) for r in rows]
        self.store: dict[str, Any] = {}
        self.written: dict[str, int] = {}
        self.hashes: dict[bytes, tuple[str, int]] = {}
        self.pending: list[tuple[str, int, int, list[int], list[bytes]]] = []
        self.copied_tokens = 0

    def scheduled(self, so: Any) -> None:
        from verity_vllm.engine.step_trace import step_geometry

        start, new, resumed, _ = step_geometry(so)
        fresh = set(new) | set(resumed)
        bs = self.bs
        for rid, n in so.num_scheduled_tokens.items():
            s = start[rid]
            lo, hi = (0 if rid in fresh else s), s + int(n)
            blocks = self.kvm.get_block_ids(rid)[0]
            self.pending.append((rid, lo, hi, [blocks[p // bs] * bs + p % bs for p in range(lo, hi)],
                                 list(self.sched.requests[rid].block_hashes[: hi // bs])))

    def flush(self) -> None:
        import numpy as np
        import torch

        if not self.pending:
            return
        torch.cuda.synchronize()
        idx = torch.tensor([x for p in self.pending for x in p[3]], device=self.flat[0].device)
        got = torch.stack([f.index_select(0, idx) for f in self.flat], dim=1).cpu().numpy()
        at = 0
        for rid, lo, hi, _, hs in self.pending:
            arr = self.store.get(rid)
            if arr is None or arr.shape[0] < hi:
                grown = np.zeros((max(hi, 2 * (0 if arr is None else arr.shape[0])), len(self.flat), self.slot), np.int8)
                if arr is not None:
                    grown[: arr.shape[0]] = arr
                arr = self.store[rid] = grown
            arr[lo:hi] = got[at:at + hi - lo]
            at += hi - lo
            self.written[rid] = max(self.written.get(rid, 0), hi)
            for p, h in enumerate(hs):
                self.hashes.setdefault(h, (rid, p))
        self.copied_tokens += at
        self.pending.clear()

    def rows(self, rid: str, n: int) -> Any:
        return self.store[rid][:n]

    def page(self, h: bytes) -> list[Any] | None:
        """A page's rows per layer, laid out as the pool's (``block_size`` slots, one after another)."""
        got = self.hashes.get(h)
        if got is None:
            return None
        rid, p = got
        x = self.store[rid][p * self.bs:(p + 1) * self.bs]
        return [x[:, i, :].reshape(-1).copy() for i in range(x.shape[1])]

    def root(self, rid: str, n: int) -> str:
        h = hashlib.sha512()
        x = self.store[rid]
        for i in range(x.shape[1]):
            h.update(x[:n, i, :].tobytes())
        return h.hexdigest()

    def save(self, path: Path, keep: set[str]) -> dict[str, Any]:
        """The host rows of every request it holds (``keep``'s whole, the rest only up to their last full page) to
        ``path`` (.bin, int8 tokens x layers x slot) with an index (.json): each request's token offset and count, and
        each page's block hash (hex) to its request and page."""
        idx: dict[str, list[int]] = {}
        at = 0
        with open(path.with_suffix(".bin"), "wb") as f:
            for rid, arr in self.store.items():
                n = self.written[rid] if rid in keep else self.written[rid] // self.bs * self.bs
                if not n:
                    continue
                f.write(arr[:n].tobytes())
                idx[rid] = [at, n]
                at += n
        meta = {"layers": len(self.flat), "slot": self.slot, "block_size": self.bs, "tokens": at, "requests": idx,
                "pages": {h.hex(): [r, p] for h, (r, p) in self.hashes.items() if r in idx}}
        path.with_suffix(".json").write_text(json.dumps(meta))
        return {"tokens": at, "bytes": at * len(self.flat) * self.slot, "requests": len(idx), "pages": len(meta["pages"])}


class Saved:
    """A `Mirror` that `Mirror.save` wrote, read back (rows by request, pages by block hash)."""

    def __init__(self, path: Path) -> None:
        import numpy as np

        self.meta = json.loads(path.with_suffix(".json").read_text())
        L, slot, self.bs = self.meta["layers"], self.meta["slot"], self.meta["block_size"]
        self.data = np.memmap(path.with_suffix(".bin"), dtype=np.int8, mode="r", shape=(self.meta["tokens"], L, slot))
        self.hashes = {bytes.fromhex(h): (r, p) for h, (r, p) in self.meta["pages"].items()}

    def rows(self, rid: str, n: int) -> Any:
        if rid not in self.meta["requests"]:
            raise SystemExit(f"{rid}: the mirror holds none of its tokens, the start needs {n}")
        at, have = self.meta["requests"][rid]
        if n > have:
            raise SystemExit(f"{rid}: the mirror holds {have} tokens, the start needs {n}")
        return self.data[at:at + n]

    def page(self, h: bytes) -> list[Any] | None:
        import numpy as np

        got = self.hashes.get(h)
        if got is None:
            return None
        rid, p = got
        x = self.rows(rid, (p + 1) * self.bs)[p * self.bs:]
        return [np.ascontiguousarray(x[:, i, :]).reshape(-1) for i in range(x.shape[1])]


def readd(eng: Any, core: Any, st: dict[str, Any], arrived: dict[str, dict], layers: list[Any], block_size: int,
          tail: str) -> dict[str, Any]:
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
    fresh_of: dict[int, int] = {}
    shared = 0
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
            continue
        k = -(-n // block_size)
        old = served_blocks[rid][:k]
        if len(old) < k:
            raise SystemExit(f"{rid}: {n} tokens in {len(old)} served blocks")
        new, j = alloc_prefix(kvm, pool, q, n, old, fresh_of, block_size)
        shared += j
        pairs += zip(old[j:], new[j:], strict=True)
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
            "blocks_loaded": len(pairs), "blocks_shared": shared, "blocks_served": len({b for v in served_blocks.values() for b in v}),
            "blocks_fresh": pool.num_gpu_blocks - 1 - pool.get_num_free_blocks(), "bytes_loaded": len(pairs) * int(rows[0].shape[1]) * len(rows),
            "partial_blocks": len(partial), "alloc_s": alloc_s, "load_s": time.monotonic() - t, "tail": tail,
            "order": order}


def readd_mirror(eng: Any, core: Any, start: dict[str, Any], mirror: Any, served_outs: dict[int, dict],
                 layout: dict[str, Any], block_size: int, tail: str) -> dict[str, Any]:
    """Every request ``start`` (the scheduler's queues when it scheduled step S, `Mirror` mode's start.json) holds and the
    step record doesn't finish before S, added to the fresh engine as `readd` adds an edge's: with its outputs through
    step S-1 (those processed by then, and the record's since), and its computed prefix, the served count when it
    scheduled S (step S-1's tokens included), copied from the mirror's rows, whose slots past it hold nothing."""
    import numpy as np
    import torch

    sched, kvm, pool, runner = B.parts(core)
    if getattr(sched, "needs_kv_cache_zeroing", False):
        raise SystemExit("this pool zeroes newly allocated blocks, which would clear the loaded pages")
    if tail == "copy":
        raise SystemExit("a mirror holds no bytes past a request's computed tokens: --tail zero or nan")
    rows = B.KvPool(runner.kv_caches, len(pool.blocks)).rows
    dtypes = [getattr(torch, v["dtype"].split(".")[-1]) for v in layout["views"]]
    L, slot, bs = len(rows), int(rows[0].shape[1]) // block_size, block_size
    if tail == "nan":
        for r, dt in zip(rows, dtypes, strict=True):
            r.view(dt).fill_(float("nan"))
    S, done = start["step"], start["processed_through"]
    t = time.monotonic()
    ids: list[int] = []
    chunks: list[Any] = []
    partial: list[tuple[int, int]] = []
    statuses: dict[str, int] = {}
    fresh_of: dict[int, int] = {}
    shared = 0
    order = []
    for rid in start["order"]:
        if rid in start["finished_before"]:
            continue
        r = start["requests"][rid]
        order.append(rid)
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        eng.add_request(rid, {"prompt_token_ids": r["prompt"]},
                        R.params({"sampling": r["sampling"], "max_tokens": r["sampling"].get("max_tokens")}))
        q = sched.requests[rid]
        outs = list(r["outputs"]) + [x for s in range(done + 1, S) for x in (served_outs.get(s, {}).get(rid) or [[]])[0]]
        if outs:
            q.append_output_token_ids(outs)
        n = min(r["computed"], q.num_tokens - 1)
        if n <= 0:
            continue
        k = -(-n // bs)
        new, j = alloc_prefix(kvm, pool, q, n, start["blocks"][rid][:k], fresh_of, bs)
        shared += j
        x = np.zeros((k * bs, L, slot), np.int8)
        x[:n] = mirror.rows(rid, n)
        chunks.append(x.reshape(k, bs, L, slot).transpose(2, 0, 1, 3).reshape(L, k, bs * slot)[:, j:])
        ids += new[j:]
        if n % bs:
            partial.append((new[-1], n % bs))
    alloc_s = time.monotonic() - t
    t = time.monotonic()
    if ids:
        dev = torch.tensor(ids, device=rows[0].device)
        for i, dst in enumerate(rows):
            dst.index_copy_(0, dev, torch.from_numpy(np.ascontiguousarray(np.concatenate([c[i] for c in chunks]))).to(dst.device))
    for b, j in partial:
        for dst, dt in zip(rows, dtypes, strict=True):
            if tail == "zero":
                dst[b, j * slot:].zero_()
            else:
                dst[b, j * slot:].view(dt).fill_(float("nan"))
    torch.cuda.synchronize()
    served = {b for rid in order for b in start["blocks"].get(rid, ())}
    return {"requests": len(order), "statuses": statuses, "skipped_finished": len(start["finished_before"]),
            "blocks_loaded": len(ids), "blocks_shared": shared, "blocks_served": len(served),
            "blocks_fresh": pool.num_gpu_blocks - 1 - pool.get_num_free_blocks(),
            "bytes_loaded": len(ids) * int(rows[0].shape[1]) * L, "partial_blocks": len(partial),
            "alloc_s": alloc_s, "load_s": time.monotonic() - t, "tail": tail, "order": order}


def step_diff(want: dict[int, dict], got: dict[int, dict], first: int, readded: set[str]) -> dict[str, Any]:
    """`restore.step_diff`, with each re-added request left out of ``new``, ``resumed`` and ``cache_hit`` on both sides at
    the step the fresh engine first schedules it: it is new to the fresh engine by construction, and a request the served
    engine had preempted resumes there."""
    at: dict[int, set[str]] = {}
    for s in sorted(got):
        for x in got[s].get("new") or ():
            if x in readded and not any(x in v for v in at.values()):
                at.setdefault(s, set()).add(x)

    def drop(steps: dict[int, dict]) -> dict[int, dict]:
        out = dict(steps)
        for s, ids in at.items():
            if s in out:
                r = dict(out[s])
                r["new"] = [x for x in r.get("new") or () if x not in ids]
                r["resumed"] = [x for x in r.get("resumed") or () if x not in ids]
                r["cache_hit"] = {k: v for k, v in (r.get("cache_hit") or {}).items() if k not in ids}
                out[s] = r
        return out
    return R.step_diff(drop({s: v for s, v in want.items() if s >= first}), drop(got))


class _Stop(Exception):
    pass


def start_state(sched: Any, kvm: Any, kv: Any, mirror: Mirror, arrived: dict[str, dict], served_steps: dict[int, dict],
                done: int, step: int, block_size: int) -> dict[str, Any]:
    """What lifetime replay starts step ``step`` from, taken right before the served-identical engine schedules it (every
    step before has finished on the GPU and the mirror holds its slots): the scheduler's queues in order, each request's
    prompt, processed outputs, computed tokens (step S-1's included), status and sampling, the last step whose output was
    processed, the requests the record finishes after it and before S, and each request's K/V root over its computed
    tokens on the GPU and in the mirror."""
    order = [r.request_id for q in (sched.running, sched.waiting, getattr(sched, "skipped_waiting", ())) for r in q]
    reqs = {}
    for rid in order:
        q = sched.requests[rid]
        reqs[rid] = {"prompt": list(q.prompt_token_ids), "outputs": list(q.output_token_ids), "computed": int(q.num_computed_tokens),
                     "placeholders": int(getattr(q, "num_output_placeholders", 0)), "status": q.status.name,
                     "preemptions": int(q.num_preemptions), "sampling": arrived[rid]["sampling"]}
    finished = sorted({rid for s in range(done + 1, step) for rid in (served_steps.get(s, {}).get("finished") or {})} & set(order))
    zombies = sorted(set(finished) & {r[0] for r in served_steps[step]["rows"]})
    if zombies:
        raise SystemExit(f"step {step} runs {zombies[:5]}, finished at the step before under async scheduling: start elsewhere")
    spans = {rid: v for rid, v in live_spans(sched, kvm).items() if rid in reqs}
    truth = roots(kv.rows, spans, block_size)
    mine = {rid: mirror.root(rid, n) for rid, (_, n) in spans.items() if rid in mirror.store and mirror.written[rid] >= n}
    return {"step": step, "processed_through": done, "order": order, "requests": reqs, "finished_before": finished,
            "blocks": {rid: list(ids) for rid, (ids, _) in spans.items()}, "truth": truth, "mirror_vs_gpu": roots_diff(truth, mine),
            "preempted_waiting": sorted(r for r in order if reqs[r]["status"] == "PREEMPTED"),
            "partial_last_pages": sum(1 for _, n in spans.values() if n % block_size)}


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
    begin = json.loads((a.from_mirror / "start.json").read_text()) if a.from_mirror else None
    first = begin["step"] if begin else w["lo"]
    if a.at_step is not None and not w["lo"] < a.at_step < w["hi"]:
        raise SystemExit(f"--at-step {a.at_step}: not inside the replay block's steps {w['lo']}..{w['hi'] - 1}")

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
    pages = edge_pages(near_file, st0["pool"])
    # forced hits read pages written inside the block from the mirror, as lifetime replay reads its host copy, where the
    # fresh pool evicted a page the served one still had cached
    mirror = Mirror(sched, kvm, kv.rows, bs) if a.mirror_out or a.hits == "forced" else None
    t = time.monotonic()
    if begin:
        saved = Saved(a.from_mirror / "mirror")
        pages = first_of(pages, saved.page, *([mirror.page] if mirror else []))
        out["rebuilt"] = readd_mirror(eng, core, begin, saved, w["outputs"], st0["kv_layout"], bs, a.tail)
        readded = set(out["rebuilt"].pop("order"))
        want0 = {r: v for r, v in begin["truth"].items() if r in readded}
        out["start"] = {k: begin[k] for k in ("step", "processed_through", "finished_before", "mirror_vs_gpu", "preempted_waiting",
                                              "partial_last_pages")}
    else:
        if mirror:
            pages = first_of(pages, mirror.page)
        out["rebuilt"] = readd(eng, core, st0, arrived, near_file, bs, a.tail)
        readded = set(out["rebuilt"].pop("order"))
        want0 = roots(near_file, served_spans(st0), bs)
    trace.scheduled = first
    got0 = roots(kv.rows, live_spans(sched, kvm), bs)
    out["near_edge"] = roots_diff(want0, got0) | {"check_s": time.monotonic() - t}
    hits = Hits(kvm, kv.rows, pages, recorded_starts(w["steps"], first), bs, lambda: trace.scheduled) if a.hits == "forced" else None

    busy = B.Busy(lambda: torch.cuda.Event(enable_timing=True), first=first)
    runner._verity_busy = busy
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    B.busy_runner(GPUModelRunner, lambda r: getattr(r, "_verity_busy", None))
    outs: dict[int, dict] = {}
    hooks = Hooks(tag="serve-tax-rebuild")
    hooks.wrap(sched, "update_from_output", step_outputs_hook(trace, lambda rec: outs.__setitem__(rec["step"], rec["out"]),
                                                              lambda: True))
    if hits:
        hooks.wrap(kvm, "get_computed_blocks", hits.lookup)
    pending = sorted((r for r in w["arrive"] if r["before_step"] >= first), key=lambda r: r["before_step"])

    def add_due() -> None:
        while pending and pending[0]["before_step"] == trace.scheduled:
            r = pending.pop(0)
            eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]}, R.params(r))
        if pending and pending[0]["before_step"] < trace.scheduled:
            raise SystemExit(f"the replay passed step {pending[0]['before_step']} before {pending[0]['engine_id']} arrived")

    stop: dict[str, Any] = {}

    def schedule(orig):
        def wrapped(*args, **kw):
            if mirror:
                mirror.flush()
                if a.mirror_out and trace.scheduled == a.at_step:
                    stop["start"] = start_state(sched, kvm, kv, mirror, arrived, w["steps"], max(outs, default=first - 1),
                                                a.at_step, bs)
                    raise _Stop
            add_due()
            if hits:
                hits.due(sched.requests, trace.scheduled)
            so = orig(*args, **kw)
            if mirror and so.total_num_scheduled_tokens:
                mirror.scheduled(so)
            return so
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
    try:
        while not taken:
            if not (sched.has_requests() or core.batch_queue):
                add_due()
                if not sched.has_requests():
                    raise SystemExit(f"the replay is idle at step {trace.scheduled} short of {w['hi']}")
            eng.step()
    except _Stop:
        pass
    steps = {r["step"]: r for r in trace.records if r["kind"] == "step" and first <= r["step"] < w["hi"]}
    block = {"from_edge": a.from_edge, "to_edge": a.to_edge, "from_step": w["lo"], "to_step": w["hi"], "first_step": first,
             "arrivals": len(w["arrive"]), "requests_at_start": len(readded)}
    if a.mirror_out:
        hooks.uninstall()
        st = stop["start"]
        keep = set(st["order"])
        t = time.monotonic()
        out["mirror"] = mirror.save(a.mirror_out / "mirror", keep) | {"save_s": time.monotonic() - t, "copied_tokens": mirror.copied_tokens}
        (a.mirror_out / "start.json").write_text(json.dumps(st))
        upto = {s: v for s, v in w["outputs"].items() if s < a.at_step - 1}
        mine = {s: v for s, v in outs.items() if s < a.at_step - 1}
        sched_upto = {s: v for s, v in w["steps"].items() if s < a.at_step - 1}
        out |= {"replay_wall_s": time.monotonic() - start, "at_step": a.at_step,
                "before_start": {"schedule": step_diff(sched_upto, {s: v for s, v in steps.items() if s < a.at_step - 1}, first, readded),
                                 "outputs": B.outputs_diff(upto, mine)},
                "start": {k: v for k, v in st.items() if k not in ("requests", "truth", "order")} | {"requests": len(st["order"])},
                "hits": hits.report() if hits else {"mode": "engine"}, "replay_block": block}
        b = out["before_start"]
        out["served_identical_to_start"] = (out["near_edge"]["differing"] == 0 and b["schedule"]["steps_differing"] == 0
                                            and b["outputs"]["first"] is None and st["mirror_vs_gpu"]["differing"] == 0
                                            and st["mirror_vs_gpu"]["compared"] == st["mirror_vs_gpu"]["requests_served"])
        return out
    out["replay_wall_s"] = taken["drained_t"] - start
    busy_s, busy_steps = busy.total(first, w["hi"])
    srv_edge = w["edges"][a.to_edge]
    out |= {"replay_busy_s": busy_s, "replay_busy_steps": busy_steps, "served_busy_s": srv_edge["since_last"]["busy_s"],
            "served_wall_s": srv_edge["since_last"]["wall_s"]}
    out["replay_over_served_busy"] = busy_s / out["served_busy_s"] if out["served_busy_s"] and first == w["lo"] else None
    t = time.monotonic()
    want1 = roots(kv_file(files[a.to_edge], st1["kv_layout"]), served_spans(st1), bs)
    served_outs = {s: v for s, v in w["outputs"].items() if s >= first}
    out |= {"schedule": step_diff(w["steps"], steps, first, readded),
            "outputs": B.outputs_diff(served_outs, outs),
            "outputs_by_kind": split_outputs(B.outputs_diff, served_outs, outs, readded),
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
    return out | {"replay_block": block}


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
    ap.add_argument("--at-step", type=int, metavar="S",
                    help="with --mirror-out: replay from the edge, copying each step's slots (`Mirror`), up to step S, and "
                         "write the start state and mirror there instead of replaying on")
    ap.add_argument("--mirror-out", type=Path, metavar="DIR")
    ap.add_argument("--from-mirror", type=Path, metavar="DIR",
                    help="start at the step a --mirror-out run stopped at, from its start state and mirror alone")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if bool(a.at_step is not None) != bool(a.mirror_out) or (a.mirror_out and a.from_mirror):
        raise SystemExit("--at-step and --mirror-out go together, and not with --from-mirror")
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did: the scheduler iterates sets of request ids")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mirror_out:
        a.mirror_out.mkdir(parents=True, exist_ok=True)
    got = run(a)
    rec = {"schema": "verity-bench/serve-tax/rebuild/v0", "arm_dir": str(a.arm_dir), "run_id": SHARED.RESEARCH_RUN_ID, **got}
    (a.out / ("mirror.json" if a.mirror_out else "rebuild.json")).write_text(json.dumps(rec, indent=1, default=str))
    if a.mirror_out:
        print(json.dumps({k: rec.get(k) for k in ("served_identical_to_start", "at_step", "mirror", "replay_wall_s")} |
                         {"mirror_vs_gpu": {k: rec["start"]["mirror_vs_gpu"][k] for k in ("compared", "differing")},
                          "preempted_waiting": rec["start"]["preempted_waiting"],
                          "partial_last_pages": rec["start"]["partial_last_pages"],
                          "before_steps_differing": rec["before_start"]["schedule"]["steps_differing"],
                          "before_outputs_first": rec["before_start"]["outputs"]["first"]}, default=str))
        return 0
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
