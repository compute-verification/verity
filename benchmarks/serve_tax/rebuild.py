"""Rebuild a replay block's batch at its first step from the served K/V pages alone, as lifetime replay starts a sampled
request's replay block, replay to a far edge, and say how close the replay came.

    rebuild.py --arm-dir OUT/restore [--from-edge 1] [--to-edge 2] --kv-dir DIR [--checkpoint DIR] [--tail copy] --out DIR
    rebuild.py ... --tail nan --at-step S --mirror-out MDIR --out DIR    # then
    rebuild.py ... --tail nan --from-mirror MDIR --out DIR
    rebuild.py --served OUT/ARM/serve --scratch DIR [--from-edge 1] [--to-edge 2] --tail copy
               [--shape-only RID [--capture TEMPLATE]] --out DIR

A restore (`restore.py`) puts the served engine's whole state back: the block pool, the scheduler and the model runner's
rows. Lifetime replay has less: a host copy of each K/V page, written when the page was, and the step record. So here a
fresh engine gets each request the served engine held at the edge as a new request, in the served running then waiting
order, with its prompt, its outputs so far, its served id and sampling; and its computed prefix's pages are copied from
the edge's K/V file into blocks the fresh engine allocates, as a KV connector's load does (`allocate_slots` with the
prefix as external tokens, then `cache_blocks`). Block ids, runner slots, the prefix cache and every other piece of
engine state are the fresh engine's own. ``--tail`` sets what a re-added request's last, partial block holds past its
prefix: the served bytes (``copy``), zeros (``zero``), or NaN, with every other slot of the fresh pool NaN too
(``nan``: then nothing outside a computed prefix may be read). The served arm must have kept the K/V file of both edges
(serve_tax's ``--save-kv FROM,TO``). The served generators are put back, each re-added request gets its served seed
when the runner adds it (`carry_rng`), and one the served engine had preempted is resumed rather than new
(`keep_preempted`), so unseeded sampling draws as serving drew.

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

``--shape-only RID`` replays one request with every other request a dummy of its shape (`Dummies`), from the edge or
from a mirror: a dummy keeps its lengths, its arrival, its prefix-cache hits and its served finish, and nothing of its
values. Its tokens are substitutes, each full page's derived from that page's chain hash so that pages equal in serving
stay equal (RID's own pages kept), and its K/V is never loaded: zero or NaN (``--dummy``) at the start, in every page a
forced hit seeds for it, and in every slot it writes, refilled once the GPU has finished the step. A dummy runs until
its served output count (``ignore_eos``), so it finishes where serving did. RID's prefix and every page it hits come
from serving's bytes: the edge's K/V file, or the mirror, which a replay identical to serving wrote. shape.json compares
RID's tokens and logprob bits and the schedule (a dummy's finish reason aside) up to RID's last step, and records what
the replay took from the run.

``--served`` replays an arm `tp_restore.py serve` wrote, at any tensor-parallel size: each edge's state
(`boundary:save_edge`'s ``state.pkl``) and the first edges' whole-pool dumps (``--dump``) under its scratch directory.
The engine is built again in this process from the served argv, and every page is loaded, filled or hashed on each rank
through `collective_rpc` (`RankKv`). A dump holds whole blocks, so the tail is the served bytes (``--tail copy``) and
the roots cover each request's full pages. A forced hit's page comes from the first edge's dump, or, written after the
replay's start, from the far edge's. A pool vLLM zeroes as it hands out blocks (mamba or mixed-precision K/V) keeps
the pages `readd` loaded, as a KV connector's. There is no mirror here. With ``--shape-only``, ``--capture TEMPLATE`` takes
the target's logits rows and ids at every rank's sampler (`verity_vllm.engine.sampler_capture`, each rank's rows under the
scratch) and writes rank 0's to OUT/capture as a span of TEMPLATE's units; shape.json says whether its ids are the tokens
serving emitted for the target.

Zero is the replay. NaN probes for reads at weight zero, which no finite dummy can show: FA3's causal varlen prefill
weights the sequence packed after RID by zero in RID's last partial key tile, so a NaN dummy there turns RID's rows NaN
where any finite one leaves them exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections.abc import Callable, Iterable
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
import tp_state as T  # noqa: E402

TAILS = ("copy", "zero", "nan")
#: substitute token ids for dummies: clear of the low ids and of the special tokens at the top of the vocabulary
SUBSTITUTE_IDS = (1000, 150000)


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


#: `Hits`' write for a dummy's seeded page: the dummies' fill, not a page's bytes
FILL_PAGE = "fill"
#: an ``own`` source's answer on a tensor-parallel engine: the target wrote the page itself, and the fresh cache holds it
OWN_PAGE = "own"


class Hits:
    """The served engine's prefix-cache hits, forced on the fresh engine as lifetime replay's schedule takes them from
    the step record. A request's lookup (`lookup`, around `get_computed_blocks`) is cut to the hit the served engine took
    at the step being scheduled, and right before the step where the record has it scheduled new or resumed (`due`), each
    page of that hit the fresh cache lacks is loaded from ``pages`` (its block hash to a source ``kv`` writes: each
    layer's row, or a dump's block; or None) into a free block, cached under its hash and freed again, so the pool counts
    it free as the served pool counted its cached copy. A step's writes go to ``kv`` together, each block's last."""

    def __init__(self, kvm: Any, kv: Any, pages: Any, starts: dict[str, list[tuple[int, int]]], block_size: int,
                 now: Any, shape: Dummies | None = None, own: Any = None) -> None:
        self.kvm, self.pool, self.kv, self.pages = kvm, kvm.block_pool, kv, pages
        self.starts, self.bs, self.now = starts, block_size, now
        # shape-only: a dummy's missing page is seeded poisoned; each page the target hits is written from serving's
        # bytes (``pages``) even where the fresh cache holds it, or is one the target wrote itself (``own``)
        self.shape, self.own = shape, own
        self.by_step: dict[int, list[str]] = {}
        for rid, occ in starts.items():
            for s, _ in occ:
                self.by_step.setdefault(s, []).append(rid)
        self.loaded, self.cut, self.short, self.missing = 0, 0, [], 0
        self.ops: dict[int, Any] = {}

    def want_at(self, rid: str, step: int) -> int | None:
        return next((n for s, n in self.starts.get(rid, ()) if s >= step), None)

    def due(self, requests: dict[str, Any], step: int) -> None:
        for rid in self.by_step.get(step, ()):
            if rid in requests:
                self.seed(requests[rid], self.want_at(rid, step) or 0)
        self.put()

    def put(self) -> None:
        if self.ops:
            self.kv.write(self.ops, self.shape.mode if self.shape else "zero")
            self.ops = {}

    def insert(self, key: Any, src: Any) -> None:
        (blk,) = self.pool.get_new_blocks(1)
        self.ops[blk.block_id] = src
        self.pool._insert_block_hash(key, blk, self.bs)
        self.pool.free_blocks([blk])

    def seed(self, q: Any, n: int) -> None:
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        sh = self.shape
        for i in range(n // self.bs):
            key = make_block_hash_with_group_id(q.block_hashes[i], 0)
            have = self.pool.cached_block_hash_to_block.get_one_block(key)
            if sh is not None and sh.dummy(q.request_id):
                if have is None:
                    self.insert(key, FILL_PAGE)
                    sh.taken["pages_seeded_poisoned"] += 1
                continue
            if sh is not None:
                src = self.pages(q.block_hashes[i])
                if src is None:
                    src = self.own(q.block_hashes[i])
                    if src is None or (isinstance(src, str) and have is None):
                        sh.taken["target_pages_unavailable"] += 1
                        continue
                    sh.taken["target_pages_own"] += 1
                    if have is not None:
                        continue
                elif have is not None:
                    self.ops[have.block_id] = src
                    sh.taken["target_pages_overwritten"] += 1
                    continue
                else:
                    sh.taken["target_pages_loaded"] += 1
            elif have is not None:
                continue
            else:
                src = self.pages(q.block_hashes[i])
            if src is None:
                self.missing += 1
                continue
            self.insert(key, src)
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
    """Pages from a served edge's K/V file, by block hash: the block the served pool cached under it there (`served_block`)."""
    import numpy as np

    def get(h: bytes) -> list[Any] | None:
        src = served_block(served_pool, h)
        return None if src is None else [np.array(f[src]) for f in layers]
    return get


def served_block(served_pool: dict[str, Any], h: bytes) -> int | None:
    """The block a served pool (`boundary.pool_state`) cached a page under, by its hash: the first, as its lookup takes."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    src = served_pool["cache"].get(make_block_hash_with_group_id(h, 0))
    if src is None:
        return None
    return src if isinstance(src, int) else next(iter(src.values()))


def dump_pages(edge_dir: Path, served_pool: dict[str, Any]) -> Any:
    """Pages from a tensor-parallel edge's dump (`tp_edge`), by block hash: (the edge's directory, the served block)."""
    def get(h: bytes) -> tuple[str, int] | None:
        src = served_block(served_pool, h)
        return None if src is None else (str(edge_dir), src)
    return get


def first_of(*sources: Any) -> Any:
    def get(h: bytes) -> list[Any] | None:
        return next((p for p in (s(h) for s in sources) if p is not None), None)
    return get


def full_pages(spans: dict[str, tuple[list[int], int]], block_size: int) -> dict[str, list[int]]:
    """Each request's blocks that its computed tokens fill."""
    return {rid: list(blocks[: n // block_size]) for rid, (blocks, n) in spans.items()}


class LocalKv:
    """The fresh pool's layers in this process (one GPU): `boundary.KvPool`'s rows (blocks x row bytes, int8), each
    layer's dtype, and the served edge's K/V file (``near``, `kv_file`) that `readd` loads prefixes from."""

    def __init__(self, rows: list[Any], dtypes: list[Any], block_size: int, near: list[Any] | None,
                 layout: dict[str, Any]) -> None:
        if not all(slot_major(v["shape"], block_size) for v in layout["views"]):
            raise SystemExit("a layer's blocks are not laid out slot by slot: the tail's slots can't be told apart")
        self.rows, self.dtypes, self.bs, self.near = rows, dtypes, block_size, near
        self.slot = int(rows[0].shape[1]) // block_size
        self.flat = [r.view(r.shape[0] * block_size, self.slot) for r in rows]
        self.block_bytes = int(rows[0].shape[1]) * len(rows)
        self.fills: dict[str, list[Any]] = {}

    def load(self, pairs: list[tuple[int, int]], partial: list[tuple[int, int]], tail: str) -> None:
        """Each (served id, fresh id) of ``pairs`` from the edge's file, and past each partial last block's prefix
        (``partial``: fresh id, slots) the served bytes, zeros, or NaN with every other slot of the pool NaN too."""
        import numpy as np
        import torch

        if tail == "nan":
            for r, dt in zip(self.rows, self.dtypes, strict=True):
                r.view(dt).fill_(float("nan"))
        if pairs:
            old_ids = np.array([p[0] for p in pairs])
            new_ids = torch.tensor([p[1] for p in pairs], device=self.rows[0].device)
            for src, dst in zip(self.near, self.rows, strict=True):
                dst.index_copy_(0, new_ids, torch.from_numpy(np.ascontiguousarray(src[old_ids])).to(dst.device))
        for b, j in partial:
            for dst, dt in zip(self.rows, self.dtypes, strict=True):
                if tail == "zero":
                    dst[b, j * self.slot:].zero_()
                elif tail == "nan":
                    dst[b, j * self.slot:].view(dt).fill_(float("nan"))
        torch.cuda.synchronize()

    def _fill(self, slots: list[int], mode: str) -> None:
        import torch

        if mode not in self.fills:
            v = 0.0 if mode == "zero" else float("nan")
            self.fills[mode] = [torch.full((self.slot // dt.itemsize,), v, dtype=dt, device=r.device).view(torch.int8)
                                for r, dt in zip(self.rows, self.dtypes, strict=True)]
        idx = torch.tensor(slots, device=self.flat[0].device)
        for f, fill in zip(self.flat, self.fills[mode], strict=True):
            f.index_copy_(0, idx, fill.expand(idx.numel(), -1).contiguous())

    def fill_blocks(self, ids: list[int], mode: str) -> None:
        self._fill([b * self.bs + j for b in ids for j in range(self.bs)], mode)

    def fill_slots(self, slots: list[int], mode: str) -> None:
        import torch

        torch.cuda.synchronize()
        self._fill(slots, mode)

    def write(self, ops: dict[int, Any], mode: str) -> None:
        """`Hits`' writes: into each block, each layer's row, or `FILL_PAGE`."""
        import torch

        for b, src in ops.items():
            if not isinstance(src, str):
                for row, dst in zip(src, self.rows, strict=True):
                    dst[b].copy_(torch.from_numpy(row).to(dst.device))
        fills = [b for b, src in ops.items() if isinstance(src, str)]
        if fills:
            self.fill_blocks(fills, mode)

    def roots(self, spans: dict[str, tuple[list[int], int]]) -> dict[str, str]:
        return roots(self.rows, spans, self.bs)


#: the byte a tensor-parallel engine's dummy K/V holds in every byte: zero, or NaN in bf16, e4m3 and fp32 alike
FILL_BYTE = {"zero": 0, "nan": T.FILL}


class RankKv:
    """The fresh pool of a tensor-parallel engine: every rank's layers through ``rpc`` (``collective_rpc``, `tp_state`),
    loaded from the served edges' dumps, ``near`` the first edge's. A dump holds whole blocks, so a re-added request's
    last partial block comes as served (``--tail copy``; a pool that zeroes new blocks, as one of mixed precision does,
    holds zeros past a prefix), a dummy's written slots are refilled block by block (the rest of a dummy's block holds
    the fill or nothing written yet), and a request's root covers the blocks its computed tokens fill (`full_pages`).
    A blocking call waits for every step in flight first, so a write lands between two steps, as `LocalKv`'s does."""

    def __init__(self, rpc: Any, block_size: int, near: Path, layers: list[dict[str, Any]]) -> None:
        self.rpc, self.bs, self.near = rpc, block_size, near
        self.block_bytes = sum(int(x["block_bytes"]) for x in layers)
        self.loaded_from: dict[str, int] = {}
        self.ranks_disagree: list[dict[str, Any]] = []

    def _copy(self, edge_dir: str, pairs: list[list[int]]) -> None:
        self.rpc(T.copy_blocks, args=(edge_dir, pairs))
        self.loaded_from[edge_dir] = self.loaded_from.get(edge_dir, 0) + len(pairs)

    def load(self, pairs: list[tuple[int, int]], partial: list[tuple[int, int]], tail: str) -> None:
        if tail != "copy":
            raise SystemExit("a tensor-parallel edge's dump holds whole blocks: --tail copy")
        if pairs:
            self._copy(str(self.near), [[int(a), int(b)] for a, b in pairs])

    def fill_blocks(self, ids: Iterable[int], mode: str) -> None:
        ids = sorted({int(b) for b in ids})
        if ids:
            self.rpc(T.fill_blocks, args=(ids, FILL_BYTE[mode]))

    def fill_slots(self, slots: list[int], mode: str) -> None:
        self.fill_blocks((s // self.bs for s in slots), mode)

    def write(self, ops: dict[int, Any], mode: str) -> None:
        """`Hits`' writes: into each block, a dump's block (its edge's directory, the served id), or `FILL_PAGE`."""
        loads: dict[str, list[list[int]]] = {}
        for b, src in ops.items():
            if not isinstance(src, str):
                loads.setdefault(src[0], []).append([int(src[1]), int(b)])
        for d, pairs in loads.items():
            self._copy(d, pairs)
        self.fill_blocks((b for b, src in ops.items() if isinstance(src, str)), mode)

    def roots(self, spans: dict[str, tuple[list[int], int]]) -> dict[str, str]:
        """Rank 0's root of each request's full pages; a rank whose roots differ is noted (``ranks_disagree``)."""
        got = sorted(self.rpc(T.block_roots, args=(full_pages(spans, self.bs),)), key=lambda r: r["rank"])
        bad = [g["rank"] for g in got[1:] if g["roots"] != got[0]["roots"]]
        if bad:
            self.ranks_disagree.append({"requests": len(spans), "ranks": bad})
        return got[0]["roots"]


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


def page_chain(tokens: list[int], bs: int) -> list[bytes]:
    """Each page's chain hash (SHA-512 over the previous page's and this page's tokens), the last one partial or not."""
    out, h = [], b""
    for i in range(0, len(tokens), bs):
        h = hashlib.sha512(h + json.dumps(tokens[i:i + bs]).encode()).digest()
        out.append(h)
    return out


def substitute(tokens: list[int], bs: int, keep: set[bytes], ids: tuple[int, int] = SUBSTITUTE_IDS) -> tuple[list[int], int]:
    """A dummy's tokens: each full page whose chain hash is in ``keep`` as it is, every other page replaced by tokens drawn
    from its chain hash, so two requests' pages are equal here exactly where they were; and how many pages were kept."""
    import random

    out, kept = [], 0
    for i, h in enumerate(page_chain(tokens, bs)):
        page = tokens[i * bs:(i + 1) * bs]
        if len(page) == bs and h in keep:
            out += page
            kept += 1
        else:
            rng = random.Random(hashlib.sha512(b"dummy" + h).digest())
            out += [rng.randrange(*ids) for _ in page]
    return out, kept


class Dummies:
    """Shape-only replay's batch-mates: every request but ``target``. `tokens` substitutes a dummy's tokens, `params` forces
    its finish (``ignore_eos``, ``max_tokens`` its served output count, ``finish``), `poison` fills blocks with
    ``mode`` (zero or NaN, in each layer's dtype) through ``kv`` (`LocalKv`, `RankKv`), and `scheduled` and `flush`
    refill each slot a dummy wrote once the GPU has finished the step, as `Mirror` copies them, never in a block the
    target holds. `scheduled` also notes each page the target has written (``target_pages``)."""

    def __init__(self, target: str, mode: str, sched: Any, kvm: Any, kv: Any, block_size: int,
                 keep_tokens: list[int], finish: dict[str, int], stops: dict[str, int] | None = None) -> None:
        self.target, self.mode, self.sched, self.kvm, self.bs = target, mode, sched, kvm, block_size
        self.kv, self.finish = kv, finish
        self.stops = {rid: s for rid, s in (stops or {}).items() if rid != target}
        full = len(keep_tokens) // block_size * block_size
        self.keep = set(page_chain(keep_tokens[:full], block_size))
        self.target_pages: set[bytes] = set()
        self.pending: list[int] = []
        self.taken = {"dummies": 0, "tokens_substituted": 0, "pages_kept_target_shared": 0, "finishes_forced": 0,
                      "blocks_poisoned_at_start": 0, "pages_seeded_poisoned": 0, "slots_poisoned": 0,
                      "target_pages_loaded": 0, "target_pages_overwritten": 0, "target_pages_own": 0,
                      "target_pages_unavailable": 0, "stops_forced": 0}

    def dummy(self, rid: str) -> bool:
        return rid != self.target

    def tokens(self, prompt: list[int], outs: list[int]) -> tuple[list[int], list[int]]:
        got, kept = substitute(list(prompt) + list(outs), self.bs, self.keep)
        self.taken["dummies"] += 1
        self.taken["tokens_substituted"] += len(got) - kept * self.bs
        self.taken["pages_kept_target_shared"] += kept
        return got[:len(prompt)], got[len(prompt):]

    def params(self, rid: str, r: dict) -> dict:
        """``r`` (sampling and max_tokens) for a dummy: it ignores EOS and its stop tokens and stops at its served output
        count, or, if serving stopped it on a token (``stops``), one past it, so the scheduler gives it the next step's
        row as serving's did, and `stopper` finishes it where serving did."""
        s = {k: v for k, v in r["sampling"].items() if k != "stop_token_ids"} | {"ignore_eos": True}
        n = self.finish.get(rid)
        if n is not None:
            self.taken["finishes_forced"] += 1
            if rid in self.stops:
                n += 1
        return {"sampling": s, "max_tokens": n if n is not None else r.get("max_tokens")}

    def stopper(self, trace: Any) -> Callable[[Any], Any]:
        """A wrapper of the scheduler's ``update_from_output`` that finishes each dummy in ``stops`` once the step that
        stopped it in serving has its output, as serving's scheduler finished it on reading the stop token, and lists it
        in that step's record as finished there. Under async scheduling that scheduler had already given it a row in the
        next step, which a length the scheduler can see ahead would not."""
        def update_from_output(orig: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(so: Any, mro: Any, *a: Any, **kw: Any) -> Any:
                step = trace._pending.get(id(so))
                got = orig(so, mro, *a, **kw)
                due = [rid for rid, s in self.stops.items() if s == step and rid in self.sched.requests]
                if due:
                    from vllm.v1.request import RequestStatus

                    self.sched.finish_requests(due, RequestStatus.FINISHED_STOPPED)
                    self.taken["stops_forced"] += len(due)
                    rec = next((r for r in reversed(trace.records) if r.get("kind") == "step" and r.get("step") == step), None)
                    if rec is not None:
                        rec["finished"].update({rid: {"reason": "stop", "stop_reason": None} for rid in due})
                return got
            return wrapped
        return update_from_output

    def target_blocks(self) -> set[int]:
        return set(self.kvm.get_block_ids(self.target)[0]) if self.target in self.sched.requests else set()

    def poison(self, blocks: Iterable[int]) -> int:
        ids = sorted(set(blocks) - self.target_blocks())
        if ids:
            self.kv.fill_blocks(ids, self.mode)
        return len(ids)

    def scheduled(self, so: Any) -> None:
        from verity_vllm.engine.step_trace import step_geometry

        start, _, _, _ = step_geometry(so)
        mine = self.target_blocks()
        bs = self.bs
        for rid, n in so.num_scheduled_tokens.items():
            if not self.dummy(rid):
                q = self.sched.requests[rid]
                self.target_pages.update(q.block_hashes[: (start[rid] + int(n)) // bs])
                continue
            blocks = self.kvm.get_block_ids(rid)[0]
            self.pending += [blocks[p // bs] * bs + p % bs for p in range(start[rid], start[rid] + int(n))
                             if blocks[p // bs] not in mine]

    def flush(self) -> None:
        if not self.pending:
            return
        self.kv.fill_slots(self.pending, self.mode)
        self.taken["slots_poisoned"] += len(self.pending)
        self.pending.clear()


PERTURB_ID = "shape-perturb-0"


def perturb_row(n: int, first: int, arrivals: list[dict]) -> dict[str, Any]:
    """The perturbed control's one dummy row more than serving ran: ``n`` prompt tokens from the first step on, sampled
    as the block's first arrival was and never stopping early, so that only the batch's shape changes (a greedy row
    would take a batch of unseeded rows off FlashInfer's sampler)."""
    sampling = dict(arrivals[0]["sampling"]) if arrivals else {}
    return {"engine_id": PERTURB_ID, "before_step": first, "prompt_ids": [0] * n, "max_tokens": 2048,
            "sampling": sampling | {"ignore_eos": True}}


def jump_queue(sched: Any, rid: str) -> None:
    """``rid`` to the head of the scheduler's waiting queue. An arrival under load waits behind every request queued
    before it, and the perturbed control's row has to run while the target does, or the batch's shape never changes."""
    q = sched.requests[rid]
    sched.waiting.remove_request(q)
    sched.waiting.prepend_request(q)


def served_finish(st_requests: dict[str, int], outputs: dict[int, dict], steps: dict[int, dict]) -> dict[str, int]:
    """Each request the record finishes inside the block: its served output count (``st_requests``, the outputs it had
    at the start, plus every step's since)."""
    n = dict(st_requests)
    for s in sorted(outputs):
        for rid, v in outputs[s].items():
            n[rid] = n.get(rid, 0) + len(v[0])
    done = {rid for rec in steps.values() for rid in (rec.get("finished") or {})}
    return {rid: n.get(rid, 0) for rid in done}


def served_stops(steps: dict[int, dict]) -> dict[str, int]:
    """Each request the record finishes inside the block on anything but its length (a stop token, an abort), which the
    scheduler learns only from a step's output: the step whose output finished it."""
    return {rid: s for s, rec in sorted(steps.items()) for rid, v in (rec.get("finished") or {}).items()
            if (v or {}).get("reason") != "length"}


def served_seeds(st: dict[str, Any]) -> dict[str, int]:
    """Each request the served runner held at the edge: its sampling seed (Model Runner V2's, drawn from numpy's global
    generator when the request reached the runner, unless the request set one)."""
    leaves = (st.get("runner") or {}).get("leaves", {})
    if "sampler.sampling_states.seeds" not in leaves:
        raise SystemExit("the edge's state holds no Model Runner V2 seeds: serve with VLLM_USE_V2_MODEL_RUNNER=1")
    _, index = leaves["req_states.req_id_to_index"]
    _, seeds = leaves["sampler.sampling_states.seeds"]
    return {rid: int(seeds[i]) for rid, i in index.items()}


def carry_rng(runner: Any, st: dict[str, Any] | None = None, start: dict[str, Any] | None = None) -> dict[str, Any]:
    """The served generators put back and each re-added request's served seed given to it when the runner adds it
    (`tp_state.SeedHook`), from an edge's state (``st``) or a mirror's start (``start``, `start_state`'s ``rng``): a
    re-added unseeded request would draw a new seed from numpy's generator, and shift every later request's."""
    if start is not None:
        if "rng" not in start:
            raise SystemExit("the mirror's start state holds no generators: take it again (--at-step --mirror-out)")
        T.set_rng(runner, start["rng"], seeds=True)
        return {"from": "start", "seeds": len(start["rng"].get("seeds") or {})}
    seeds = served_seeds(st)
    B.set_rng(st["rng"])
    runner._verity_seed_hook = T.SeedHook(runner, seeds)
    return {"from": "edge", "seeds": len(seeds)}


def keep_preempted(q: Any, status: str) -> None:
    """A re-added request the served engine had preempted keeps that status, so the fresh scheduler resumes it as serving
    did: Model Runner V2 adds a step's resumed requests after its new ones, and an unseeded request draws its seed when
    the runner adds it."""
    if status == "PREEMPTED":
        q.status = type(q.status).PREEMPTED


def readd(eng: Any, core: Any, st: dict[str, Any], arrived: dict[str, dict], kv: Any, block_size: int,
          tail: str, shape: Dummies | None = None, params: Any = R.params) -> dict[str, Any]:
    """Every request ``st`` (the served edge's state) holds, added to the fresh engine as `rebuild`'s docstring says,
    its pages loaded through ``kv`` (`LocalKv`, `RankKv`); what it loaded."""
    sched = core.scheduler
    kvm = sched.kv_cache_manager
    pool = kvm.block_pool
    zeroing = bool(getattr(sched, "needs_kv_cache_zeroing", False))
    if len(st["managers"]) != 1:
        raise SystemExit(f"{len(st['managers'])} KV cache groups: the rebuild copies one")
    reqs = {r.request_id: r for r in B.unpickle_requests(st["requests"], {"block_hasher": None})}
    s = st["scheduler"]
    order = [x.key for q in ("running", "waiting", "skipped_waiting") for x in (s.get(q) or [])]
    if sorted(order) != sorted(reqs):
        raise SystemExit(f"the served queues hold {len(order)} requests, the scheduler {len(reqs)}")
    served_blocks = {rid: [b.key for b in ids] for rid, ids in st["managers"][0]["req_to_blocks"].items()}
    t = time.monotonic()
    pairs: list[tuple[int, int]] = []
    partial: list[tuple[int, int]] = []
    statuses: dict[str, int] = {}
    fresh_of: dict[int, int] = {}
    shared = 0
    allocated: set[int] = set()
    for rid in order:
        r = reqs[rid]
        statuses[r.status.name] = statuses.get(r.status.name, 0) + 1
        a = {"sampling": arrived[rid]["sampling"], "max_tokens": arrived[rid]["sampling"].get("max_tokens")}
        prompt, outs = list(r.prompt_token_ids), list(r.output_token_ids)
        mate = shape is not None and shape.dummy(rid)
        if mate:
            prompt, outs = shape.tokens(prompt, outs)
            a = shape.params(rid, a)
        eng.add_request(rid, {"prompt_token_ids": prompt}, params(a))
        q = sched.requests[rid]
        keep_preempted(q, r.status.name)
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
        allocated.update(new)
        if mate:
            continue
        # a shape-only target's leading blocks may be shared with a dummy re-added before it, which loaded nothing
        lo = 0 if shape is not None else j
        pairs += zip(old[lo:], new[lo:], strict=True)
        if n % block_size:
            partial.append((new[-1], n % block_size))
    alloc_s = time.monotonic() - t
    t = time.monotonic()
    kv.load(pairs, partial, tail)
    if zeroing:
        # the workers zero a step's newly allocated blocks before it runs, but those a KV connector loads: these too
        sched._skip_zero_block_ids.update(allocated)
    if shape is not None:
        shape.taken["blocks_poisoned_at_start"] += shape.poison(allocated)
        shape.taken["target_prefix_blocks_loaded"] = len(pairs)
    return {"requests": len(order), "statuses": statuses, "with_prefix": len({p for p in order if reqs[p].num_computed_tokens > 0}),
            "blocks_loaded": len(pairs), "blocks_shared": shared, "blocks_served": len({b for v in served_blocks.values() for b in v}),
            "blocks_fresh": pool.num_gpu_blocks - 1 - pool.get_num_free_blocks(), "bytes_loaded": len(pairs) * kv.block_bytes,
            "partial_blocks": len(partial), "alloc_s": alloc_s, "load_s": time.monotonic() - t, "tail": tail,
            "zeroing": zeroing, "order": order}


def readd_mirror(eng: Any, core: Any, start: dict[str, Any], mirror: Any, served_outs: dict[int, dict],
                 layout: dict[str, Any], block_size: int, tail: str, shape: Dummies | None = None) -> dict[str, Any]:
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
    allocated: set[int] = set()
    for rid in start["order"]:
        if rid in start["finished_before"]:
            continue
        r = start["requests"][rid]
        order.append(rid)
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        a = {"sampling": r["sampling"], "max_tokens": r["sampling"].get("max_tokens")}
        prompt = list(r["prompt"])
        outs = list(r["outputs"]) + [x for s in range(done + 1, S) for x in (served_outs.get(s, {}).get(rid) or [[]])[0]]
        mate = shape is not None and shape.dummy(rid)
        if mate:
            prompt, outs = shape.tokens(prompt, outs)
            a = shape.params(rid, a)
        eng.add_request(rid, {"prompt_token_ids": prompt}, R.params(a))
        q = sched.requests[rid]
        keep_preempted(q, r["status"])
        if outs:
            q.append_output_token_ids(outs)
        n = min(r["computed"], q.num_tokens - 1)
        if n <= 0:
            continue
        k = -(-n // bs)
        new, j = alloc_prefix(kvm, pool, q, n, start["blocks"][rid][:k], fresh_of, bs)
        shared += j
        allocated.update(new)
        if mate:
            continue
        lo = 0 if shape is not None else j
        x = np.zeros((k * bs, L, slot), np.int8)
        x[:n] = mirror.rows(rid, n)
        chunks.append(x.reshape(k, bs, L, slot).transpose(2, 0, 1, 3).reshape(L, k, bs * slot)[:, lo:])
        ids += new[lo:]
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
    if shape is not None:
        shape.taken["blocks_poisoned_at_start"] += shape.poison(allocated)
        shape.taken["target_prefix_blocks_loaded"] = len(ids)
    torch.cuda.synchronize()
    served = {b for rid in order for b in start["blocks"].get(rid, ())}
    # the served engine frees a request that finished at S-1 only once S-1's output is processed, after it schedules S:
    # until then the fresh pool holds as many blocks as those requests held alone
    gone = {b for rid in start["finished_before"] for b in start["blocks"].get(rid, ())} - served
    held = pool.get_new_blocks(len(gone)) if gone else []
    return {"requests": len(order), "statuses": statuses, "skipped_finished": len(start["finished_before"]),
            "blocks_held_through_first_schedule": held,
            "blocks_loaded": len(ids), "blocks_shared": shared, "blocks_served": len(served) + len(gone),
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
                done: int, step: int, block_size: int, runner: Any) -> dict[str, Any]:
    """What lifetime replay starts step ``step`` from, taken right before the served-identical engine schedules it (every
    step before has finished on the GPU and the mirror holds its slots): the scheduler's queues in order, each request's
    prompt, processed outputs, computed tokens (step S-1's included), status and sampling, the last step whose output was
    processed, the requests the record finishes after it and before S, each request's K/V root over its computed tokens
    on the GPU and in the mirror, and the generators with each request's seed (`tp_state.rng_state`)."""
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
            "partial_last_pages": sum(1 for _, n in spans.values() if n % block_size), "rng": T.rng_state(runner)}


def split_outputs(diff_of: Any, served: dict[int, dict], rebuilt: dict[int, dict], readded: set[str]) -> dict[str, Any]:
    """`boundary.outputs_diff` over the re-added requests (the batch-mates) and over the rest (the requests whose lifetime
    starts inside the block, as a sampled request's does) apart."""
    def only(outs: dict[int, dict], keep: bool) -> dict[int, dict]:
        return {s: {r: v for r, v in o.items() if (r in readded) == keep} for s, o in outs.items()}
    return {"readded": diff_of(only(served, True), only(rebuilt, True)),
            "arrived": diff_of(only(served, False), only(rebuilt, False))}


def dummies(a: argparse.Namespace, w: dict[str, Any], st0: dict[str, Any], begin: dict[str, Any] | None, first: int,
            sched: Any, kvm: Any, kv: Any, bs: int) -> tuple[Dummies, int]:
    """``--shape-only``'s `Dummies`, from the record: the target's tokens (its prompt and every output it has by the
    block's end, whose pages a dummy keeps where it shares them), each request's output count at the start, and the
    served count of each request the record finishes; and the last step with an output of the target."""
    rid = a.shape_only
    if a.hits != "forced":
        raise SystemExit("--shape-only takes the prefix-cache hits from the record: --hits forced")
    outs = {s: v for s, v in w["outputs"].items() if s >= first}
    if begin:
        done = begin["processed_through"]
        held = {r: v["prompt"] + v["outputs"] + [x for s in range(done + 1, first) for x in (w["outputs"].get(s, {}).get(r) or [[]])[0]]
                for r, v in begin["requests"].items() if r not in begin["finished_before"]}
        base = {r: len(t) - len(begin["requests"][r]["prompt"]) for r, t in held.items()}
    else:
        reqs = B.unpickle_requests(st0["requests"], {"block_hasher": None})
        held = {r.request_id: list(r.prompt_token_ids) + list(r.output_token_ids) for r in reqs}
        base = {r.request_id: len(r.output_token_ids) for r in reqs}
    arrival = next((r for r in w["arrive"] if r["engine_id"] == rid and r["before_step"] >= first), None)
    if rid in held:
        mine = held[rid]
    elif arrival is not None:
        mine = list(arrival["prompt_ids"])
    else:
        raise SystemExit(f"{rid}: neither held at step {first} nor arriving inside the block")
    mine += [x for s in sorted(outs) for x in (outs[s].get(rid) or [[]])[0]]
    r_end = max((s for s, o in outs.items() if rid in o), default=None)
    if r_end is None:
        raise SystemExit(f"{rid}: the record has no output of it from step {first}")
    block = {s: v for s, v in w["steps"].items() if s >= first}
    return Dummies(rid, a.dummy, sched, kvm, kv, bs, mine, served_finish(base, outs, block), served_stops(block)), r_end


def shape_view(steps: dict[int, dict], target: str) -> dict[int, dict]:
    """The steps with each dummy's finish reason dropped (forced to its length, where serving's may be EOS)."""
    return {s: r | {"finished": {k: (v if k == target else None) for k, v in (r.get("finished") or {}).items()}}
            for s, r in steps.items()}


def shape_report(target: str, shape: Dummies, w: dict[str, Any], steps: dict[int, dict], outs: dict[int, dict], first: int,
                 r_end: int, readded: set[str], hits: Hits) -> dict[str, Any]:
    """``--shape-only``'s verdict: the target's tokens and logprob bits against serving's, the schedule (rows, new,
    resumed, cache hits, preemptions, emitted counts, finishes) up to its last step, and what the replay took."""
    def only(o: dict[int, dict]) -> dict[int, dict]:
        return {s: {target: v[target]} for s, v in o.items() if target in v and first <= s <= r_end}
    want = {s: v for s, v in w["steps"].items() if first <= s <= r_end}
    got = {s: v for s, v in steps.items() if s <= r_end}
    life = [s for s in sorted(want) if any(r[0] == target for r in want[s]["rows"])]
    t0, t1 = (life[0], life[-1]) if life else (first, r_end)
    lived = [want[s] for s in range(t0, t1 + 1) if s in want]
    target_out = B.outputs_diff(only(w["outputs"]), only(outs))
    sched_diff = step_diff(shape_view(want, target), shape_view(got, target), first, readded)
    taken = shape.taken | {"forced_hits": hits.report(), "arrivals_in_block": sum(1 for r in w["arrive"] if r["before_step"] >= first),
                           "requests_at_start": len(readded)}
    return {"shape": {
        "target": target, "first_step": first, "target_steps": [t0, t1], "target_rows": len(life),
        "target_outputs": target_out,
        "schedule": sched_diff,
        "during_target": {"steps": len(lived), "steps_with_preemption": sum(1 for r in lived if r.get("preempted")),
                          "preemptions": sum(len(r.get("preempted") or ()) for r in lived),
                          "target_preempted": [s for s in range(t0, t1 + 1) if target in ((want.get(s) or {}).get("preempted") or ())],
                          "steps_with_hits": sum(1 for r in lived if r.get("cache_hit")),
                          "target_hits": {s: want[s]["cache_hit"][target] for s in range(t0, t1 + 1)
                                          if target in ((want.get(s) or {}).get("cache_hit") or {})},
                          "max_rows": max((len(r["rows"]) for r in lived), default=0)},
        "taken": taken},
        "bit_exact": (target_out["first"] is None and target_out["tokens"] > 0 and sched_diff["steps_differing"] == 0
                      and shape.taken["target_pages_unavailable"] == 0)}


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
    dtypes = [getattr(torch, v["dtype"].split(".")[-1]) for v in st0["kv_layout"]["views"]]
    lk = LocalKv(kv.rows, dtypes, bs, near_file, st0["kv_layout"])
    pages = edge_pages(near_file, st0["pool"])
    # forced hits read pages written inside the block from the mirror, as lifetime replay reads its host copy, where the
    # fresh pool evicted a page the served one still had cached
    mirror = Mirror(sched, kvm, kv.rows, bs) if a.mirror_out or a.hits == "forced" else None
    shape, own, target, r_end = None, None, a.shape_only, None
    if target:
        shape, r_end = dummies(a, w, st0, begin, first, sched, kvm, lk, bs)
        # the target's pages come from serving's bytes, or are its own: a page a dummy wrote here never stands in
        own = lambda h: mirror.page(h) if mirror.hashes.get(h, ("",))[0] == target else None  # noqa: E731
    t = time.monotonic()
    if begin:
        saved = Saved(a.from_mirror / "mirror")
        pages = first_of(pages, saved.page, *([mirror.page] if mirror and not shape else []))
        out["rebuilt"] = readd_mirror(eng, core, begin, saved, w["outputs"], st0["kv_layout"], bs, a.tail, shape)
        readded = set(out["rebuilt"].pop("order"))
        held = out["rebuilt"].pop("blocks_held_through_first_schedule")
        out["rebuilt"]["blocks_held_through_first_schedule"] = len(held)
        want0 = {r: v for r, v in begin["truth"].items() if r in readded}
        out["start"] = {k: begin[k] for k in ("step", "processed_through", "finished_before", "mirror_vs_gpu", "preempted_waiting",
                                              "partial_last_pages")}
        out["rng"] = carry_rng(runner, start=begin)
    else:
        if mirror and not shape:
            pages = first_of(pages, mirror.page)
        out["rebuilt"] = readd(eng, core, st0, arrived, lk, bs, a.tail, shape)
        readded = set(out["rebuilt"].pop("order"))
        held = []
        want0 = roots(near_file, served_spans(st0), bs)
        out["rng"] = carry_rng(runner, st=st0)
    trace.scheduled = first
    got0 = roots(kv.rows, live_spans(sched, kvm), bs)
    if shape:
        want0, got0 = ({k: v for k, v in x.items() if k == target} for x in (want0, got0))
    out["near_edge"] = roots_diff(want0, got0) | {"check_s": time.monotonic() - t}
    hits = Hits(kvm, lk, pages, recorded_starts(w["steps"], first), bs, lambda: trace.scheduled,
                shape, own) if a.hits == "forced" else None

    busy = B.Busy(lambda: torch.cuda.Event(enable_timing=True), first=first)
    runner._verity_busy = busy
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    B.busy_runner(GPUModelRunner, lambda r: getattr(r, "_verity_busy", None))
    outs: dict[int, dict] = {}
    hooks = Hooks(tag="serve-tax-rebuild")
    hooks.wrap(sched, "update_from_output", step_outputs_hook(trace, lambda rec: outs.__setitem__(rec["step"], rec["out"]),
                                                              lambda: True))
    if shape:
        hooks.wrap(sched, "update_from_output", shape.stopper(trace))
    if hits:
        hooks.wrap(kvm, "get_computed_blocks", hits.lookup)
    pending = sorted((r for r in w["arrive"] if r["before_step"] >= first), key=lambda r: r["before_step"])
    if shape and a.perturb:
        pending.insert(0, perturb_row(a.perturb, first, w["arrive"]))

    def add_due() -> None:
        while pending and pending[0]["before_step"] == trace.scheduled:
            r = pending.pop(0)
            prompt = r["prompt_ids"]
            if shape and shape.dummy(r["engine_id"]):
                prompt, _ = shape.tokens(prompt, [])
                r = r | shape.params(r["engine_id"], r)
            eng.add_request(r["engine_id"], {"prompt_token_ids": prompt}, R.params(r))
            if r["engine_id"] == PERTURB_ID:
                jump_queue(sched, PERTURB_ID)
        if pending and pending[0]["before_step"] < trace.scheduled:
            raise SystemExit(f"the replay passed step {pending[0]['before_step']} before {pending[0]['engine_id']} arrived")

    stop: dict[str, Any] = {}

    def schedule(orig):
        def wrapped(*args, **kw):
            if mirror:
                mirror.flush()
                if a.mirror_out and trace.scheduled == a.at_step:
                    stop["start"] = start_state(sched, kvm, kv, mirror, arrived, w["steps"], max(outs, default=first - 1),
                                                a.at_step, bs, runner)
                    raise _Stop(f"stopped right before step {a.at_step}, with its start state taken")
            if shape:
                shape.flush()
            add_due()
            if hits:
                hits.due(sched.requests, trace.scheduled)
            so = orig(*args, **kw)
            if held:
                pool.free_blocks(held)
                held.clear()
            if mirror and so.total_num_scheduled_tokens:
                mirror.scheduled(so)
            if shape and so.total_num_scheduled_tokens:
                shape.scheduled(so)
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
            if shape and outs and max(outs) >= r_end:
                break
    except _Stop as e:
        out["stopped"] = str(e)
    steps = {r["step"]: r for r in trace.records if r["kind"] == "step" and first <= r["step"] < w["hi"]}
    block = {"from_edge": a.from_edge, "to_edge": a.to_edge, "from_step": w["lo"], "to_step": w["hi"], "first_step": first,
             "arrivals": len(w["arrive"]), "requests_at_start": len(readded)}
    if shape:
        hooks.uninstall()
        out |= shape_report(target, shape, w, steps, outs, first, r_end, readded, hits)
        out |= {"replay_wall_s": time.monotonic() - start, "replay_block": block, "dummy": a.dummy}
        return out
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


# ---------------------------------------------------------------------------------------- a tensor-parallel engine

def served_tp(served: Path, a: int, b: int) -> dict[str, Any]:
    """`restore.served` for an arm `tp_restore.py serve` wrote (its OUT, ``served``): edges ``a`` and ``b`` (from 1, over
    meta.json's), each a directory with edge.json, the scheduler's state (state.pkl, `boundary.save_edge`) and, at the
    first ``--dump`` edges, the pool's blocks; the step trace between them, the arrivals the replay adds, and the served
    steps' outputs."""
    import tp_restore as TR

    meta = json.loads((served / "meta.json").read_text())
    if len(meta.get("edges") or ()) < max(a, b):
        raise SystemExit(f"{served}: {len(meta.get('edges') or ())} edge(s), not {a} and {b}")
    dirs = {k: Path(meta["edges"][k - 1]["dir"]) for k in (a, b)}
    recs = {k: json.loads((d / "edge.json").read_text()) for k, d in dirs.items()}
    for k, d in dirs.items():
        if not (d / "state.pkl").is_file():
            raise SystemExit(f"{d}: no state.pkl: serve with SAVER=boundary:save_edge")
    if not recs[a]["dump"]:
        raise SystemExit(f"edge {a} kept no blocks: serve with --dump {b}")
    outputs = TR.read_outputs(served / "edges" / "outputs.jsonl", 0, recs[b]["trace_step"])
    if not outputs:
        raise SystemExit(f"{served}: no step outputs recorded under edges/: nothing to compare the replay to")
    st = {k: pickle.loads((d / "state.pkl").read_bytes()) for k, d in dirs.items()}
    lo, hi = recs[a]["trace_step"], recs[b]["trace_step"]
    tr = TR.served_trace(served)
    client = {c["id"]: c for c in (json.loads(x) for x in (served / "load" / "client.jsonl").read_text().splitlines() if x)}
    plan = {r["i"]: r for r in (json.loads(x) for x in Path(meta["plan"]).read_text().splitlines() if x.strip())}
    held = set(recs[a]["requests"])
    return {"meta": meta, "dirs": dirs, "recs": recs, "state": st, "lo": lo, "hi": hi, "header": tr[0],
            "arrive": [r for r in TR.arrivals(tr, plan, client, lo, hi) if r["engine_id"] not in held],
            "arrived": {r["request"]: r for r in tr if r["kind"] == "arrive"},
            "steps": {r["step"]: r for r in tr if r["kind"] == "step" and lo <= r["step"] < hi},
            "outputs": {s: o for s, o in outputs.items() if lo <= s < hi},
            "rngs": {k: {r["rank"]: r["rng"] for r in recs[k]["ranks"]} for k in (a, b)}}


def served_pins(served: Path) -> dict[str, str]:
    """The pins the served engine ran under (its meta.json's), set in this process before anything loads vLLM: Inductor
    reads its own once, when vLLM first imports it, and at one GPU the model runs here."""
    pins = json.loads((served / "meta.json").read_text()).get("pins") or {}
    if pins.get("PYTHONHASHSEED", "0") != "0" or sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did: the scheduler iterates sets of request ids")
    if "torch._inductor.config" in sys.modules:
        raise SystemExit("Inductor's config was imported before the served pins were set: set them first (served_pins)")
    os.environ.update({k: v for k, v in pins.items() if k != "PYTHONHASHSEED"})
    return pins


def tp_engine(meta: dict[str, Any], header: dict[str, Any], scratch: Path) -> tuple[Any, dict[str, Any]]:
    """The served engine built again in this process as `tp_restore.restore` builds it: from the served server's argv
    and block count, under its pins, with its FlashInfer autotune results; and how it was built."""
    import tp_restore as TR
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    os.environ.update(TR.PINS | {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", TR.AUTOTUNE_ENV: str(scratch / "autotune"),
                                 "PYTHONPATH": os.pathsep.join(dict.fromkeys([str(HERE), *filter(None, sys.path)]))})
    usage = UsageContext.OPENAI_API_SERVER
    args = TR.engine_args(meta["server_argv"], header["config"])
    vc = args.create_engine_config(usage)
    info = {"autotune": TR.carry_autotune(Path(meta["autotune_dir"]), scratch / "autotune", vc.compute_hash(include_version=False)),
            "log_stats": not args.disable_log_stats}
    eng = LLMEngine(vllm_config=vc, executor_class=Executor.get_class(vc), log_stats=info["log_stats"], usage_context=usage,
                    multiprocess_mode=False)
    return eng, info


def capture(a: argparse.Namespace, takes: list[dict], steps: dict[int, dict], outputs: dict[int, dict], first: int,
            r_end: int) -> dict[str, Any]:
    """``--capture``: the target's rows from every rank's `take`, held to rank 0's (`sampler_capture.spans`), written to
    OUT/capture, and whether its ids are the tokens serving emitted for it from ``first`` to ``r_end``."""
    from verity_vllm.engine import sampler_capture as SC

    t = time.monotonic()
    span = SC.spans(takes, [steps[s] for s in sorted(steps)]).get(a.shape_only)
    if span is None:
        raise SystemExit(f"--capture: no rank's sampler took a row of {a.shape_only}")
    SC.write(a.out / "capture", a.capture, {a.shape_only: span})
    served = [x for s in sorted(outputs) if first <= s <= r_end for x in (outputs[s].get(a.shape_only) or [[]])[0]]
    return {"template": a.capture, "tokens": len(span["tokens"]), "as_served": span["tokens"] == served,
            "program": span["program"], "write_s": time.monotonic() - t}


def run_tp(a: argparse.Namespace) -> dict[str, Any]:
    """`run` for an arm `tp_restore.py serve` wrote (``--served``), on an engine of its tensor-parallel size: from the
    first edge only (no mirror), each rank's pages from the edges' dumps (`RankKv`), the generators and seeds each rank
    served with, and the far edge compared over full pages."""
    import tp_edge as E
    import tp_restore as TR

    sys.path.insert(0, str(ROOT / "integrations/vllm"))
    sys.path.insert(0, str(HERE))
    from servetax_plugin import step_outputs_hook
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    w = served_tp(a.served, a.from_edge, a.to_edge)
    st0, st1 = w["state"][a.from_edge], w["state"][a.to_edge]
    near, far = w["dirs"][a.from_edge], w["dirs"][a.to_edge]
    far_dumped = bool(w["recs"][a.to_edge]["dump"])
    bs = int(w["header"]["config"]["block_size"])
    first, target = w["lo"], a.shape_only
    logprobs = (w["meta"].get("load") or {}).get("logprobs")

    def params(r: dict) -> Any:
        s = dict(r["sampling"]) | ({"max_tokens": r["max_tokens"]} if r.get("max_tokens") is not None else {})
        return TR.sampling_params(s, logprobs)

    t = time.monotonic()
    eng, out = tp_engine(w["meta"], w["header"], a.scratch)
    out["engine_ready_s"] = time.monotonic() - t
    core = eng.engine_core.engine_core
    sched = core.scheduler
    kvm = sched.kv_cache_manager
    trace = StepTrace().install(eng)
    out["config_difference"] = TR.config_difference(w["header"]["config"], trace.records[0]["config"])
    if out["config_difference"]:
        raise SystemExit(f"the fresh engine's scheduler or cache settings differ from the served one's: {out['config_difference']}")
    rpc = core.model_executor.collective_rpc
    if a.capture:
        from verity_vllm.engine import sampler_capture as SC
        rpc(SC.install, args=([target],))
    kv = RankKv(rpc, bs, near, w["recs"][a.from_edge]["layers"])
    # a page written after the first edge comes from the far edge's dump, the bytes serving committed for it
    pages = first_of(dump_pages(near, st0["pool"]), *([dump_pages(far, st1["pool"])] if far_dumped else []))
    shape, own, r_end = None, None, None
    if target:
        shape, r_end = dummies(a, w, st0, None, first, sched, kvm, kv, bs)
        own = lambda h: OWN_PAGE if h in shape.target_pages else None  # noqa: E731
    t = time.monotonic()
    out["rebuilt"] = readd(eng, core, st0, w["arrived"], kv, bs, a.tail, shape, params)
    readded = set(out["rebuilt"].pop("order"))
    out["rng"] = {"ranks": [x["rng_digest"] for x in rpc(T.set_rng_ranked, args=(w["rngs"][a.from_edge], True))]}
    want0 = T.dump_roots(near, full_pages(served_spans(st0), bs))
    got0 = kv.roots(live_spans(sched, kvm))
    if shape:
        want0, got0 = ({k: v for k, v in x.items() if k == target} for x in (want0, got0))
    out["near_edge"] = roots_diff(want0, got0) | {"check_s": time.monotonic() - t, "over": "full pages"}
    trace.scheduled = first
    hits = Hits(kvm, kv, pages, recorded_starts(w["steps"], first), bs, lambda: trace.scheduled,
                shape, own) if a.hits == "forced" else None

    outs: dict[int, dict] = {}
    hooks = Hooks(tag="serve-tax-rebuild")
    hooks.wrap(sched, "update_from_output", step_outputs_hook(trace, lambda rec: outs.__setitem__(rec["step"], rec["out"]),
                                                              lambda: True))
    if shape:
        hooks.wrap(sched, "update_from_output", shape.stopper(trace))
    if hits:
        hooks.wrap(kvm, "get_computed_blocks", hits.lookup)
    pending = sorted(w["arrive"], key=lambda r: r["before_step"])
    if shape and a.perturb:
        pending.insert(0, perturb_row(a.perturb, first, w["arrive"]))

    def add_due() -> None:
        while pending and pending[0]["before_step"] == trace.scheduled:
            r = pending.pop(0)
            prompt = r["prompt_ids"]
            if shape and shape.dummy(r["engine_id"]):
                prompt, _ = shape.tokens(prompt, [])
                r = r | shape.params(r["engine_id"], r)
            eng.add_request(r["engine_id"], {"prompt_token_ids": prompt}, params(r))
            if r["engine_id"] == PERTURB_ID:
                jump_queue(sched, PERTURB_ID)
        if pending and pending[0]["before_step"] < trace.scheduled:
            raise SystemExit(f"the replay passed step {pending[0]['before_step']} before {pending[0]['engine_id']} arrived")

    def schedule(orig):
        def wrapped(*args, **kw):
            if shape:
                shape.flush()
            add_due()
            if hits:
                hits.due(sched.requests, trace.scheduled)
            so = orig(*args, **kw)
            if shape and so.total_num_scheduled_tokens:
                shape.scheduled(so)
            return so
        return wrapped

    hooks.wrap(sched, "schedule", schedule)
    taken: dict[str, Any] = {}

    def far_roots(_core: Any, _d: Path) -> dict[str, Any]:
        taken["roots"] = kv.roots(live_spans(sched, kvm))
        return {"requests": len(taken["roots"])}

    edges = None if shape else E.install(core, E.Edges(a.scratch / "replay-edges", steps=[w["hi"]], dump=0,
                                                         savers=[far_roots], counter=lambda c: trace.scheduled))
    start = time.monotonic()
    while not (edges and edges.taken):
        if shape and outs and max(outs) >= r_end:
            break
        if not (sched.has_requests() or core.batch_queue):
            add_due()
            if not sched.has_requests():
                raise SystemExit(f"the replay is idle at step {trace.scheduled} short of {w['hi']}")
        eng.step()
    out["replay_wall_s"] = time.monotonic() - start
    takes = rpc(SC.take, args=(str(a.scratch / "capture"),)) if a.capture else None
    hooks.uninstall()
    steps = {r["step"]: r for r in trace.records if r["kind"] == "step" and first <= r["step"] < w["hi"]}
    block = {"from_edge": a.from_edge, "to_edge": a.to_edge, "from_step": w["lo"], "to_step": w["hi"], "first_step": first,
             "arrivals": len(w["arrive"]), "requests_at_start": len(readded), "tp": len(w["recs"][a.from_edge]["ranks"])}
    out["pages_from"] = {("near" if d == str(near) else "far" if d == str(far) else d): n for d, n in kv.loaded_from.items()}
    out["ranks_disagree"] = kv.ranks_disagree
    if shape:
        if a.capture:
            out["capture"] = capture(a, takes, steps, w["outputs"], first, r_end)
        return out | shape_report(target, shape, w, steps, outs, first, r_end, readded, hits) | {
            "replay_block": block, "dummy": a.dummy}
    outs = {s: v for s, v in outs.items() if s < w["hi"]}
    t = time.monotonic()
    want1 = T.dump_roots(far, full_pages(served_spans(st1), bs)) if far_dumped else None
    out |= {"schedule": step_diff(w["steps"], steps, first, readded),
            "outputs": B.outputs_diff(w["outputs"], outs),
            "outputs_by_kind": split_outputs(B.outputs_diff, w["outputs"], outs, readded),
            "far_edge": (roots_diff(want1, taken["roots"]) | {"check_s": time.monotonic() - t, "over": "full pages"})
            if want1 is not None else None}
    out["bit_exact"] = (out["near_edge"]["differing"] == 0 and out["near_edge"]["compared"] == out["near_edge"]["requests_served"]
                        and out["schedule"]["steps_differing"] == 0 and out["outputs"]["first"] is None
                        and (want1 is None or out["far_edge"]["roots_equal"]) and not kv.ranks_disagree)
    out["hits"] = hits.report() if hits else {"mode": "engine"}
    with open(a.out / "rebuild_outputs.jsonl", "w") as f:
        for s in sorted(outs):
            f.write(json.dumps({"step": s, "out": outs[s]}, separators=(",", ":")) + "\n")
    with open(a.out / "rebuild_steps.jsonl", "w") as f:
        for s in sorted(steps):
            f.write(json.dumps(steps[s], separators=(",", ":")) + "\n")
    return out | {"replay_block": block}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm-dir", type=Path, help="a serve_tax arm on one GPU (with --kv-dir)")
    ap.add_argument("--from-edge", type=int, default=1)
    ap.add_argument("--to-edge", type=int, default=2)
    ap.add_argument("--kv-dir", type=Path, help="where the served arm kept both edges' K/V files and states")
    ap.add_argument("--served", type=Path, metavar="OUT",
                    help="instead, the OUT of `tp_restore.py serve` (any tensor-parallel size; `run_tp`)")
    ap.add_argument("--scratch", type=Path, metavar="DIR", help="with --served: where the engine's autotune results and "
                    "the far edge's digests go")
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
    ap.add_argument("--shape-only", metavar="RID",
                    help="replay RID with every other request a dummy of its shape (`Dummies`), until its last served "
                         "output; writes shape.json")
    ap.add_argument("--dummy", choices=("zero", "nan"), default="zero",
                    help="what a dummy's K/V holds: zero, or NaN to probe for reads at weight zero")
    ap.add_argument("--perturb", type=int, default=0, metavar="N",
                    help="with --shape-only, a control: add one dummy of N prompt tokens at the first step, a row serving "
                         "never ran")
    ap.add_argument("--capture", metavar="TEMPLATE",
                    help="with --served and --shape-only: take the target's logits rows and ids at every rank's sampler "
                         "into OUT/capture, a span of TEMPLATE's units (a token select's, e.g. TokenSelect_v4{V=154880})")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if bool(a.served) == bool(a.arm_dir and a.kv_dir) or bool(a.served) != bool(a.scratch):
        raise SystemExit("either --arm-dir with --kv-dir (one GPU), or --served with --scratch (tp_restore.py serve's)")
    if a.served and (a.mirror_out or a.from_mirror):
        raise SystemExit("--served replays from its first edge: no mirror")
    if bool(a.at_step is not None) != bool(a.mirror_out) or (a.mirror_out and a.from_mirror):
        raise SystemExit("--at-step and --mirror-out go together, and not with --from-mirror")
    if a.shape_only and a.mirror_out:
        raise SystemExit("--shape-only starts from an edge or --from-mirror, not with --mirror-out")
    if a.capture and not (a.served and a.shape_only):
        raise SystemExit("--capture takes the sampler rows of --served's --shape-only target")
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did: the scheduler iterates sets of request ids")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    if a.served:
        served_pins(a.served)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mirror_out:
        a.mirror_out.mkdir(parents=True, exist_ok=True)
    got = run_tp(a) if a.served else run(a)
    rec = {"schema": "verity-bench/serve-tax/rebuild/v0", "arm_dir": str(a.served or a.arm_dir), "run_id": SHARED.RESEARCH_RUN_ID,
           **got}
    name = "mirror.json" if a.mirror_out else "shape.json" if a.shape_only else "rebuild.json"
    (a.out / name).write_text(json.dumps(rec, indent=1, default=str))
    if a.shape_only:
        sh = rec["shape"]
        print(json.dumps({"bit_exact": rec["bit_exact"], "target": sh["target"], "dummy": a.dummy, "first_step": sh["first_step"],
                          "target_steps": sh["target_steps"], "target_tokens": sh["target_outputs"]["tokens"],
                          "target_first_differing": sh["target_outputs"]["first"],
                          "steps_differing": sh["schedule"]["steps_differing"], "during_target": sh["during_target"],
                          "near_differing": rec["near_edge"]["differing"],
                          "taken": {k: v for k, v in sh["taken"].items() if k != "forced_hits"}}
                         | {k: rec[k] for k in ("pages_from", "ranks_disagree", "capture") if k in rec}, default=str))
        return 0
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
                      "far_roots_equal": (rec["far_edge"] or {}).get("roots_equal"),
                      "far_differing": (rec["far_edge"] or {}).get("differing"),
                      "hits": {k: v for k, v in rec["hits"].items() if k != "short_first"}}
                     | {k: rec[k] for k in ("pages_from", "ranks_disagree") if k in rec}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
