"""`Store`: the node's store as one registry, the reference implementation of `StoreView` and `Lifecycle`.

Regions are numbered into one block index space in registration order.  A KV pool is one page table shared by its
layer regions (a vLLM block id is a page in every layer): ``fill[page]`` slots of an allocated page are written, and
a block of a page is *registered* unless a write reached it and has not completed it.  So:

- ``kv_alloc`` takes pages out of filler and touches no byte: each block stays registered under its last record (the
  filler it was, or the KV it held before the page was freed) until a write reaches it;
- ``kv_append`` makes the blocks a write reaches dirty (their bytes are no registered encoding while they fill) and
  re-registers those it completes, under the step that wrote them; the codec encodes the slots as they are written
  (``encode_slots_`` on the device, `kv_write_reference` here);
- ``kv_free`` returns pages to filler with their bytes as they are: a registered block stays registered, a dirty block
  (a page freed part-written) stays out of every plan until a later write completes it.

A pass's plan is the weights' regions whole, then each KV layer region over the blocks attention reads (the live
pages' blocks that writes completed), then the filler the `filler_plan` policy picks (`idle_blocks` lists the
registered blocks no attention reads: free pages, and allocated pages' blocks no write reached).  Positions run in
that order, and no block is read twice in a pass (a plan that repeats one is refused).  Inside a window each pass is
seeded from the window's challenge and the root before it (`chain`): the session's previous pass, across windows, and
the registration's root before the session's first.  The reference computes every receipt from the bytes
(`receipts`); a device reader hands each segment's root to `end_pass`.

The seed and the plan digest are the remote bandwidth driver's (`remote_bandwidth.receipts`: `seed`, `Plan.digest`
over the plan's canonical runs); the tree over a pass's receipts is this store's (each segment's tree, then theirs).
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity_vllm.store.interface import (KINDS, TIERS, Append, Backing, Codec, Counters, Layout, Opening, PageLayout,
                                         PassCommitment, Receipt, Registration, RegionInfo, Segment, Source, Stale,
                                         WindowRecord)

CHALLENGE_BYTES = 32
SEED_BYTES = 64
KEY_BYTES = 32
REGISTRATION_DOMAIN = b"verity/store/registration/v0\0"
SEED_DOMAIN = b"verity/remote-bandwidth/seed/v0\0"
PLAN_DOMAIN = b"verity/remote-bandwidth/plan/v0\0"
PASS_DOMAIN = b"verity/store/pass/v1\0"
EMPTY_ROOT = hashlib.sha512(b"").digest()

FillerPlan = Callable[["Store", int], Sequence[tuple[str, Sequence[int]]]]


def frame(b: bytes) -> bytes:
    return struct.pack(">Q", len(b)) + b


def chain(challenge: bytes, previous: bytes, index: int) -> bytes:
    """Pass ``index``'s seed in a window: ``SHA512(D ‖ σ ‖ u32be index ‖ root before it)``, the driver's `seed`."""
    if len(challenge) != CHALLENGE_BYTES or len(previous) != 64 or not 0 <= index < 1 << 32:
        raise ValueError("a seed is of a 32-byte challenge, a u32 pass index and a 64-byte root")
    return hashlib.sha512(SEED_DOMAIN + challenge + struct.pack(">I", index) + previous).digest()


def runs_of(blocks: np.ndarray) -> np.ndarray:
    """``(R, 2)`` int64 ``(first, count)``: ``blocks`` (in read order) as runs of consecutive indices."""
    b = np.asarray(blocks, np.int64)
    if not len(b):
        return np.zeros((0, 2), np.int64)
    cut = np.flatnonzero(np.diff(b) != 1) + 1
    starts = np.concatenate([[0], cut])
    return np.stack([b[starts], np.diff(np.concatenate([starts, [len(b)]]))], axis=1)


def canonical(runs: np.ndarray) -> np.ndarray:
    """``runs`` with each run that starts where the one before it ends merged into it (the driver's canonical plan)."""
    if len(runs) < 2:
        return runs
    new = np.concatenate([[True], runs[1:, 0] != runs[:-1, 0] + runs[:-1, 1]])
    at = np.flatnonzero(new)
    return np.stack([runs[at, 0], np.add.reduceat(runs[:, 1], at)], axis=1)


def distinct(runs: np.ndarray) -> bool:
    """No block is in two runs."""
    r = runs[np.argsort(runs[:, 0], kind="stable")]
    return bool((r[:-1, 0] + r[:-1, 1] <= r[1:, 0]).all())


def plan_digest(runs: np.ndarray) -> bytes:
    """The driver's `Plan.digest`: ``SHA512(D ‖ u32be R ‖ (u64be first ‖ u64be count) × R)`` over canonical runs."""
    return hashlib.sha512(PLAN_DOMAIN + struct.pack(">I", len(runs)) + np.asarray(runs, ">u8").tobytes()).digest()


def _leaves(receipt: Receipt, seed: bytes, ys: Sequence[bytes]) -> list[bytes]:
    from verity.core.protocols.bandwidth import tags
    head, per = tags.leaf_head(receipt, seed[:KEY_BYTES]), receipt.per_leaf
    return [tags.leaf_hash(head, c, ys[c * per:(c + 1) * per]) for c in range(-(-len(ys) // per))]


def _tree(leaves: Sequence[bytes]):
    from verity.core.protocols.pous.verifier import MerkleTree
    return MerkleTree(leaves)


def segment_root(receipt: Receipt, seed: bytes, ys: Sequence[bytes]) -> bytes:
    """The RFC 6962 SHA-512 tree over a segment's receipt leaves (the bandwidth audit's `leaf_hash`, ``per_leaf``
    receipts a leaf); an empty segment's root is SHA-512 of nothing."""
    return _tree(_leaves(receipt, seed, ys)).root if ys else EMPTY_ROOT


def pass_root(index: int, seed: bytes, plan_digest: bytes, segments: Sequence[Segment],
              roots: Sequence[bytes]) -> bytes:
    """``SHA512(D ‖ u32be j ‖ seed ‖ plan digest ‖ u32be S ‖ (u64be count) × S ‖ tree(segment roots))``."""
    body = (PASS_DOMAIN + struct.pack(">I", index) + seed + plan_digest + struct.pack(">I", len(segments))
            + b"".join(struct.pack(">Q", s.count) for s in segments))
    return hashlib.sha512(body + (_tree(list(roots)).root if roots else EMPTY_ROOT)).digest()


@dataclass(frozen=True)
class TagReceipt:
    """A bandwidth audit `Tag` (``ro-blake3/v1``, ``ro-sha512/v1``) as a receipt that binds only the block."""

    tag: Any
    binds: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.tag.name

    @property
    def receipt_bytes(self) -> int:
        return self.tag.tag_bytes

    @property
    def per_leaf(self) -> int:
        return self.tag.per_leaf

    def keyed(self, seed: bytes):
        k = self.tag.keyed(seed[:KEY_BYTES])
        return lambda j, block, z=None: k(j, block)


class ArrayBacking:
    """A region's bytes in a numpy array: host DRAM, or a CPU test's store."""

    def __init__(self, a: np.ndarray) -> None:
        self.array = a.reshape(-1).view(np.uint8)
        self.nbytes = self.array.size

    def read(self, offset: int, n: int) -> bytes:
        return self.array[offset:offset + n].tobytes()

    def write(self, offset: int, data: bytes) -> None:
        self.array[offset:offset + len(data)] = np.frombuffer(data, np.uint8)


@dataclass
class _Region:
    info: RegionInfo
    backing: Backing
    codec: Codec
    receipt: Receipt
    source: str
    pool: _Pool | None = None
    layer: int = 0


@dataclass
class _Pool:
    """A KV pool over ``layout``: each page is ``per_page`` blocks, ``ids[p, c]`` the region block of page ``p``'s
    block ``c``, covering slots ``[lo[c], hi[c])`` of the page."""

    name: str
    regions: list[str]
    layout: PageLayout
    ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    allocated: np.ndarray
    fill: np.ndarray
    dirty: np.ndarray            # (pages, per_page): written since registered, not complete
    epoch: np.ndarray            # (pages, per_page): the block's registrations since the pool's
    step: np.ndarray             # (pages, per_page): the step that registered it, -1 for filler
    filler: str
    page_of: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # region block -> page
    slot_of: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # region block -> block of its page

    @property
    def per_page(self) -> int:
        return self.ids.shape[1]

    def read(self, pages: np.ndarray) -> np.ndarray:
        """(len(pages), per_page) bool: the blocks attention reads, those a write completed."""
        return ~self.dirty[pages] & (self.hi[None, :] <= self.fill[pages][:, None])


@dataclass
class _Pass:
    index: int
    seed: bytes | None
    segments: list[Segment]
    blocks: list[np.ndarray | None]             # each segment's region blocks in read order; None: its run
    epochs: dict[int, np.ndarray] = field(default_factory=dict)
    receipts: dict[int, list[bytes]] = field(default_factory=dict)
    digest: bytes = b""

    def segment_blocks(self, i: int) -> np.ndarray:
        b, s = self.blocks[i], self.segments[i]
        return np.arange(s.start, s.start + s.count, dtype=np.int64) if b is None else b

    def block(self, i: int, q: int) -> int:
        b = self.blocks[i]
        return self.segments[i].start + q if b is None else int(b[q])


class Store:
    """The node's store.  ``filler_plan(store, pass_index)`` returns the filler a pass reads after the weights and the
    KV, as ``(region, blocks)`` pairs (region block indices in read order); ``chaining`` derives each pass's seed
    (`chain`)."""

    def __init__(self, *, filler_plan: FillerPlan | None = None,
                 chaining: Callable[[bytes, bytes, int], bytes] = chain) -> None:
        self._regions: list[_Region] = []
        self._by_name: dict[str, _Region] = {}
        self._blocks = 0
        self._words = 0
        self._log_count = 0
        self._log_root = hashlib.sha512(REGISTRATION_DOMAIN).digest()
        self._pool: _Pool | None = None
        self._challenge: bytes | None = None
        self._last_root: bytes | None = None
        self._passes: list[PassCommitment] = []
        self._retained: dict[int, _Pass] = {}
        self._open_pass: _Pass | None = None
        self.filler_plan = filler_plan
        self.chaining = chaining
        self.counters = Counters()

    # --- registration -----------------------------------------------------------------------------------------------

    def _log(self, event: dict[str, Any], digest: bytes = b"") -> None:
        body = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
        self._log_root = hashlib.sha512(REGISTRATION_DOMAIN + self._log_root + frame(body) + frame(digest)).digest()
        self._log_count += 1

    def _add(self, name: str, tier: str, kind: str, backing: Backing, block_bytes: int, codec: Codec,
             receipt: Receipt, source: str) -> _Region:
        if name in self._by_name:
            raise ValueError(f"region {name!r} is already registered")
        if tier not in TIERS or kind not in KINDS:
            raise ValueError(f"tier {tier!r} must be one of {TIERS}, kind {kind!r} one of {KINDS}")
        if block_bytes <= 0 or block_bytes % codec.word_bytes or backing.nbytes % block_bytes:
            raise ValueError(f"{name}: {backing.nbytes} bytes are not whole blocks of {block_bytes} bytes of "
                             f"{codec.word_bytes}-byte words")
        info = RegionInfo(name, tier, kind, self._blocks, backing.nbytes // block_bytes, block_bytes, self._words,
                          codec.word_bytes, codec.name, receipt.name)
        r = _Region(info, backing, codec, receipt, source)
        self._regions.append(r)
        self._by_name[name] = r
        self._blocks += info.blocks
        self._words += backing.nbytes // codec.word_bytes
        return r

    def _region_record(self, r: _Region) -> dict[str, Any]:
        i = r.info
        return {"name": i.name, "tier": i.tier, "kind": i.kind, "first": i.first, "blocks": i.blocks,
                "block_bytes": i.block_bytes, "first_word": i.first_word, "codec": i.codec, "receipt": i.receipt,
                "source": r.source}

    def register(self, name: str, tier: str, kind: str, backing: Backing, *, block_bytes: int, codec: Codec,
                 receipt: Receipt, source: str, encode: bool = True) -> RegionInfo:
        """Register a region.  With ``encode`` its backing holds the payload and is encoded here, in place; without, it
        holds the encoding already (a device codec encoded it).  ``source`` names the payload (the weights' digest, a
        filler rule) for the verifier's decoding check."""
        t = time.perf_counter()
        r = self._add(name, tier, kind, backing, block_bytes, codec, receipt, source)
        if encode:
            self.encode_range(name, 0, backing.nbytes)
        self._log({"event": "region", **self._region_record(r)})
        self.counters.add("register", time.perf_counter() - t)
        return r.info

    def encode_range(self, name: str, offset: int, n: int) -> None:
        """Encode ``n`` bytes of region ``name`` from ``offset`` in place (whole words of a codec whose payload fills
        its word)."""
        r = self._by_name[name]
        wb = r.codec.word_bytes
        if offset % wb or n % wb or r.codec.payload_bytes != wb:
            raise ValueError(f"{name}: in-place encoding needs whole words of a codec whose payload fills its word")
        r.backing.write(offset, r.codec.encode(r.backing.read(offset, n), r.info.first_word + offset // wb))

    def register_kv_pool(self, name: str, layers: Sequence[Backing], layout: PageLayout, *, block_bytes: int,
                         codec: Codec, receipt: Receipt, tier: str = "hbm",
                         filler: Callable[[int, int], bytes] | None = None, filler_name: str = "as-allocated",
                         encode: bool = True) -> tuple[RegionInfo, ...]:
        """Register a KV pool: one region per layer, every page filler.  ``filler(layer, page)`` gives a page's filler
        payload (its planes in order); without it the pool's bytes as allocated are the filler payload, named
        ``filler_name``.  With ``encode`` the filler is encoded here, in place."""
        if self._pool is not None:
            raise ValueError("the store holds one KV pool")
        ids, lo, hi = _page_blocks(layout, block_bytes)
        names = []
        for li, b in enumerate(layers):
            rn = f"{name}/{li}"
            r = self._add(rn, tier, "kv", b, block_bytes, codec, receipt, f"filler:{filler_name}")
            if r.info.blocks <= int(ids.max()):
                raise ValueError(f"layer {li}: {b.nbytes} bytes don't hold the pool's pages")
            r.layer = li
            names.append(rn)
            if filler is not None:
                ps, ks = layout.strides()
                for p in range(layout.pages):
                    data = filler(li, p)
                    for k in range(layout.planes):
                        b.write(p * ps + k * ks, data[k * layout.plane_bytes:(k + 1) * layout.plane_bytes])
            if encode:
                self.encode_range(rn, 0, b.nbytes)
        shape = ids.shape
        pool = _Pool(name, names, layout, ids, lo, hi, np.zeros(layout.pages, bool), np.zeros(layout.pages, np.int64),
                     np.zeros(shape, bool), np.zeros(shape, np.uint32), np.full(shape, -1, np.int64), filler_name)
        nb = self._by_name[names[0]].info.blocks if names else 0
        pool.page_of = np.full(nb, -1, np.int64)
        pool.slot_of = np.full(nb, -1, np.int64)
        pool.page_of[ids.ravel()] = np.repeat(np.arange(layout.pages), ids.shape[1])
        pool.slot_of[ids.ravel()] = np.tile(np.arange(ids.shape[1]), layout.pages)
        self._pool = pool
        for rn in names:
            self._by_name[rn].pool = pool
        self._log({"event": "kv-pool", "name": name, "layers": [self._region_record(self._by_name[n]) for n in names],
                   "pages": layout.pages, "slots": layout.slots, "slot_bytes": layout.slot_bytes,
                   "planes": layout.planes, "strides": list(layout.strides()), "slot_major": layout.slot_major,
                   "filler": filler_name})
        return tuple(self._by_name[n].info for n in names)

    def registration(self) -> Registration:
        return Registration(self._log_count, self._log_root)

    def layout(self) -> Layout:
        by_tier: dict[str, int] = {}
        for r in self._regions:
            by_tier[r.info.tier] = by_tier.get(r.info.tier, 0) + r.backing.nbytes
        return Layout(tuple(r.info for r in self._regions), self._blocks, by_tier)

    def region(self, name: str) -> RegionInfo:
        return self._by_name[name].info

    # --- the KV pool's pages ----------------------------------------------------------------------------------------

    @property
    def pool(self) -> _Pool:
        if self._pool is None:
            raise RuntimeError("no KV pool is registered")
        return self._pool

    def kv_alloc(self, pages: Sequence[int]) -> None:
        t = time.perf_counter()
        p = self.pool
        ix = np.asarray(pages, np.int64)
        if p.allocated[ix].any():
            raise RuntimeError(f"pages {ix[p.allocated[ix]][:8].tolist()} are already allocated")
        p.allocated[ix] = True
        p.fill[ix] = 0
        self.counters.add("kv_alloc", time.perf_counter() - t)

    def kv_free(self, pages: Sequence[int]) -> None:
        t = time.perf_counter()
        p = self.pool
        ix = np.asarray(pages, np.int64)
        if not p.allocated[ix].all():
            raise RuntimeError(f"pages {ix[~p.allocated[ix]][:8].tolist()} are not allocated")
        p.allocated[ix] = False
        self.counters.add("kv_free", time.perf_counter() - t)

    def kv_append(self, appends: Sequence[Append], step: int) -> None:
        """The step's KV writes: each block a write reaches turns dirty, and each it completes is registered again
        (epoch + 1, under ``step``).  One registration record for the step."""
        t = time.perf_counter()
        p = self.pool
        if appends:
            n = len(appends)
            pg = np.fromiter((a.page for a in appends), np.int64, n)
            first = np.fromiter((a.first for a in appends), np.int64, n)
            end = first + np.fromiter((a.count for a in appends), np.int64, n)
            if not p.allocated[pg].all() or (end > p.layout.slots).any() or (first < 0).any():
                raise RuntimeError("an append outside an allocated page's slots")
            touched = (p.lo[None, :] < end[:, None]) & (p.hi[None, :] > first[:, None])
            np.maximum.at(p.fill, pg, end)
            p.dirty[pg] |= touched
            done = p.dirty[pg] & (p.hi[None, :] <= p.fill[pg][:, None])
            rows, cols = np.nonzero(done)
            key = np.unique(pg[rows] * p.per_page + cols)
            kp, kc = np.divmod(key, p.per_page)
            p.dirty[kp, kc] = False
            p.epoch[kp, kc] += 1
            p.step[kp, kc] = step
            self._log({"event": "kv-append", "step": step, "appends": n, "registered": len(key)},
                      hashlib.sha512(np.stack([pg, first, end]).astype(">i8").tobytes()).digest())
        self.counters.add("kv_append", time.perf_counter() - t)

    def kv_write_reference(self, appends: Sequence[Append], payload: Callable[[int, int, int], bytes],
                           step: int) -> None:
        """The CPU reference of a step's KV write: slot ``s`` of page ``p`` in layer ``l`` takes ``payload(l, p, s)``
        (its planes in order, ``slot_bytes`` each), encoded on append, then `kv_append`."""
        p = self.pool
        lay = p.layout
        ps, ks = lay.strides()
        if not lay.slot_major:
            raise NotImplementedError("the reference writes slot-major (NHD) pages")
        for rn in p.regions:
            r = self._by_name[rn]
            for a in appends:
                for s in range(a.first, a.first + a.count):
                    data = payload(r.layer, a.page, s)
                    for k in range(lay.planes):
                        off = a.page * ps + k * ks + s * lay.slot_bytes
                        r.backing.write(off, data[k * lay.slot_bytes:(k + 1) * lay.slot_bytes])
                        self.encode_range(rn, off, lay.slot_bytes)
        self.kv_append(appends, step)

    def idle_blocks(self, region: str) -> np.ndarray:
        """A KV layer region's registered blocks that no attention reads: every block of a free page, and the blocks of
        an allocated page no write reached since it left filler.  Region block indices, ascending."""
        p = self.pool
        idle = ~p.dirty & (~p.allocated[:, None] | (p.lo[None, :] >= p.fill[:, None]))
        return np.sort(p.ids[idle])

    # --- windows and passes -----------------------------------------------------------------------------------------

    @property
    def in_pass(self) -> bool:
        return self._open_pass is not None

    def begin_window(self, challenge: bytes) -> None:
        """Open a window under ``challenge`` (32 bytes).  Its first pass chains from the session's last root, or from
        the registration's root for the session's first window."""
        if self._open_pass is not None:
            raise RuntimeError("a window begins between passes")
        if self._challenge is not None:
            raise RuntimeError("a window is already open")
        if len(challenge) != CHALLENGE_BYTES:
            raise ValueError(f"a window's challenge is {CHALLENGE_BYTES} bytes")
        if self._last_root is None:
            self._last_root = self._log_root
        self._challenge, self._passes, self._retained = challenge, [], {}

    def end_window(self) -> WindowRecord:
        if self._challenge is None:
            raise RuntimeError("no window is open")
        if self._open_pass is not None:
            raise RuntimeError("a window ends between passes")
        rec = WindowRecord(self._challenge, tuple(self._passes))
        self._challenge = None
        return rec

    def begin_pass(self, live_pages: Sequence[int] | None = ()) -> _Pass:
        """Plan pass j: every weights region whole, each KV layer over ``live_pages``' blocks that attention reads
        (None: every allocated page), then the filler plan's.  Outside a window the pass has no seed and commits
        nothing."""
        t = time.perf_counter()
        if self._open_pass is not None:
            raise RuntimeError("a pass is already open")
        j = len(self._passes)
        seed = None
        if self._challenge is not None:
            seed = self.chaining(self._challenge, self._last_root, j)
        ps = _Pass(j, seed, [], [])
        for r in self._regions:
            if r.info.kind == "weights":
                ps.segments.append(Segment(r.info.name, 0, r.info.blocks, 0))
                ps.blocks.append(None)
        if self._pool is not None and live_pages is None:
            live_pages = np.nonzero(self._pool.allocated)[0]
        if self._pool is not None and len(live_pages):
            p = self._pool
            pages = np.unique(np.asarray(live_pages, np.int64))
            read = p.read(pages)
            ids, epochs = p.ids[pages][read], p.epoch[pages][read]
            for rn in p.regions:
                ps.segments.append(Segment(rn, -1, len(ids), 0))
                ps.blocks.append(ids)
                ps.epochs[len(ps.segments) - 1] = epochs
        if self.filler_plan is not None:
            for rn, blocks in self.filler_plan(self, j):
                blk = np.asarray(blocks, np.int64)
                run = len(blk) > 0 and bool((np.diff(blk) == 1).all())
                ps.segments.append(Segment(rn, int(blk[0]) if run else -1, len(blk), 0))
                ps.blocks.append(None if run else blk)
                r = self._by_name[rn]
                if r.pool is not None:
                    ps.epochs[len(ps.segments) - 1] = r.pool.epoch[r.pool.page_of[blk], r.pool.slot_of[blk]]
        pos = 0
        for i, s in enumerate(ps.segments):
            ps.segments[i] = Segment(s.region, s.start, s.count, pos)
            pos += s.count
        if seed is not None:
            runs = self.plan_runs(ps)
            if not distinct(runs):
                raise ValueError(f"pass {j}'s plan reads a block twice")
            ps.digest = plan_digest(runs)
        self._open_pass = ps
        self.counters.add("begin_pass", time.perf_counter() - t)
        return ps

    def plan_runs(self, ps: _Pass) -> np.ndarray:
        """The pass's plan as canonical runs of store indices, in read order (``(R, 2)``: first, count)."""
        parts, local = [], {}
        for i, s in enumerate(ps.segments):
            first, b = self._by_name[s.region].info.first, ps.blocks[i]
            if b is None:
                if s.count:
                    parts.append(np.array([[first + s.start, s.count]], np.int64))
                continue
            key = id(b)
            if key not in local:
                local[key] = runs_of(b)
            parts.append(local[key] + np.array([first, 0], np.int64))
        return canonical(np.concatenate(parts)) if parts else np.zeros((0, 2), np.int64)

    def receipts(self, ps: _Pass, index: int, z: Callable[[str, int], bytes | None] | None = None) -> list[bytes]:
        """The reference reader: segment ``index``'s receipts, from its blocks' bytes now."""
        seg = ps.segments[index]
        r = self._by_name[seg.region]
        keyed, bb, first = r.receipt.keyed(ps.seed), r.info.block_bytes, r.info.first
        return [keyed(first + b, r.backing.read(b * bb, bb), z(seg.region, b) if z else None)
                for b in ps.segment_blocks(index).tolist()]

    def end_pass(self, roots: dict[int, bytes] | None = None,
                 z: Callable[[str, int], bytes | None] | None = None) -> PassCommitment | None:
        """Close the open pass.  Each segment's root is the device reader's (``roots``) or the reference's from its
        receipts now; returns the commitment (None outside a window)."""
        t = time.perf_counter()
        ps = self._open_pass
        if ps is None:
            raise RuntimeError("no pass is open")
        self._open_pass = None
        if ps.seed is None:
            self.counters.add("end_pass", time.perf_counter() - t)
            return None
        seg_roots = []
        for i, s in enumerate(ps.segments):
            if roots is not None and i in roots:
                seg_roots.append(roots[i])
                continue
            ps.receipts[i] = self.receipts(ps, i, z)
            seg_roots.append(segment_root(self._by_name[s.region].receipt, ps.seed, ps.receipts[i]))
        root = pass_root(ps.index, ps.seed, ps.digest, ps.segments, seg_roots)
        pc = PassCommitment(ps.index, ps.seed, tuple(ps.segments), ps.digest, tuple(seg_roots), root)
        self._passes.append(pc)
        self._last_root = root
        self._retained[ps.index] = ps
        self.counters.add("end_pass", time.perf_counter() - t)
        return pc

    # --- plans and openings -----------------------------------------------------------------------------------------

    def plan(self, pass_index: int) -> np.ndarray:
        ps = self._retained[pass_index]
        parts = [self._by_name[s.region].info.first + ps.segment_blocks(i) for i, s in enumerate(ps.segments)]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def source(self, region: str, block: int) -> Source:
        r = self._by_name[region]
        if r.pool is None:
            return Source(r.info.kind, r.source, 0)
        p = r.pool
        page, c = int(p.page_of[block]), int(p.slot_of[block])
        st = int(p.step[page, c])
        return Source("kv" if st >= 0 else "filler", f"step:{st}" if st >= 0 else f"filler:{p.filler}",
                      int(p.epoch[page, c]))

    def kind_now(self, region: str, block: int) -> str:
        r = self._by_name[region]
        if r.pool is None:
            return r.info.kind
        return "kv" if r.pool.allocated[r.pool.page_of[block]] else "filler"

    def open(self, sample: Sequence[tuple[int, int]]) -> list[Opening]:
        """Each sampled (pass, position) as that pass read it.  A KV block re-registered since (its epoch moved) or
        dirty now raises `Stale`: the binder keeps freed pages out of reuse within a window."""
        out = []
        for j, position in sample:
            ps = self._retained.get(j)
            if ps is None:
                raise KeyError(f"pass {j} is not retained")
            k = next(i for i, s in enumerate(ps.segments) if s.position <= position < s.position + s.count)
            seg, q = ps.segments[k], position - ps.segments[k].position
            r = self._by_name[seg.region]
            b = ps.block(k, q)
            if k in ps.epochs:
                p = r.pool
                page, c = p.page_of[b], p.slot_of[b]
                if p.epoch[page, c] != ps.epochs[k][q] or p.dirty[page, c]:
                    raise Stale(f"{seg.region} block {b} was rewritten after pass {j}")
            if k not in ps.receipts:
                ps.receipts[k] = self.receipts(ps, k)
            ys, per = ps.receipts[k], r.receipt.per_leaf
            seg_tree = _tree(_leaves(r.receipt, ps.seed, ys))
            pass_tree = _tree(list(self._passes[j].segment_roots))
            bb = r.info.block_bytes
            out.append(Opening(j, position, r.info.first + b, seg.region, r.info.tier, self.kind_now(seg.region, b),
                               r.backing.read(b * bb, bb), ys[q], tuple(ys[(q // per) * per:(q // per + 1) * per]),
                               seg_tree.path(q // per), pass_tree.path(k), self.source(seg.region, b)))
        return out


def _page_blocks(layout: PageLayout, block_bytes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ids, lo, hi)``: the region blocks of each page (planes in order) and the slots each covers."""
    pb, sb = layout.plane_bytes, layout.slot_bytes
    ps, ks = layout.strides()
    if pb % block_bytes or ps % block_bytes or ks % block_bytes:
        raise ValueError(f"a {pb}-byte plane at strides {ps}, {ks} is not whole {block_bytes}-byte blocks")
    per_plane = pb // block_bytes
    i = np.arange(per_plane, dtype=np.int64)
    if layout.slot_major:
        lo, hi = (i * block_bytes) // sb, -(-((i + 1) * block_bytes) // sb)
    else:
        lo, hi = np.zeros(per_plane, np.int64), np.full(per_plane, layout.slots, np.int64)
    pages = np.arange(layout.pages, dtype=np.int64)[:, None]
    ids = np.concatenate([(pages * ps + k * ks) // block_bytes + i[None, :] for k in range(layout.planes)], axis=1)
    if len(np.unique(ids)) != ids.size:
        raise ValueError("the layout's pages overlap")
    return ids, np.tile(lo, layout.planes), np.tile(hi, layout.planes)


def verify_opening(info: RegionInfo, pc: PassCommitment, plan: Sequence[int], o: Opening, receipt: Receipt,
                   decode: Callable[[bytes, int], bytes], payload: bytes | None, z: bytes | None = None) -> bool:
    """The verifier's side of one opening (the reference): the plan's block at the position, its digest, the receipt
    recomputed from the block, the leaf and segment paths to the pass's root, and, with ``payload`` (what the block's
    registration source names), the block decoding to it."""
    from verity.core.protocols.bandwidth import tags
    from verity.core.protocols.pous.verifier import verify_path
    k = next(i for i, s in enumerate(pc.segments) if s.position <= o.position < s.position + s.count)
    seg, per = pc.segments[k], receipt.per_leaf
    q = o.position - seg.position
    if int(plan[o.position]) != o.block or seg.region != o.region or not info.first <= o.block < info.first + info.blocks:
        return False
    runs = canonical(runs_of(plan))
    if not distinct(runs) or plan_digest(runs) != pc.plan_digest or sum(s.count for s in pc.segments) != len(plan):
        return False
    if receipt.keyed(pc.seed)(o.block, o.data, z) != o.receipt or o.leaf_receipts[q % per] != o.receipt:
        return False
    leaf = tags.leaf_hash(tags.leaf_head(receipt, pc.seed[:KEY_BYTES]), q // per, o.leaf_receipts)
    if not verify_path(pc.segment_roots[k], -(-seg.count // per), q // per, leaf, o.leaf_path):
        return False
    if not verify_path(_tree(list(pc.segment_roots)).root, len(pc.segment_roots), k, pc.segment_roots[k],
                       o.segment_path):
        return False
    if pass_root(pc.index, pc.seed, pc.plan_digest, pc.segments, pc.segment_roots) != pc.root:
        return False
    return payload is None or decode(o.data, info.word_index(o.block - info.first)) == payload
