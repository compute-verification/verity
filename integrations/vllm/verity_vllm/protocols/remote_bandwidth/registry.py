"""`NodeMemory`: the node's store as one registry, the reference implementation of `StoreView` and `Lifecycle`.

Regions are numbered into one block index space in registration order.  A KV pool is one page table shared by its
layer regions (a vLLM block id is a page in every layer): ``fill[page]`` slots of an allocated page are written, and
a block of a page is *registered* unless a write reached it and has not completed it.  So:

- ``kv_alloc`` takes pages out of filler and touches no byte: each block stays registered under its last record (the
  filler it was, or the KV it held before the page was freed) until a write reaches it;
- ``kv_append`` makes the blocks a write reaches dirty (their bytes are no registered encoding while they fill) and
  re-registers those it completes, under the step that wrote them; the codec encodes the slots as they are written
  (``encode_slots_`` on the device, `kv_write_reference` here);
- ``kv_free`` returns pages to filler with their bytes as they are: a registered block stays registered, a dirty block
  (a page freed part-written) stays out of every plan until a later write completes it.  While a memory map records
  (`record_memory_map`), a freed page's appended blocks (written since the map began) are released instead: dirty, so
  out of every plan until a write completes them again under a new index, as the audit's plans hold only blocks the
  circuit keeps live; its untouched blocks stay registered filler until a write reaches them.

A pass reads the weights' regions whole, each KV layer region over the blocks attention reads (the live pages'
blocks that writes completed), and the filler the `filler_plan` policy picks (`idle_blocks` lists the registered
blocks no attention reads: free pages, and allocated pages' blocks no write reached); no block twice (a plan that
repeats one is refused).  Its segments are laid out as subtrees of the pass's tree: each part cut into power-of-two
runs of whole leaves, the largest first, so that each starts at a multiple of its size, and the parts' last
``count % PER_LEAF`` blocks after them all (the tail, whose leaves may mix regions).  Inside a window each pass is
seeded from the window's challenge and the root before it (`chain`): the session's previous pass, across windows, and
the registration's root before the session's first.  The reference computes every receipt from the bytes
(`receipts`); a device reader hands `end_pass` its aligned segments' roots, or the pass's root.

The plan and the tree are the remote bandwidth driver's, imported (`rb`, its `receipts`): a pass's plan is its
`Plan` (canonical runs of distinct blocks; `plan_digest` is its digest in numpy), its tree `rb.domain`'s
frame-v3-sha512 over leaves of ``PER_LEAF`` receipts in plan order, each `rb.hidden` as an hm96-sha512/row/v2 leaf
under its salt, and an opening verifies by `rb.verify_leaf`.  `RemoteBandwidthReceipt` is its receipt.  The seed is
the chained one (`chain`, under its seed domain).  A leaf's salt is `NodeMemory.salt`, a PRF of the store's secret key, so
no salt is kept.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import secrets
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity.core.protocols.remote_bandwidth import receipts as rb
from verity.core.protocols.remote_bandwidth.receipts import PER_LEAF, SALT_BYTES, Plan
from verity.ml.circuits import requirement
from verity.ml.commitments import merkle_depth
from verity_vllm.protocols.remote_bandwidth.interface import (KINDS, TIERS, Append, Backing, Codec, Counters, Layout, Opening, PageLayout,
                                         PassCommitment, Receipt, Registration, RegionInfo, Segment, Source, Stale,
                                         WindowRecord)
from verity_vllm.protocols.remote_bandwidth import memory_map as mm

CHALLENGE_BYTES = 32
SEED_BYTES = 64
REGISTRATION_DOMAIN = b"verity/store/registration/v0\0"
SALT_DOMAIN = b"verity/store/salt/v0\0"

FillerPlan = Callable[["NodeMemory", int], Sequence[tuple[str, Sequence[int]]]]


def frame(b: bytes) -> bytes:
    return struct.pack(">Q", len(b)) + b


def chain(challenge: bytes, previous: bytes, index: int) -> bytes:
    """Pass ``index``'s seed in a window: the driver's chained `rb.seed`, ``SHA512(D ‖ σ ‖ u32be index ‖ root before
    it)``."""
    return rb.seed(challenge, index, previous)


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
    """`Plan.digest` of canonical ``runs``, without building the `Plan`: its construction and digest loop over the runs
    in Python, about 200 ms a pass at the 245k runs of a step with scattered live pages."""
    return hashlib.sha512(rb.PLAN_DOMAIN + struct.pack(">I", len(runs)) + np.asarray(runs, ">u8").tobytes()).digest()


def plan_of(runs: np.ndarray, blocks: int) -> Plan:
    """The driver's `Plan` of canonical ``runs`` in a store of ``blocks`` blocks; it refuses a plan that reads a block
    twice."""
    return Plan(tuple(map(tuple, runs.tolist())), blocks)


def leaf_digest(dom: Any, c: int, receipts: bytes, salt: bytes) -> bytes:
    """Leaf ``c`` of the driver's receipt tree: its receipts `rb.hidden` under ``salt``."""
    return dom.leaf(c, c, rb.SCHEMA, rb.hidden(receipts, salt))


def node_of(segment: Segment) -> tuple[int, int]:
    """``(level, index)``: the least node of the pass's tree over the segment's leaves, its subtree's root when the
    segment is `aligned`."""
    a, b = segment.position // PER_LEAF, (segment.position + segment.count - 1) // PER_LEAF
    level = (a ^ b).bit_length()
    return level, a >> level


def aligned(segment: Segment) -> bool:
    """Whether the segment is exactly a subtree: ``PER_LEAF · 2^h`` blocks from a multiple of that."""
    level, index = node_of(segment)
    return segment.count == PER_LEAF << level and segment.position == (index << level) * PER_LEAF


def _node(dom: Any, leaves: int, level: int, index: int, leaf: Callable[[int], bytes], memo: dict) -> bytes:
    """Node ``(level, index)`` of the tree over ``leaves`` leaves (``leaf(c)``), pads past them; ``memo`` holds the nodes
    known and gains those computed."""
    key = (level, index)
    if key not in memo:
        if level == 0:
            memo[key] = leaf(index) if index < leaves else dom.pad(index)
        else:
            memo[key] = dom.node(level - 1, index, _node(dom, leaves, level - 1, 2 * index, leaf, memo),
                                 _node(dom, leaves, level - 1, 2 * index + 1, leaf, memo))
    return memo[key]


@dataclass(frozen=True)
class KindReceipt:
    """The driver's receipt of a registration's kind (`rb.KINDS`: sketch-sha512, its default, or block-sha512) of the
    block alone, under the whole pass seed: `rb.receipter`'s, at the kind's default rows."""

    name: str = rb.DEFAULT
    receipt_bytes: int = rb.DIGEST_BYTES
    binds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rb.receipt_kind(self.name)

    def keyed(self, seed: bytes):
        r = rb.receipter(self.name, seed)
        return lambda j, block, z=None: r(j, block)


@dataclass(frozen=True)
class RemoteBandwidthReceipt:
    """The remote bandwidth driver's receipt (`rb.Receipter`): ``SHA512(Z(D ‖ frame(seed)) ‖ u64be j ‖ E_j ⊕
    spread(z))`` of a 4 KiB block, bound to the output ``z`` of the tile before its read."""

    name: str = "verity/remote-bandwidth/receipt/v0"
    receipt_bytes: int = rb.DIGEST_BYTES
    binds: tuple[str, ...] = ("z",)

    def keyed(self, seed: bytes):
        return rb.Receipter(seed)


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
    dirty: np.ndarray            # (pages, per_page): not registered: written and not complete, or released
    epoch: np.ndarray            # (pages, per_page): the block's registrations since the pool's
    step: np.ndarray             # (pages, per_page): the step that registered it, -1 for filler
    filler: str
    page_of: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # region block -> page
    slot_of: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # region block -> block of its page
    released: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))   # freed, its bytes still the block's

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
    views: dict[int, tuple[int, int]] = field(default_factory=dict)    # array segments: (id of the part's, offset)
    epochs: dict[int, np.ndarray] = field(default_factory=dict)
    receipts: dict[int, list[bytes]] = field(default_factory=dict)
    runs: np.ndarray | None = None              # inside a window: the plan's canonical runs, and their digest
    digest: bytes = b""
    store_blocks: int = 0
    top: dict[tuple[int, int], bytes] | None = None     # the tree's nodes outside the aligned segments; None: the reader's
    positions: int = 0
    starts: list[int] = field(default_factory=list)
    z: Callable[[str, int], bytes | None] | None = None     # the z each receipt is bound to, kept for the openings
    step: int | None = None
    _plan: Plan | None = None

    def plan(self) -> Plan:
        """The driver's `Plan` of the pass, built when the store builds or opens its tree."""
        if self._plan is None:
            self._plan = plan_of(self.runs, self.store_blocks)
        return self._plan

    def lay_out(self, parts: Sequence[tuple[str, int, int, np.ndarray | None, np.ndarray | None]]) -> None:
        """Segments from ``parts`` (region, start, count, blocks or None for a run, epochs or None) as subtrees: each
        part's power-of-two runs of whole leaves, the largest first, then each part's last ``count % PER_LEAF``."""
        by_size: dict[int, list] = {}
        tail = []
        for part in parts:
            q, r = divmod(part[2], PER_LEAF)
            lo = 0
            for h in range(q.bit_length() - 1, -1, -1):
                if q >> h & 1:
                    by_size.setdefault(h, []).append((part, lo, PER_LEAF << h))
                    lo += PER_LEAF << h
            if r:
                tail.append((part, lo, r))
        segments, blocks_of, starts, position = self.segments, self.blocks.append, self.starts.append, self.positions
        for (region, start, _, blocks, epochs), lo, n in [p for h in sorted(by_size, reverse=True) for p in by_size[h]] + tail:
            k = len(segments)
            segments.append(Segment(region, start + lo if blocks is None else -1, n, position))
            if blocks is None:
                blocks_of(None)
            else:
                blocks_of(blocks[lo:lo + n])
                self.views[k] = (id(blocks), lo)
            if epochs is not None:
                self.epochs[k] = epochs[lo:lo + n]
            starts(position)
            position += n
        self.positions = position

    def at(self, position: int) -> int:
        """The segment holding ``position``."""
        return bisect.bisect_right(self.starts, position) - 1

    def segment_blocks(self, i: int) -> np.ndarray:
        b, s = self.blocks[i], self.segments[i]
        return np.arange(s.start, s.start + s.count, dtype=np.int64) if b is None else b

    def block(self, i: int, q: int) -> int:
        b = self.blocks[i]
        return self.segments[i].start + q if b is None else int(b[q])


class NodeMemory:
    """The node's memory.  ``filler_plan(store, pass_index)`` returns the filler a pass reads after the weights and the
    KV, as ``(region, blocks)`` pairs (region block indices in read order); ``chaining`` derives each pass's seed
    (`chain`); ``salt_key`` keys the leaves' salts (`salt`), fresh by default."""

    def __init__(self, *, filler_plan: FillerPlan | None = None,
                 chaining: Callable[[bytes, bytes, int], bytes] = chain, salt_key: bytes | None = None) -> None:
        self._salt_key = secrets.token_bytes(32) if salt_key is None else salt_key
        self._regions: list[_Region] = []
        self._by_name: dict[str, _Region] = {}
        self._blocks = self._words = self._log_count = 0
        self._log_root = hashlib.sha512(REGISTRATION_DOMAIN).digest()
        self._pool: _Pool | None = None
        self._challenge: bytes | None = None
        self._last_root: bytes | None = None
        self._passes: list[PassCommitment] = []
        self._window = self._pass_step = -1
        self._retained: dict[tuple[int, int], _Pass] = {}
        self._open_pass: _Pass | None = None
        self.filler_plan, self.chaining = filler_plan, chaining
        self.counters = Counters()
        self.memory_map: mm.MemoryMap | None = None
        self.on_write: Callable[[int, np.ndarray], None] | None = None

    # --- registration -----------------------------------------------------------------------------------------------

    def _log(self, event: dict[str, Any], digest: bytes = b"") -> None:
        body = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
        self._log_root = hashlib.sha512(REGISTRATION_DOMAIN + self._log_root + frame(body) + frame(digest)).digest()
        self._log_count += 1

    def _add(self, name: str, tier: str, kind: str, backing: Backing, block_bytes: int, codec: Codec,
             receipt: Receipt, source: str) -> _Region:
        if name in self._by_name:
            raise ValueError(f"region {name!r} is already registered")
        if self.memory_map is not None:
            raise RuntimeError(f"region {name!r} after the memory map's header fixed the store: register regions first")
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
        pool.page_of, pool.slot_of, pool.released = np.full(nb, -1, np.int64), np.full(nb, -1, np.int64), np.zeros(shape, bool)
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

    def backing(self, name: str) -> Backing:
        return self._by_name[name].backing

    def payload(self, slot: int) -> bytes:
        """Store block ``slot``'s payload now: its region codec's decoding of the bytes it holds."""
        r = self._regions[bisect.bisect_right([x.info.first for x in self._regions], slot) - 1]
        b, bb = slot - r.info.first, r.info.block_bytes
        return r.codec.decode(r.backing.read(b * bb, bb), r.info.word_index(b))

    def record_memory_map(self, path: Any = None, registration: bytes | None = None, lifetime: int = mm.LIFETIME,
                          on_line: Any = None, on_append: Any = None) -> mm.MemoryMap:
        """Record each step's KV blocks as memory-map/v1 into ``path`` (the prover's own file) or kept, beside
        ``registration``, the remote audit's `Registration.digest`, which the session's first window chains from, under
        the session's lifetime cap of ``lifetime`` windows, each append to ``on_append`` and each closed line to
        ``on_line`` (`memory_map.record`).  The registration took every block as its bytes are, so none is dirty now."""
        m = mm.record(self, path, registration, lifetime, on_line, on_append)
        if self._pool is not None:
            self._pool.dirty[:] = self._pool.released[:] = False
        return m

    def audit_store(self) -> requirement.Store:
        """The node's store, the audit's `requirement.Store`, as its memory map records it (`memory_map.audit_store`)."""
        return mm.audit_store(self.memory_map)

    def end_step(self) -> None:
        """The step's forward has written its KV (`Scheduler.update_from_output`): a memory map takes its digests."""
        if self.memory_map is not None:
            self.memory_map.end_step()

    # --- the KV pool's pages ----------------------------------------------------------------------------------------

    @property
    def pool(self) -> _Pool:
        if self._pool is None:
            raise RuntimeError("no KV pool is registered")
        return self._pool

    def kv_alloc(self, pages: Sequence[int]) -> None:
        t, p = time.perf_counter(), self.pool
        ix = np.asarray(pages, np.int64)
        if p.allocated[ix].any():
            raise RuntimeError(f"pages {ix[p.allocated[ix]][:8].tolist()} are already allocated")
        p.allocated[ix] = True
        p.fill[ix] = 0
        self.counters.add("kv_alloc", time.perf_counter() - t)

    def kv_free(self, pages: Sequence[int]) -> None:
        t, p = time.perf_counter(), self.pool
        ix = np.asarray(pages, np.int64)
        if not p.allocated[ix].all():
            raise RuntimeError(f"pages {ix[~p.allocated[ix]][:8].tolist()} are not allocated")
        p.allocated[ix] = False
        if self.memory_map is not None:
            keys = (ix[:, None] * p.per_page + np.arange(p.per_page))[~p.dirty[ix]]
            gone = keys[self.memory_map.free(keys)]
            p.dirty[gone // p.per_page, gone % p.per_page] = p.released[gone // p.per_page, gone % p.per_page] = True
        self.counters.add("kv_free", time.perf_counter() - t)

    def kv_append(self, appends: Sequence[Append], step: int) -> None:
        """The step's KV writes: each block a write reaches turns dirty, and each it completes is registered again
        (epoch + 1, under ``step``).  Those it reaches while registered or released with their bytes as they were (a
        plan may still hold them) go first to ``on_write(step, pool blocks)``, if set, before the forward writes them.  One registration record for the step.  A memory map closes the last step's line, opens this one's (`memory_map`).  A store with no KV pool takes steps with no appends."""
        t, p, m = time.perf_counter(), self.pool if appends else self._pool, self.memory_map
        if m is not None:
            m.flush(step)
        key = reached = np.zeros(0, np.int64)
        if appends:
            n = len(appends)
            pg = np.fromiter((a.page for a in appends), np.int64, n)
            first = np.fromiter((a.first for a in appends), np.int64, n)
            end = first + np.fromiter((a.count for a in appends), np.int64, n)
            if not p.allocated[pg].all() or (end > p.layout.slots).any() or (first < 0).any():
                raise RuntimeError("an append outside an allocated page's slots")
            touched = (p.lo[None, :] < end[:, None]) & (p.hi[None, :] > first[:, None])
            blk = pg[:, None] * p.per_page + np.arange(p.per_page)
            reached = np.unique(blk[touched & ~p.dirty[pg]])
            np.maximum.at(p.fill, pg, end)
            if self.on_write is not None:
                self.on_write(step, np.unique(blk[touched & (~p.dirty[pg] | p.released[pg])]))
            p.dirty[pg] |= touched
            p.released.flat[blk[touched]] = False
            done = p.dirty[pg] & (p.hi[None, :] <= p.fill[pg][:, None])
            rows, cols = np.nonzero(done)
            key = np.unique(pg[rows] * p.per_page + cols)
            kp, kc = np.divmod(key, p.per_page)
            p.dirty[kp, kc] = False
            p.epoch[kp, kc] += 1
            p.step[kp, kc] = step
            self._log({"event": "kv-append", "step": step, "appends": n, "registered": len(key)},
                      hashlib.sha512(np.stack([pg, first, end]).astype(">i8").tobytes()).digest())
        if m is not None:
            m.open(step, reached, key)
        self.counters.add("kv_append", time.perf_counter() - t)

    def kv_write_reference(self, appends: Sequence[Append], payload: Callable[[int, int, int], bytes],
                           step: int) -> None:
        """The CPU reference of a step's KV write: slot ``s`` of page ``p`` in layer ``l`` takes ``payload(l, p, s)``
        (its planes in order, ``slot_bytes`` each), encoded on append, then `kv_append` and `end_step`."""
        p = self.pool
        lay = p.layout
        ps, ks = lay.strides()
        if not lay.slot_major:
            raise NotImplementedError("the reference writes slot-major (NHD) pages")
        if self.memory_map is not None:
            self.memory_map.flush(step)
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
        self.end_step()

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

    @property
    def window(self) -> int:
        """The latest window's number, -1 before the first."""
        return self._window

    @property
    def window_open(self) -> bool:
        return self._challenge is not None

    def begin_window(self, challenge: bytes, registration: bytes | None = None) -> None:
        """Open a window under ``challenge`` (32 bytes).  Its first pass chains from the session's last root.  The
        session's first window chains from ``registration``, the verifier's registration digest (the driver's
        `Registration.digest`), or without one from the store's own registration root."""
        if self._open_pass is not None:
            raise RuntimeError("a window begins between passes")
        if self._challenge is not None:
            raise RuntimeError("a window is already open")
        if len(challenge) != CHALLENGE_BYTES:
            raise ValueError(f"a window's challenge is {CHALLENGE_BYTES} bytes")
        if self.memory_map is not None and self._last_root is None and registration != self.memory_map.header.registration:
            raise ValueError("the session's first window chains from the registration digest its memory map names")
        if self._last_root is None:
            self._last_root = self._log_root if registration is None else registration
        elif registration is not None:
            raise ValueError("only a session's first window chains from the registration")
        self._window += 1
        self._retained = {k: ps for k, ps in self._retained.items() if k[0] == self._window - 1}
        self._challenge, self._passes = challenge, []

    def end_window(self) -> WindowRecord:
        if self._challenge is None:
            raise RuntimeError("no window is open")
        if self._open_pass is not None:
            raise RuntimeError("a window ends between passes")
        rec = WindowRecord(self._challenge, tuple(self._passes), self._window)
        self._challenge = None
        return rec

    def begin_pass(self, live_pages: Sequence[int] | None = (), step: int | None = None) -> _Pass:
        """Plan pass j: every weights region whole, each KV layer over ``live_pages``' blocks that attention reads
        (None: every allocated page), then the filler plan's.  Outside a window the pass has no seed and commits
        nothing.  ``step`` is the engine step whose reads it covers, above the last pass's; while a memory map records,
        the step `kv_append` opened last."""
        t = time.perf_counter()
        if self._open_pass is not None:
            raise RuntimeError("a pass is already open")
        if step is not None and (step <= self._pass_step or (self.memory_map is not None and step != self.memory_map.last)):
            raise ValueError(f"a pass covers the step kv_append opened last, above the last pass's {self._pass_step}: "
                             f"not {step}")
        self._pass_step = self._pass_step if step is None else step
        j = len(self._passes)
        seed = None
        if self._challenge is not None:
            seed = self.chaining(self._challenge, self._last_root, j)
        ps = _Pass(j, seed, [], [], step=step)
        parts = [(r.info.name, 0, r.info.blocks, None, None) for r in self._regions if r.info.kind == "weights"]
        if self._pool is not None and live_pages is None:
            live_pages = np.nonzero(self._pool.allocated)[0]
        if self._pool is not None and len(live_pages):
            p = self._pool
            pages = np.unique(np.asarray(live_pages, np.int64))
            read = p.read(pages)
            ids, epochs = p.ids[pages][read], p.epoch[pages][read]
            parts += [(rn, -1, len(ids), ids, epochs) for rn in p.regions]
        if self.filler_plan is not None:
            for rn, blocks in self.filler_plan(self, j):
                blk = np.asarray(blocks, np.int64)
                run = len(blk) > 0 and bool((np.diff(blk) == 1).all())
                r = self._by_name[rn]
                ep = r.pool.epoch[r.pool.page_of[blk], r.pool.slot_of[blk]] if r.pool is not None else None
                if r.pool is not None and r.pool.dirty[r.pool.page_of[blk], r.pool.slot_of[blk]].any():
                    raise ValueError(f"pass {j}'s filler reads {rn} blocks that are dirty or released (`idle_blocks`)")
                parts.append((rn, int(blk[0]) if run else -1, len(blk), None if run else blk, ep))
        ps.lay_out(parts)
        if seed is not None:
            ps.runs, ps.store_blocks = self.plan_runs(ps), self._blocks
            if not len(ps.runs):
                raise ValueError(f"pass {j} reads no block")
            if not distinct(ps.runs):
                raise ValueError(f"pass {j}'s plan reads a block twice")
            ps.digest = plan_digest(ps.runs)
        self._open_pass = ps
        self.counters.add("begin_pass", time.perf_counter() - t)
        return ps

    def plan_runs(self, ps: _Pass) -> np.ndarray:
        """The pass's plan as canonical runs of store indices, in read order (``(R, 2)``: first, count)."""
        parts, local, runs = [], {}, []
        for i, s in enumerate(ps.segments):
            first, b = self._by_name[s.region].info.first, ps.blocks[i]
            if b is None:
                runs.append((first + s.start, s.count))
                continue
            if runs:
                parts.append(np.array(runs, np.int64))
                runs = []
            key = (*ps.views[i], s.count)
            if key not in local:
                local[key] = runs_of(b)
            parts.append(local[key] + np.array([first, 0], np.int64))
        if runs:
            parts.append(np.array(runs, np.int64))
        return canonical(np.concatenate(parts)) if parts else np.zeros((0, 2), np.int64)

    def salt(self, seed: bytes, leaf: int) -> bytes:
        """Leaf ``leaf``'s salt in the pass seeded ``seed``: ``SHAKE256(D ‖ key ‖ seed ‖ u64be leaf)``, 192 bytes."""
        return hashlib.shake_256(SALT_DOMAIN + self._salt_key + seed + struct.pack(">Q", leaf)).digest(SALT_BYTES)

    def receipts_at(self, ps: _Pass, lo: int, hi: int, z: Callable[[str, int], bytes | None] | None = None) -> list[bytes]:
        """The receipts of positions ``[lo, hi)``, the reference reader's (`receipts`) for the segments they fall in."""
        out, k = [], ps.at(lo)
        while lo < hi:
            s = ps.segments[k]
            if k not in ps.receipts:
                ps.receipts[k] = self.receipts(ps, k, z)
            end = min(hi, s.position + s.count)
            out += ps.receipts[k][lo - s.position:end - s.position]
            lo, k = end, k + 1
        return out

    def _leaf(self, ps: _Pass, dom: Any, z: Callable[[str, int], bytes | None] | None = None) -> Callable[[int], bytes]:
        def leaf(c: int) -> bytes:
            ys = self.receipts_at(ps, c * PER_LEAF, min((c + 1) * PER_LEAF, ps.positions), z)
            return leaf_digest(dom, c, b"".join(ys), self.salt(ps.seed, c))
        return leaf

    def receipts(self, ps: _Pass, index: int, z: Callable[[str, int], bytes | None] | None = None) -> list[bytes]:
        """The reference reader: segment ``index``'s receipts, from its blocks' bytes now."""
        seg, z = ps.segments[index], ps.z if z is None else z
        r = self._by_name[seg.region]
        keyed, bb, first = r.receipt.keyed(ps.seed), r.info.block_bytes, r.info.first
        return [keyed(first + b, r.backing.read(b * bb, bb), z(seg.region, b) if z else None)
                for b in ps.segment_blocks(index).tolist()]

    def end_pass(self, roots: dict[int, bytes] | None = None, z: Callable[[str, int], bytes | None] | None = None, *,
                 root: bytes | None = None) -> PassCommitment | None:
        """Close the open pass and return its commitment (None outside a window).  A device reader hands in its
        aligned segments' roots (``roots``, by segment number), the reference computing the rest of the tree from its
        receipts now, or the pass's ``root`` when it built the whole tree (then it, not the store, opens the pass).
        ``z(region, block)`` is what each receipt is bound to; the pass keeps it for its openings."""
        t = time.perf_counter()
        ps = self._open_pass
        if ps is None:
            raise RuntimeError("no pass is open")
        roots = roots or {}
        if any(not aligned(ps.segments[k]) for k in roots):
            raise ValueError(f"pass {ps.index}'s segments {sorted(roots)} are not all subtrees of its tree")
        self._open_pass, ps.z = None, z
        if ps.seed is None:
            self.counters.add("end_pass", time.perf_counter() - t)
            return None
        if root is not None:
            pc = PassCommitment(ps.index, ps.seed, tuple(ps.segments), ps.digest, (), root, ps.step)
        else:
            dom = rb.domain(ps.seed, ps.plan())
            leaf, n = self._leaf(ps, dom, z), dom.count
            ps.top = {node_of(s): roots[k] if k in roots else _node(dom, n, *node_of(s), leaf, {})
                      for k, s in enumerate(ps.segments) if aligned(s)}
            root = _node(dom, n, merkle_depth(n), 0, leaf, ps.top)
            seg_roots = tuple(_node(dom, n, *node_of(s), leaf, ps.top) for s in ps.segments)
            pc = PassCommitment(ps.index, ps.seed, tuple(ps.segments), ps.digest, seg_roots, root, ps.step)
        self._passes.append(pc)
        self._last_root = root
        self._retained[(self._window, ps.index)] = ps
        self.counters.add("end_pass", time.perf_counter() - t)
        return pc

    # --- plans and openings -----------------------------------------------------------------------------------------

    def plan(self, pass_index: int, window: int | None = None) -> np.ndarray:
        ps = self._retained[(self._window if window is None else window, pass_index)]
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

    def open(self, sample: Sequence[tuple[int, int]], window: int | None = None) -> list[Opening]:
        """Each sampled (pass, position) of ``window`` (None: the latest) as that pass read it.  A KV block
        re-registered since (its epoch moved) or dirty now raises `Stale`: the binder reuses freed pages as late as
        the pool allows."""
        out, inner = [], {}
        t = self._window if window is None else window
        for j, position in sample:
            ps = self._retained.get((t, j))
            if ps is None:
                raise KeyError(f"pass {j} of window {t} is not retained")
            if ps.top is None:
                raise RuntimeError(f"pass {j}'s tree is its reader's, which opens it")
            k = ps.at(position)
            seg, q = ps.segments[k], position - ps.segments[k].position
            r = self._by_name[seg.region]
            b = ps.block(k, q)
            if k in ps.epochs:
                p = r.pool
                page, c = p.page_of[b], p.slot_of[b]
                if p.epoch[page, c] != ps.epochs[k][q] or p.dirty[page, c]:
                    raise Stale(f"{seg.region} block {b} was rewritten after pass {j}")
            dom = rb.domain(ps.seed, ps.plan())
            leaf, n, c, (level, _) = self._leaf(ps, dom), dom.count, position // PER_LEAF, node_of(seg)
            memo = inner.setdefault((j, k), {}) if aligned(seg) else ps.top
            path = [_node(dom, n, h, (c >> h) ^ 1, leaf, memo if h < level else ps.top) for h in range(merkle_depth(n))]
            ys = self.receipts_at(ps, c * PER_LEAF, min((c + 1) * PER_LEAF, ps.positions))
            bb = r.info.block_bytes
            out.append(Opening(j, position, r.info.first + b, seg.region, r.info.tier, self.kind_now(seg.region, b),
                               r.backing.read(b * bb, bb), ys[position % PER_LEAF], tuple(ys), tuple(path[:level]),
                               tuple(path[level:]), self.source(seg.region, b), self.salt(ps.seed, c)))
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
