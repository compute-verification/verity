"""The store's vocabulary and its two interfaces: `StoreView` for the protocol driver, `Lifecycle` for the engine hooks.

~~~text
store       every region the prover registered, in one index space of blocks: block b is block b - first of the region
            whose range holds it
tier        where a region's bytes live: "hbm" or "host" (host DRAM, read over PCIe)
kind        what a block holds: "weights", "kv", "filler" or "activations" (off unless a binder registers them)
block       the unit a receipt covers, fixed per region (a 4 KiB block, or a sketch vector of a fused GEMM)
codec       a region's encoding: payload words in, stored words out, `encode(payload, index)`, keyed by the word's index
receipt     a keyed hash of one block read in a pass, `keyed(seed)(j, block, z)`; `z` is the input a receipt bound to the
            prior tile's output reads (None for receipts that bind only the block)
registration  putting a block into the store: its bytes are the codec's encoding of a payload a `Source` names; every
            registration is a record in the log, and `Registration.root` commits the log
window      the interval of one challenge σ: `begin_window(σ)` ... `end_window()`
pass        one read of the live store: weights, the batch's written KV, and filler to top up; its seed is chained,
            seed_0 = H(σ), seed_j = H(σ, commitment_(j-1)), so pass j's receipts can't start before pass j-1 is committed
segment     a run of one region's blocks read in a pass, in read order; one reader (a fused GEMM, attention, the filler
            sweep) receipts it and its receipts make one subtree
commitment  a pass's root: SHA-512 over the pass index, its seed, the plan's digest and the RFC 6962 tree over its
            segments' roots, each segment's root the tree over its receipt leaves (`tags.leaf_hash`, `per_leaf` a leaf)
opening     for a sampled (pass, position): the block's bytes as read, its receipt, its paths to the pass's root and
            its registration record, from which the verifier recomputes the receipt and checks the decoding
~~~
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

TIERS = ("hbm", "host")
KINDS = ("weights", "kv", "filler", "activations")


class Keyed(Protocol):
    """A receipt under one pass's seed: ``keyed(j, block, z)`` is block ``j``'s receipt."""

    def __call__(self, j: int, block: bytes, z: bytes | None = None) -> bytes: ...


@runtime_checkable
class Codec(Protocol):
    """A region's encoding.  ``payload_bytes`` of payload go into each ``word_bytes`` stored word; ``index`` is the store
    index of the buffer's first word, which keys the encoding (a word encodes differently at another index).  A codec
    with device kernels also has ``encode_tensor_(tensor, index)`` (a region's bytes in place, at registration) and
    ``encode_slots_(cache, slot_mapping, layer)`` (a KV append, captured in the step's CUDA graph); the identity codec's
    are no-ops."""

    name: str
    word_bytes: int
    payload_bytes: int

    def encode(self, payload: bytes, index: int) -> bytes: ...

    def decode(self, stored: bytes, index: int) -> bytes: ...


@runtime_checkable
class Receipt(Protocol):
    """A receipt construction (the bandwidth audit's `Tag` with the optional bound input ``z``).  ``binds`` names what a
    receipt reads beside the block: ``()`` for sketch-then-hash, ``("z",)`` for one bound to the prior tile's output."""

    name: str
    receipt_bytes: int
    per_leaf: int
    binds: tuple[str, ...]

    def keyed(self, seed: bytes) -> Keyed: ...


class Backing(Protocol):
    """Where a region's bytes are: ``read(offset, n)`` and ``write(offset, data)`` in bytes from the region's start."""

    nbytes: int

    def read(self, offset: int, n: int) -> bytes: ...

    def write(self, offset: int, data: bytes) -> None: ...


@dataclass(frozen=True)
class Source:
    """What defines a registered block's payload, so a verifier can check its decoding: ``kind`` and ``ref`` (the weights'
    digest, the filler rule's name, the KV step that wrote it) and ``epoch``, the block's registration count."""

    kind: str
    ref: str
    epoch: int


@dataclass(frozen=True)
class RegionInfo:
    """A region as the driver sees it: its name, tier, kind, the store blocks ``[first, first + blocks)`` and their size,
    ``first_word``, the store index of its first codec word (a block's words are indexed from ``first_word + b *
    block_bytes / word_bytes``), and the codec's and receipt's names."""

    name: str
    tier: str
    kind: str
    first: int
    blocks: int
    block_bytes: int
    first_word: int
    word_bytes: int
    codec: str
    receipt: str

    def word_index(self, block: int) -> int:
        """The codec index of region block ``block``'s first word."""
        return self.first_word + block * (self.block_bytes // self.word_bytes)


@dataclass(frozen=True)
class PageLayout:
    """How a KV layer's pages sit in its backing: ``pages`` pages of ``slots`` token slots; a page is ``planes`` planes
    (K and V) of ``slots * slot_bytes`` bytes, plane ``k`` of page ``p`` at byte ``p * page_stride + k * plane_stride``.
    With ``slot_major`` a plane's slot axis is outermost (NHD), so a block covers a run of slots; otherwise (HND) every
    block of the plane holds part of every slot."""

    pages: int
    slots: int
    slot_bytes: int
    planes: int = 2
    page_stride: int = 0
    plane_stride: int = 0
    slot_major: bool = True

    @property
    def plane_bytes(self) -> int:
        return self.slots * self.slot_bytes

    def strides(self) -> tuple[int, int]:
        """``(page_stride, plane_stride)``, page-major by default."""
        return (self.page_stride or self.planes * self.plane_bytes, self.plane_stride or self.plane_bytes)


@dataclass(frozen=True)
class Layout:
    """The store: its regions in index order, its block count, and its bytes per tier."""

    regions: tuple[RegionInfo, ...]
    blocks: int
    bytes_by_tier: dict[str, int]


@dataclass(frozen=True)
class Registration:
    """The registration log so far: ``count`` records and ``root``, the hash chain over them (one link per registration
    event: a region, a page's written blocks, a page's return to filler)."""

    count: int
    root: bytes


@dataclass(frozen=True)
class Segment:
    """``count`` blocks of region ``region``, from its block ``start``, read in a pass in that order, at positions
    ``[position, position + count)`` of the pass."""

    region: str
    start: int
    count: int
    position: int


@dataclass(frozen=True)
class PassCommitment:
    """Pass ``index`` of its window: its seed, its plan (the segments in read order, and ``plan_digest`` over every
    block each reads, `StoreView.plan`), each segment's root and the pass's root, which seeds the next pass."""

    index: int
    seed: bytes
    segments: tuple[Segment, ...]
    plan_digest: bytes
    segment_roots: tuple[bytes, ...]
    root: bytes

    @property
    def blocks(self) -> int:
        return sum(s.count for s in self.segments)


@dataclass(frozen=True)
class WindowRecord:
    """A window: its challenge and its passes' commitments in order."""

    challenge: bytes
    passes: tuple[PassCommitment, ...]


@dataclass(frozen=True)
class Opening:
    """Block ``block`` (region ``region``, tier and kind as read) at position ``position`` of pass ``pass_index``: the
    bytes read, the receipt, the leaf's receipts (its ``per_leaf`` neighbours), the leaf's path in its segment's tree,
    the segment's path in the pass's tree, and the registration ``source`` the decoding is checked against."""

    pass_index: int
    position: int
    block: int
    region: str
    tier: str
    kind: str
    data: bytes
    receipt: bytes
    leaf_receipts: tuple[bytes, ...]
    leaf_path: tuple[bytes, ...]
    segment_path: tuple[bytes, ...]
    source: Source


class Stale(RuntimeError):
    """An opening of a block re-registered (rewritten) since the pass read it."""


@runtime_checkable
class StoreView(Protocol):
    """All a protocol driver sees of the store.  `layout` and `registration` go to the verifier at registration (and
    `registration` again whenever the log grows); `begin_window` takes a window's challenge; the engine's passes run
    inside it; `end_window` returns its pass commitments (each seeded by the previous, chained from the challenge);
    `plan` is a pass's blocks in read order (store indices; their digest is in the commitment); `open` answers the
    verifier's sample with each block as that pass read it, or raises `Stale`."""

    def layout(self) -> Layout: ...

    def registration(self) -> Registration: ...

    def begin_window(self, challenge: bytes) -> None: ...

    def end_window(self) -> WindowRecord: ...

    def plan(self, pass_index: int) -> Sequence[int]: ...

    def open(self, sample: Sequence[tuple[int, int]]) -> list[Opening]: ...


@dataclass(frozen=True)
class Append:
    """KV slots ``[first, first + count)`` of page ``page`` written in this step (a page holds ``page_slots`` tokens)."""

    page: int
    first: int
    count: int


@runtime_checkable
class Lifecycle(Protocol):
    """What the engine hooks call.  `register` is a region's registration (weight load: encode once; host filler).  A
    KV pool's pages start as filler (`register_kv_pool`); `kv_alloc` takes pages out of filler without touching their
    bytes, `kv_append` re-registers the blocks the step's tokens complete (encoded on append, on the device, by the
    codec's ``encode_slots_``), and `kv_free` returns pages to filler with their encoded bytes, already registered
    blocks, as they are.  `begin_pass` and `end_pass` bound a step: the pass's plan, then its commitment, from the
    segment roots a device reader hands in (``roots``, by segment number) or from the reference's receipts."""

    def register(self, name: str, tier: str, kind: str, backing: Backing, *, block_bytes: int, codec: Codec,
                 receipt: Receipt, source: str, encode: bool = True) -> RegionInfo: ...

    def register_kv_pool(self, name: str, layers: Sequence[Backing], layout: PageLayout, *, block_bytes: int,
                         codec: Codec, receipt: Receipt, tier: str = "hbm",
                         filler: Callable[[int, int], bytes] | None = None, filler_name: str = "as-allocated",
                         encode: bool = True) -> tuple[RegionInfo, ...]: ...

    def kv_alloc(self, pages: Sequence[int]) -> None: ...

    def kv_append(self, appends: Sequence[Append], step: int) -> None: ...

    def kv_free(self, pages: Sequence[int]) -> None: ...

    def begin_pass(self, live_pages: Sequence[int] | None = ()) -> Any: ...

    def end_pass(self, roots: dict[int, bytes] | None = None) -> PassCommitment | None: ...


@dataclass
class Counters:
    """What the hooks did, for the benchmark: calls and the host seconds spent in each hook."""

    calls: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def add(self, hook: str, seconds: float) -> None:
        self.calls[hook] = self.calls.get(hook, 0) + 1
        self.seconds[hook] = self.seconds.get(hook, 0.0) + seconds


class Identity:
    """The identity codec: a stored word is its payload.  The hooks' own overhead is measured under it."""

    name = "identity"
    word_bytes = 64
    payload_bytes = 64

    def encode(self, payload: bytes, index: int) -> bytes:
        return payload

    def decode(self, stored: bytes, index: int) -> bytes:
        return stored

    def encode_tensor_(self, tensor: Any, index: int) -> None:
        return None

    def encode_slots_(self, cache: Any, slot_mapping: Any, layer: int) -> None:
        return None


IDENTITY = Identity()
