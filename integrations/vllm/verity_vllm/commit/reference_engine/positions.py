"""Canonical positions: leaf index <-> (class, instance, element) for one engine step (EXPERIMENTAL, CMT-1).

A step's committed stream is the concatenation of *segments*: one per acquired tensor (an Instance output = one manifest class
at one instance), in a fixed acquisition order (module execution order of the pinned model, which is fixed by the Program for a
given token count).  Every segment starts on a 4-byte boundary (bf16 / i32 / f32 values never straddle a leaf); the stream is
cut into fixed `LEAF_BYTES` = 64-byte leaves (32 bf16 values, 16 f32 / i32 values) and the last leaf is zero-padded.

    rank(class_id, instance, element) -> (leaf_index, byte offset inside the leaf, item bytes)
    unrank(leaf_index, in_leaf_offset)  -> (class_id, instance, element)

The layout is static context: it is written to disk (`positions.json`) and its digest is bound into the request root, so a
prover cannot move a value to another position after the fact.  A verifier derives the position of the value it wants from
the layout -- it never accepts a prover-supplied index.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import asdict, dataclass

LEAF_BYTES = 64
ALIGN = 4
ITEM_BYTES = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "u8": 1}


@dataclass(frozen=True, slots=True)
class Segment:
    class_id: str        # manifest class id (e.g. "gemm1/out/gemmcoordinate_v1/f2fpbf16")
    instance: str        # instance name (module path of the pinned model, e.g. "model.layers.3.self_attn.qkv_proj")
    dtype: str           # element type of the committed words (bf16 / f32 / i32)
    shape: tuple[int, ...]
    offset: int          # byte offset of element 0 in the step stream (multiple of ALIGN)
    nbytes: int          # numel * item bytes

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


class StepLayout:
    """Ordered segments of one step; built incrementally (`add`) as tensors are acquired, then frozen."""

    __slots__ = ("segments", "_offsets", "_by_key", "cursor", "frozen", "step", "n_tokens")

    def __init__(self, step: int, n_tokens: int) -> None:
        self.segments: list[Segment] = []
        self._offsets: list[int] = []
        self._by_key: dict[tuple[str, str], int] = {}
        self.cursor = 0
        self.frozen = False
        self.step = int(step)
        self.n_tokens = int(n_tokens)

    def add(self, class_id: str, instance: str, dtype: str, shape: tuple[int, ...]) -> Segment:
        if self.frozen:
            raise RuntimeError("layout frozen")
        if dtype not in ITEM_BYTES:
            raise ValueError(dtype)
        numel = 1
        for d in shape:
            numel *= int(d)
        nbytes = numel * ITEM_BYTES[dtype]
        off = self.cursor
        seg = Segment(class_id, instance, dtype, tuple(int(d) for d in shape), off, nbytes)
        key = (class_id, instance)
        if key in self._by_key:
            raise ValueError(f"duplicate segment {key} in step {self.step}")
        self._by_key[key] = len(self.segments)
        self.segments.append(seg)
        self._offsets.append(off)
        self.cursor = off + ((nbytes + ALIGN - 1) // ALIGN) * ALIGN
        return seg

    def freeze(self) -> None:
        self.frozen = True

    @property
    def stream_bytes(self) -> int:
        return self.cursor

    @property
    def leaf_count(self) -> int:
        return (self.cursor + LEAF_BYTES - 1) // LEAF_BYTES

    @property
    def padded_bytes(self) -> int:
        return self.leaf_count * LEAF_BYTES

    def segment(self, class_id: str, instance: str) -> Segment:
        return self.segments[self._by_key[(class_id, instance)]]

    def rank(self, class_id: str, instance: str, element: int) -> tuple[int, int, int]:
        seg = self.segment(class_id, instance)
        if not 0 <= element < seg.numel:
            raise IndexError(element)
        ib = ITEM_BYTES[seg.dtype]
        b = seg.offset + element * ib
        return b // LEAF_BYTES, b % LEAF_BYTES, ib

    def unrank(self, leaf_index: int, in_leaf: int) -> tuple[str, str, int] | None:
        """(class_id, instance, element) at that byte, or None for padding bytes."""
        b = leaf_index * LEAF_BYTES + in_leaf
        i = bisect.bisect_right(self._offsets, b) - 1
        if i < 0:
            return None
        seg = self.segments[i]
        if b >= seg.offset + seg.nbytes:
            return None
        ib = ITEM_BYTES[seg.dtype]
        return seg.class_id, seg.instance, (b - seg.offset) // ib

    def to_json(self) -> dict:
        return {"step": self.step, "n_tokens": self.n_tokens, "stream_bytes": self.cursor, "leaf_count": self.leaf_count,
                "leaf_bytes": LEAF_BYTES, "align": ALIGN,
                "segments": [{**asdict(s), "shape": list(s.shape)} for s in self.segments]}

    @classmethod
    def from_json(cls, d: dict) -> StepLayout:
        L = cls(d["step"], d["n_tokens"])
        for s in d["segments"]:
            seg = L.add(s["class_id"], s["instance"], s["dtype"], tuple(s["shape"]))
            if seg.offset != s["offset"] or seg.nbytes != s["nbytes"]:
                raise ValueError(f"layout json inconsistent at {s}")
        L.freeze()
        if L.cursor != d["stream_bytes"]:
            raise ValueError("layout json stream_bytes mismatch")
        return L

    def class_bytes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.segments:
            out[s.class_id] = out.get(s.class_id, 0) + s.nbytes
        return out


def positions_document(layouts: list[StepLayout], ctx: dict) -> dict:
    return {"schema": "verity-gen/cmt-ref/positions/v1", "leaf_bytes": LEAF_BYTES, "align": ALIGN, "context": ctx,
            "steps": [L.to_json() for L in layouts]}


def positions_digest(doc: dict) -> bytes:
    """Digest of the layout part only (steps), independent of the free-text context block."""
    return hashlib.sha256(json.dumps(doc["steps"], sort_keys=True, separators=(",", ":")).encode()).digest()
