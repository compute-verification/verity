"""Canonical encoding of in-kernel FlashAttention values (R8 lane B-commit, 2026-09-18).

The commitment engine (C0, `tests/commit/semantic_layout.py`) hashes ONLY packed value bytes; the identity of
a leaf is its POSITION in a layout that is derived by arithmetic from static context.  This module fixes that layout for the
interior values of `Attention_v3{T, NH, KVH, D, BN}` (`verity/ml/circuits/ops/b1.py::AttnBlockV3 / AttentionHeadV3`).

Value word
    fp32 stage -> the 4 little-endian bytes of the IEEE-754 bit pattern (u32); bf16 stage -> 2 bytes (u16).  No conversion.

Identity   ``ValueId(layer, head, qrow, stage, block, index, version)``
    layer, head : the Program's (layer, attention head) of the Attention Instance
    qrow        : the query token position t (the Instance `Attention_v3{T=t+1}` has ONE query row; T = qrow + 1 visible keys)
    stage       : one of the committed stages below (or the derived `P16`, see :data:`DERIVED`)
    block       : key-block index b (keys [b*BN, (b+1)*BN)); the epilogue uses block = nb
    index       : key column c (S, P32) | lane 0..3 (L) | d 0..D-1 (O) | 0 (M, ALPHA, INV)
    version     : online-state version = the visit ordinal of the block (FA2 visits blocks LAST -> FIRST, so version
                  v = nb - 1 - b; the epilogue has version nb).  Redundant with (T, block) by construction, kept so that a
                  reader can name "the running state after v blocks" without knowing T.

Committed stages, in the kernel's emission order inside one visited block (this IS the positional layout):
    S[c]   fp32  score q.k (DotBf16)                       for every visible column c of the block
    M      fp32  running max after this block (m_new)
    ALPHA  fp32  rescale factor exp2((m_prev - m_new) * scale_log2)   -- absent on the first visited block
    P32[c] fp32  exp2(S*scale_log2 - m_new*scale_log2)    for every visible column
    L[ln]  fp32  the 4 lane sums after this block
    O[d]   fp32  the PV accumulator after this block
  epilogue:
    INV    fp32  1/sum (Fa2InvSum)

Derived (NOT committed; opened through their source under a declared conversion, see `derived.py`):
    P16[c] = F2fpBf16(P32[c])   the bf16 MMA operand.  RNE fp32->bf16 is NOT injective, so the source P32 is the committed
                               word and P16 is derived from it (the reverse direction would lose information).
    (the RMSNorm `Bf16ToF32` rule of deliverable 3 is the same mechanism with an injective conversion: the bf16 native
     word is committed, the fp32 lift is derived.)

What is not in the stream and why: `s - m` / `max_scaled` (fan-out 1, absorbable per the census §4a), `P16` (derived),
the bf16 attention OUTPUT word (a native Instance output, already committed by the RU tree; opened there, not here).

Positions.  For one row (layer, head, qrow) with T = qrow + 1, nb = ceil(T / BN), the stream is
    for v in 0..nb-1:  b = nb - 1 - v;  nvis = min(T, (b+1)*BN) - b*BN
        S[b*BN .. b*BN+nvis)  M  [ALPHA if v > 0]  P32[..]  L[0..3]  O[0..D-1]
    INV
so |row| = 2T + nb*(D + 6).  Positions are arithmetic in (T, D, BN): `RowLayout.rank / unrank`.  A tile (layer, head,
q-block) concatenates its rows; the hierarchical layout (`hier.py`) regroups the SAME identities by k-block so that a
per-(head, q-block, k-block) subtree exists before any challenge.  Domain-separation tags and the context binding
(Program digest, request id, step) live in `hier.py` / `packed.py` next to the roots they protect; the identity above is
what every layout ranks.

Canonical production id (for the cross-check against the production id-carrying verifier, C0 §C0.2):
    p=<profile>/r=<request>/t=<qrow>/ph=<phase>/l=<layer>/op=self_attn.attn.fa2.<stage>/port=0/c=<head>,<block>,<index>,<version>
which is inside the frozen `verity_vllm.commit.identity` grammar (`c=` is a comma-joined coordinate).
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from typing import Iterator

COMMITTED_STAGES: tuple[str, ...] = ("S", "M", "ALPHA", "P32", "L", "O", "INV")
DERIVED: dict[str, tuple[str, str]] = {"P16": ("P32", "F2fpBf16")}  # derived stage -> (source stage, conversion name)
STAGE_WIDTH: dict[str, int] = {"S": 4, "M": 4, "ALPHA": 4, "P32": 4, "L": 4, "O": 4, "INV": 4, "P16": 2}
STAGE_DTYPE: dict[str, str] = {s: ("bfloat16" if w == 2 else "float32") for s, w in STAGE_WIDTH.items()}
NLANES = 4

ENCODING_VERSION = "verity/fa2c/encoding/v0"


@dataclass(frozen=True, slots=True, order=True)
class ValueId:
    layer: int
    head: int
    qrow: int
    stage: str
    block: int
    index: int
    version: int

    @property
    def width(self) -> int:
        return STAGE_WIDTH[self.stage]

    @property
    def dtype(self) -> str:
        return STAGE_DTYPE[self.stage]

    def canonical_id(self, profile: str, request: str, phase: str) -> str:
        return (f"p={profile}/r={request}/t={self.qrow}/ph={phase}/l={self.layer}/op=self_attn.attn.fa2.{self.stage}"
                f"/port=0/c={self.head},{self.block},{self.index},{self.version}")


def lane_of(col: int) -> int:
    """FA2 quad lane of key column c: elements `8*nj + 2*q + j` -> q (`b1._lane`)."""
    return (col % 8) // 2


@dataclass(frozen=True, slots=True)
class Geometry:
    """Static shape of one Attention Instance family: head dim D, kernel key block BN, query block QB (kBlockM)."""

    D: int = 64
    BN: int = 128
    QB: int = 64
    NH: int = 9
    KVH: int = 3

    def digest(self) -> bytes:
        doc = {"v": ENCODING_VERSION, "D": self.D, "BN": self.BN, "QB": self.QB, "NH": self.NH, "KVH": self.KVH,
               "stages": list(COMMITTED_STAGES), "derived": DERIVED, "widths": STAGE_WIDTH, "nlanes": NLANES}
        return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).digest()


class RowLayout:
    """rank/unrank between the positions of ONE row's stream and :class:`ValueId` (layer, head, qrow fixed)."""

    __slots__ = ("layer", "head", "qrow", "T", "D", "BN", "nb", "_starts", "_sizes", "count")

    def __init__(self, geo: Geometry, layer: int, head: int, qrow: int) -> None:
        self.layer, self.head, self.qrow = layer, head, qrow
        self.T = qrow + 1
        self.D, self.BN = geo.D, geo.BN
        self.nb = (self.T + self.BN - 1) // self.BN
        sizes = []
        for v in range(self.nb):
            b = self.nb - 1 - v
            nvis = min(self.T, (b + 1) * self.BN) - b * self.BN
            sizes.append(2 * nvis + 1 + (1 if v > 0 else 0) + NLANES + self.D)
        starts = [0]
        for s in sizes:
            starts.append(starts[-1] + s)
        self._sizes, self._starts = sizes, starts
        self.count = starts[-1] + 1  # + INV

    # -- block geometry --------------------------------------------------------------------------------------------
    def block_of_version(self, v: int) -> int:
        return self.nb - 1 - v

    def version_of_block(self, b: int) -> int:
        return self.nb - 1 - b

    def nvis(self, b: int) -> int:
        return min(self.T, (b + 1) * self.BN) - b * self.BN

    def block_span(self, v: int) -> tuple[int, int]:
        """Positions [lo, hi) of the block visited at version v (the epilogue is [count-1, count))."""
        if v == self.nb:
            return self.count - 1, self.count
        return self._starts[v], self._starts[v + 1]

    # -- rank / unrank -----------------------------------------------------------------------------------------------
    def unrank(self, i: int) -> ValueId:
        if not 0 <= i < self.count:
            raise IndexError(i)
        if i == self.count - 1:
            return ValueId(self.layer, self.head, self.qrow, "INV", self.nb, 0, self.nb)
        v = bisect.bisect_right(self._starts, i) - 1
        b = self.block_of_version(v)
        off = i - self._starts[v]
        nvis, c0 = self.nvis(b), b * self.BN
        mk = lambda stage, index: ValueId(self.layer, self.head, self.qrow, stage, b, index, v)  # noqa: E731
        if off < nvis:
            return mk("S", c0 + off)
        off -= nvis
        if off == 0:
            return mk("M", 0)
        off -= 1
        if v > 0:
            if off == 0:
                return mk("ALPHA", 0)
            off -= 1
        if off < nvis:
            return mk("P32", c0 + off)
        off -= nvis
        if off < NLANES:
            return mk("L", off)
        off -= NLANES
        return mk("O", off)

    def rank(self, vid: ValueId) -> int:
        if (vid.layer, vid.head, vid.qrow) != (self.layer, self.head, self.qrow):
            raise KeyError(f"{vid} is not in row (layer={self.layer}, head={self.head}, qrow={self.qrow})")
        if vid.stage == "INV":
            if vid.block != self.nb or vid.version != self.nb or vid.index != 0:
                raise KeyError(vid)
            return self.count - 1
        if vid.stage not in COMMITTED_STAGES:
            raise KeyError(f"{vid.stage} is not a committed stage (derived stages open through their source)")
        b = vid.block
        if not 0 <= b < self.nb or vid.version != self.version_of_block(b):
            raise KeyError(vid)
        v = vid.version
        nvis, c0 = self.nvis(b), b * self.BN
        base = self._starts[v]
        if vid.stage == "S":
            if not c0 <= vid.index < c0 + nvis:
                raise KeyError(vid)
            return base + (vid.index - c0)
        base += nvis
        if vid.stage == "M":
            if vid.index != 0:
                raise KeyError(vid)
            return base
        base += 1
        if v > 0:
            if vid.stage == "ALPHA":
                if vid.index != 0:
                    raise KeyError(vid)
                return base
            base += 1
        elif vid.stage == "ALPHA":
            raise KeyError(f"{vid}: the first visited block has no rescale factor")
        if vid.stage == "P32":
            if not c0 <= vid.index < c0 + nvis:
                raise KeyError(vid)
            return base + (vid.index - c0)
        base += nvis
        if vid.stage == "L":
            if not 0 <= vid.index < NLANES:
                raise KeyError(vid)
            return base + vid.index
        base += NLANES
        if vid.stage == "O":
            if not 0 <= vid.index < self.D:
                raise KeyError(vid)
            return base + vid.index
        raise KeyError(vid)

    def ids(self) -> Iterator[ValueId]:
        for i in range(self.count):
            yield self.unrank(i)


def row_count(T: int, geo: Geometry) -> int:
    """|row| = 2T + nb*(D + 6) words, nb = ceil(T/BN)."""
    nb = (T + geo.BN - 1) // geo.BN
    return 2 * T + nb * (geo.D + 6)


def word_bytes(value: int, width: int) -> bytes:
    return int(value).to_bytes(width, "little")


def word_of(b: bytes) -> int:
    return int.from_bytes(b, "little")
