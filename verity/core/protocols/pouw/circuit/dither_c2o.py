"""Dither's C2-O served form: BYTE-CELL's C2-O, a unit kind beside `dither`'s `PouwDitherBlock_v1`, with its own shape
rule. It is experimental and rests on SAKHigh-O, a claim Daniel hasn't ruled on, so nothing of it is in a
Lean lock.

**A served call** at S = 1 computes O = (A + d_A)(B + d_B)ᵀ over the integers with `dither`'s 1-bit dithers and no noise:
no U, V or π, so no peel. A is in the call's operand root and B is a published checkpoint's s8 weights, which registration
pins by digest (`dither.checkpoint_digest`). Per cell (i, j) the value root holds only the window

    w(i, j) = floor(O_ij / 2^R) mod 2^W      (O at full width, exact and signed)

of O (`C2OShape.r`, `C2OShape.w`; BYTE-CELL's is bit 9 alone, 2 KiB a 128 × 128 tile). Every Definition refuses K at or past
`dither.K_LIMIT`: each product (a + d_a)(b + d_b) is at most 2^14 in magnitude, so O fits s32 and its window is bits
R .. R + W − 1 of O's two's complement.

**The shape rule** (`C2OShape`):

* the window covers bit `PREMISE_BIT` = 9, the bit SAKHigh-O is stated for. It sits above every mod-2^k orbit the H200 runs
  cheaply: mod 2^8 is free and mod 2^9 breaks even through b1 corrections (BYTE-CELL, C2-O (b)). A window inside bits 0–8
  is refused.
* B is pinned (`pinned`). With weights the prover chooses, at A = B = 0, O = d_A d_Bᵀ is one b1 product, and d_A's count
  times B's mean passes 80% of cells (BYTE-CELL, C2-O (b)). So a call on unpinned weights is refused.

**Credit.** The exact route, Strassen at depth 1 with its sparse corrections, computes O at pub_O ≈ 0.90 of the int8 GEMM
on the H200 with B pinned (0.903 on the served Llama 3.1 8B int8 weights; 0.886 was BYTE-CELL's estimate on random B). So an
accepted C2-O call is credited at pub_O, not at the GEMM. pub_O is a public figure of the protocol, not of this Definition.

**A proof block** is `CELLS` = 8 adjacent cells (i, j0 .. j0 + 7) of one row, j0 a multiple of 8, `PouwDitherServedC2O_v1{K,
CELLS, R, W}`:

    (a: Array<K, i8>, da: Array<K, bit>, b: Array<CELLS, Array<K, i8>>, db: Array<CELLS, Array<K, bit>>,
     w: Array<CELLS, Array<W, bit>>) -> ok

over A's row, B's CELLS rows, their dithers (public, from the coins) and the CELLS committed windows. It is 1 exactly when
each cell's window of sum_k (a_k + da_k)(b_ck + db_ck) is w_c. The sums run through `dither.PouwDitherOutDot0_v2` and
`PouwDitherOutDot_v2`, and each window bit is checked by `PouwDitherBitEq_v2` (the block's first) and
`PouwDitherBitEqAnd_v2` (the rest, one AND each).
The check of down's A′ beside it (up's cells, through SiLU·mul and the requant) is open.

**The leaves** (`C2OShape`): per tile (128 × 128 cells, tile tm (n / 128) + tn, cells row-major), its cells' windows packed
W bits a cell, least significant bit first (cell c's at bits [c W, (c + 1) W) of the tile), 2 W KiB a tile; the tiles in
order, cut into rows of `row` bytes, the last zero-padded. A block's windows are W consecutive bytes of one row. Each row
is a `dither.leaf_h` hiding leaf in one binary frame-v3-sha512 tree a call, under `TAG` and salted as `dither.salts`.

`block_definition` is the verifier's dispatch for this form: it takes only a `C2OShape`, and `dither.block_definition` only
a `dither.Shape`, so neither form's proof block is checked by the other's Definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import mul
from typing import Any, Callable, Mapping, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind, primitive
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

from . import dither as D

TAG = "verity/pouw/dither/c2o/v1"
#: the cells of one row a proof block checks
CELLS = 8
#: the bit SAKHigh-O is stated for: a committed window covers it
PREMISE_BIT = 9
BIT = BL.BIT
I8 = D.I8
TILE = D.TILE


# -- the shape -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class C2OShape:
    """A C2-O served call: m rows of A against n rows of the pinned B at contraction k, one segment, the committed window bits
    r .. r + w − 1 of O, leaf rows of `row` bytes; `pinned` is registration's pin of B."""

    m: int
    n: int
    k: int
    r: int = PREMISE_BIT
    w: int = 1
    row: int = 16384
    pinned: bool = True

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.m % TILE or self.n % TILE:
            raise ValueError(f"m and n are whole tiles of {TILE}, not {self.m} × {self.n}")
        if self.k <= 0 or self.k % 32 or self.k >= D.K_LIMIT:
            raise ValueError(f"K is a positive multiple of 32 below {D.K_LIMIT} (`dither.K_LIMIT`), not {self.k}")
        if not (isinstance(self.r, int) and isinstance(self.w, int) and 0 <= self.r <= PREMISE_BIT < self.r + self.w <= 32):
            raise ValueError(f"C2-O's window covers bit {PREMISE_BIT} (`dither_c2o.PREMISE_BIT`) inside 32 bits, not bits "
                             f"{self.r} .. {self.r + self.w - 1}")
        if self.row <= 0 or self.row % 16 or self.row % self.w:
            raise ValueError(f"a row is a multiple of 16 bytes and of W = {self.w} (a block's bytes), not {self.row}")
        if self.pinned is not True:
            raise ValueError("C2-O needs B pinned by registration (`dither.checkpoint_digest`): on weights the prover chooses, "
                             "O = d_A d_Bᵀ at A = B = 0 is one b1 product")

    S = 1

    @property
    def tiles(self) -> int:
        return (self.m // TILE) * (self.n // TILE)

    @property
    def tile_bytes(self) -> int:
        """A tile's bytes: its 16,384 cells' windows, W bits each."""
        return D.CELLS * self.w // 8

    @property
    def leaves(self) -> int:
        return -(-self.tiles * self.tile_bytes // self.row)

    def label(self) -> dict:
        return {"m": self.m, "n": self.n, "k": self.k, "r": self.r, "w": self.w, "cells": CELLS, "row": self.row,
                "leaves": self.leaves}

    def domain(self) -> CommitmentDomain:
        return CommitmentDomain(identity_digest_sha512(TAG, self.label()), -1, RangeIndexedDomain(0, self.leaves),
                                hash="sha512")

    def block(self, i: int, j0: int) -> tuple[int, int]:
        """(leaf position, byte offset in its row) of proof block (i, j0 .. j0 + 7)'s W bytes of windows."""
        if not (0 <= i < self.m and 0 <= j0 < self.n) or j0 % CELLS:
            raise ValueError(f"a block is {CELLS} cells of one row from a column that is a multiple of {CELLS}, not ({i}, {j0}) "
                             f"in {self.m} × {self.n}")
        t, c = (i // TILE) * (self.n // TILE) + j0 // TILE, (i % TILE) * TILE + j0 % TILE
        return divmod(t * self.tile_bytes + c * self.w // 8, self.row)


# -- the values ------------------------------------------------------------------------------------------------------------------

Rows = list[list[int]]


def window(x: int, r: int, w: int) -> int:
    """floor(x / 2^r) mod 2^w of a signed integer x."""
    return (x >> r) & ((1 << w) - 1)


def _dithered(sh: C2OShape, seeds: Mapping[str, bytes], rows: Sequence[Sequence[int]], side: str,
              idx: Sequence[int]) -> Rows:
    return [[x + d for x, d in zip(D._row(f"{side}'s row {r}", v, sh.k), ds)]
            for r, v, ds in zip(idx, rows, D.dithers(sh, seeds, side, idx))]


def outputs(sh: C2OShape, seeds: Mapping[str, bytes], A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> Rows:
    """O = (A + d_A)(B + d_B)ᵀ (m × n, exact and signed) on A (m × K int8) and B (n × K int8), from a served call's
    `dither.seeds`."""
    if len(A) != sh.m or len(B) != sh.n:
        raise ValueError(f"A × B is m × n = {sh.m} × {sh.n} rows, not {len(A)} × {len(B)}")
    Ad, Bd = _dithered(sh, seeds, A, "A", range(sh.m)), _dithered(sh, seeds, B, "B", range(sh.n))
    return [[sum(map(mul, x, y)) for y in Bd] for x in Ad]


def values(sh: C2OShape, seeds: Mapping[str, bytes], A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> Rows:
    """The call's committed windows (m × n): `window` of each `outputs` entry at the shape's (r, w)."""
    return [[window(x, sh.r, sh.w) for x in row] for row in outputs(sh, seeds, A, B)]


def values_buffer(sh: C2OShape, V: Sequence[Sequence[int]]) -> bytes:
    """The call's leaf rows end to end: per tile, its cells' windows, W bits a cell, least significant first, cells
    row-major, the last row zero-padded."""
    bits = bytearray(sh.tiles * D.CELLS * sh.w)
    p = 0
    for tm in range(sh.m // TILE):
        for tn in range(sh.n // TILE):
            for r in range(tm * TILE, (tm + 1) * TILE):
                for v in V[r][tn * TILE:(tn + 1) * TILE]:
                    for e in range(sh.w):
                        bits[p] = (int(v) >> e) & 1
                        p += 1
    out = bytearray(sh.leaves * sh.row)
    for q, b in enumerate(bits):
        out[q >> 3] |= b << (q & 7)
    return bytes(out)


def commit(sh: C2OShape, buf: bytes, salt: Callable[[int], bytes]) -> D.Tree:
    """The call's windows (`values_buffer`) under its value root."""
    if len(buf) != sh.leaves * sh.row:
        raise ValueError(f"the call's windows are {sh.leaves * sh.row} bytes, not {len(buf)}")
    return D.tree(sh.domain(), [buf[p * sh.row:(p + 1) * sh.row] for p in range(sh.leaves)], salt)


def block_words(sh: C2OShape, row: bytes, off: int) -> list[int]:
    """The CELLS windows a block's W bytes hold, at byte `off` of its opened row."""
    x = int.from_bytes(row[off:off + sh.w], "little")
    return [(x >> (c * sh.w)) & ((1 << sh.w) - 1) for c in range(CELLS)]


# -- the Definitions -------------------------------------------------------------------------------------------------------------

@primitive("PouwDitherBitEq", 1, [("x", Value(1)), ("w", Value(1))], Value(1), conformance="protocol-defined")
def PouwDitherBitEqWord(x: int, w: int) -> int:
    """1 when x = w: a block's first window bit."""
    return int((x ^ w) & 1 == 0)


@primitive("PouwDitherBitEqAnd", 1, [("ok", Value(1)), ("x", Value(1)), ("w", Value(1))], Value(1),
           conformance="protocol-defined")
def PouwDitherBitEqAndWord(ok: int, x: int, w: int) -> int:
    """1 when ok and x = w: the block's pass bit so far, and one more window bit."""
    return int(ok & 1 and (x ^ w) & 1 == 0)


PouwDitherBitEq = definition("PouwDitherBitEq", 2, lambda C, x, w: [NOT(x[0] ^ w[0])], (("x", BIT), ("w", BIT)), BIT,
                             word=PouwDitherBitEqWord, conformance="protocol-defined",
                             doc="x = w on bits: PouwDitherBitEq_v1")
PouwDitherBitEqAnd = definition("PouwDitherBitEqAnd", 2, lambda C, ok, x, w: [C.AND(ok[0], NOT(x[0] ^ w[0]))],
                                (("ok", BIT), ("x", BIT), ("w", BIT)), BIT, word=PouwDitherBitEqAndWord,
                                conformance="protocol-defined", doc="ok and x = w on bits: PouwDitherBitEqAnd_v1")


def _signature(S: Any) -> tuple:
    D._length("PouwDitherServedC2O", "K", S.K)
    for what in ("CELLS", "R", "W"):
        if not isinstance(getattr(S, what), int) or isinstance(getattr(S, what), bool):
            raise ValueError(f"PouwDitherServedC2O: {what} is an int, not {getattr(S, what)!r}")
    if S.CELLS < 1 or S.W < 1 or S.R < 0 or S.R + S.W > 32:
        raise ValueError(f"PouwDitherServedC2O: CELLS ≥ 1 and a window of W ≥ 1 bits from R ≥ 0 inside 32, not CELLS "
                         f"{S.CELLS}, R {S.R}, W {S.W}")
    return (("a", Array(S.K, I8)), ("da", Array(S.K, BIT)), ("b", Array(S.CELLS, Array(S.K, I8))),
            ("db", Array(S.CELLS, Array(S.K, BIT))), ("w", Array(S.CELLS, Array(S.W, BIT)))), BIT


def _body(B: Any, S: Any, a, da, b, db, w) -> Any:
    ok = None
    for c in range(S.CELLS):
        acc = B.call(D.PouwDitherOutDot0, a[0:32], da[0:32], b[c][0:32], db[c][0:32])
        for q in range(32, S.K, 32):
            acc = B.call(D.PouwDitherOutDot, acc, a[q:q + 32], da[q:q + 32], b[c][q:q + 32], db[c][q:q + 32])
        for e in range(S.W):
            x, y = acc[S.R + e], w[c][e]
            ok = B.call(PouwDitherBitEq, x, y) if ok is None else B.call(PouwDitherBitEqAnd, ok, x, y)
    return ok


PouwDitherServedC2O = CompositeDefinition(
    "PouwDitherServedC2O", 1, ("K", "CELLS", "R", "W"), _signature, _body, conformance="protocol-defined",
    doc="Dither's C2-O served proof block (`circuit.dither_c2o`): CELLS adjacent cells of one row of an un-noised call at "
        "S = 1, A's whole row a and B's CELLS rows b (int8 each) with their dithers da and db and the cells' committed "
        "windows w (W bits each), to 1 exactly when, for every cell c, bits R .. R + W - 1 of the exact sum "
        "sum_k (a_k + da_k)(b_ck + db_ck) are w_c.")


def block_definition(sh: C2OShape) -> Any:
    """A C2-O call's proof block Definition: `PouwDitherServedC2O_v1` at the shape's K, `CELLS` and window. Any other form is
    refused: v3.2's blocks are `dither.block_definition`'s."""
    if not isinstance(sh, C2OShape):
        raise ValueError(f"C2-O's blocks are a C2OShape's, not a {type(sh).__name__}'s: v3.2's are `dither.block_definition`'s")
    return bind(PouwDitherServedC2O, K=sh.k, CELLS=CELLS, R=sh.r, W=sh.w)


# -- a proof block in the reference ----------------------------------------------------------------------------------------------

def block_inputs(sh: C2OShape, seeds: Mapping[str, bytes], a_i: Sequence[int], b_rows: Sequence[Sequence[int]], i: int,
                 j0: int, words: Sequence[int]) -> list[list[int]]:
    """Proof block (i, j0 .. j0 + 7)'s inputs to `block_definition`, as word lists: a and da over the whole row, b and db (the
    CELLS rows end to end), and the CELLS committed windows."""
    sh.block(i, j0)
    if len(b_rows) != CELLS or len(words) != CELLS:
        raise ValueError(f"a block reads {CELLS} rows of B and {CELLS} windows, not {len(b_rows)} and {len(words)}")
    jj = range(j0, j0 + CELLS)
    a, bs = D._row(f"A's row {i}", a_i, sh.k), [D._row(f"B's row {j}", v, sh.k) for j, v in zip(jj, b_rows)]
    da, db = D.dithers(sh, seeds, "A", [i])[0], D.dithers(sh, seeds, "B", jj)
    return [D._int8s(a), da, [x for v in bs for x in D._int8s(v)], [d for ds in db for d in ds],
            [int(x) & ((1 << sh.w) - 1) for x in words]]


def block_bits(ins: Sequence[Sequence[int]], w: int) -> list[list[int]]:
    """`block_inputs` on bits, as the Definition reads them (each word least significant bit first)."""
    return [[(v >> e) & 1 for v in words for e in range(n)] for words, n in zip(ins, (8, 1, 8, 1, w), strict=True)]


def check_block(ins: Sequence[Sequence[int]], r: int, w: int) -> int:
    """`PouwDitherServedC2O`'s pass bit on `block_inputs`, in the clear."""
    a, da, b, db, words = ins
    K = len(a)
    x = [D._sx8(p) + (q & 1) for p, q in zip(a, da, strict=True)]
    for c, want in enumerate(words):
        y = [D._sx8(p) + (q & 1) for p, q in zip(b[c * K:(c + 1) * K], db[c * K:(c + 1) * K], strict=True)]
        if window(sum(map(mul, x, y)), r, w) != want:
            return 0
    return 1


__all__ = ["CELLS", "C2OShape", "PREMISE_BIT", "PouwDitherBitEq", "PouwDitherBitEqAnd", "PouwDitherServedC2O", "TAG",
           "block_bits", "block_definition", "block_inputs", "block_words", "check_block", "commit", "outputs", "values",
           "values_buffer", "window"]
