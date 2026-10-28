"""Dither's C2-O served form: BYTE-CELL's C2-O, a unit kind beside `dither`'s `PouwDitherBlock_v1` and `dither_c2t`'s filler,
with its own shape rule. It is experimental and rests on SAKHigh-O, a claim Daniel hasn't ruled on, so nothing of it is in a
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

**Credit.** The exact route, Strassen–Winograd at A = 0 with its sparse corrections, computes O at pub_O ≈ 0.886 of the int8
GEMM on the H200 with B pinned. So an accepted C2-O call is credited at pub_O, not at the GEMM.

**A proof block** is `CELLS` = 8 adjacent cells (i, j0 .. j0 + 7) of one row, j0 a multiple of 8, `PouwDitherServedC2O_v1{K,
CELLS, R, W}`:

    (a: Array<K, i8>, da: Array<K, bit>, b: Array<CELLS, Array<K, i8>>, db: Array<CELLS, Array<K, bit>>,
     w: Array<CELLS, Array<W, bit>>) -> ok

over A's row, B's CELLS rows, their dithers (public, from the coins) and the CELLS committed windows. It is 1 exactly when
each cell's window of sum_k (a_k + da_k)(b_ck + db_ck) is w_c. The sums run through `dither.PouwDitherOutDot0_v2` and
`PouwDitherOutDot_v2`, and each window bit is checked by `dither_c2t`'s `PouwDitherBitEq_v2` and `PouwDitherBitEqAnd_v2`.
The check of down's A′ beside it (up's cells, through SiLU·mul and the requant) is open.

**The leaves** are `dither_c2t`'s at one segment, under `TAG`: per tile, its cells' W-bit windows packed least significant bit
first, cut into rows of `row` bytes. So a block's windows are W bytes of one row, and `dither_c2t.values_buffer`, `commit`
and `block_words` serve this form too.

`block_definition` is the verifier's dispatch for this form: it takes only a `C2OShape`. `dither.block_definition` takes
only a `dither.Shape`, and `dither_c2t.block_definition` only a `C2TShape`.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import mul
from typing import Any, Mapping, Sequence

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

from . import dither as D
from . import dither_c2t as T

TAG = "verity/pouw/dither/c2o/v1"
CELLS = T.CELLS
#: the bit SAKHigh-O is stated for: a committed window covers it
PREMISE_BIT = 9
BIT = T.BIT
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


def values(sh: C2OShape, seeds: Mapping[str, bytes], A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> list[Rows]:
    """The call's committed windows, as `dither_c2t.values_buffer` takes them (one segment × m × n)."""
    return [[[T.window(x, sh.r, sh.w) for x in row] for row in outputs(sh, seeds, A, B)]]


# -- the Definition --------------------------------------------------------------------------------------------------------------

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
            ok = B.call(T.PouwDitherBitEq, x, y) if ok is None else B.call(T.PouwDitherBitEqAnd, ok, x, y)
    return ok


PouwDitherServedC2O = CompositeDefinition(
    "PouwDitherServedC2O", 1, ("K", "CELLS", "R", "W"), _signature, _body, conformance="protocol-defined",
    doc="Dither's C2-O served proof block (`circuit.dither_c2o`): CELLS adjacent cells of one row of an un-noised call at "
        "S = 1, A's whole row a and B's CELLS rows b (int8 each) with their dithers da and db and the cells' committed "
        "windows w (W bits each), to 1 exactly when, for every cell c, bits R .. R + W - 1 of the exact sum "
        "sum_k (a_k + da_k)(b_ck + db_ck) are w_c.")


def block_definition(sh: C2OShape) -> Any:
    """A C2-O call's proof block Definition: `PouwDitherServedC2O_v1` at the shape's K, `CELLS` and window. Any other form is
    refused: v3.2's blocks are `dither.block_definition`'s and C2-T's `dither_c2t.block_definition`'s."""
    if not isinstance(sh, C2OShape):
        raise ValueError(f"C2-O's blocks are a C2OShape's, not a {type(sh).__name__}'s: v3.2's are `dither.block_definition`'s "
                         f"and C2-T's `dither_c2t.block_definition`'s")
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
        if T.window(sum(map(mul, x, y)), r, w) != want:
            return 0
    return 1


__all__ = ["CELLS", "C2OShape", "PREMISE_BIT", "PouwDitherServedC2O", "TAG", "block_bits", "block_definition", "block_inputs",
           "check_block", "outputs", "values"]
