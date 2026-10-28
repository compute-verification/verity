"""Dither's C2-T filler form: BYTE-CELL's C2 as filler, a unit kind beside `dither`'s served `PouwDitherBlock_v1`, with its
own shape rule. It is experimental and rests on SAKHigh-T, a claim Daniel hasn't ruled on, so nothing of it is in a Lean
lock.

**A filler call** is Dither's noised product on a filler's operands (`dither`'s docstring): A the call's pad
(`dither.pad_rows` of `dither.fill_seeds`), B n × K int8, X = wrap8(A + d_A + V) and W′ = wrap8(B + d_B + U π_τ) with
`dither`'s coins and HYB noise of period L′. It commits no O and no 16-bit word: per cell (i, j) and segment s it commits
only the window

    w_s(i, j) = floor(T_s / 2^R) mod 2^W,    T_s = sum over segment s's inner indices k of X_ik W′_jk   (exact, signed)

of the segment's own sum (not a prefix sum), R and W the form's window (`C2TShape.r`, `C2TShape.w`). Every Definition
refuses K at or past `dither.K_LIMIT`, so T_s fits s32 and its window is bits R .. R + W − 1 of T_s's two's complement.

**The shape rule** (`C2TShape`), not the served unit's t ≤ 8, which `dither.Shape` keeps:

* t = min(L, K) / L′ ≤ `T_MAX` = 16. The separation route T = d_A(B + d_B)ᵀ + Z Uᵀ + V Y′ᵀ costs 0.886 + 3/t of the int8
  GEMM at A = 0, so it reaches the GEMM at t ≈ 26; SAKHigh-T is stated at t ≤ 16 (BYTE-CELL, C2 (b)).
* the window covers bit `PREMISE_BIT` = 9, the bit SAKHigh-T is stated for. A prover that passes a window covering it
  outputs that bit, so no wider window is weaker. A window inside bits 0–8 is refused: the orbit over Z/2^8, with its mod-2
  corrections at bit 8, passes it with probability 1 (BYTE-CELL, C1).
* segments are equal: L ≥ K (one segment, BYTE-CELL's form) or L divides K.

**A proof block** is `CELLS` = 8 adjacent cells (i, j0 .. j0 + 7) of one row and one segment, j0 a multiple of 8: one-cell
blocks leave the free guess at 2^-W a block, and 8 cells put every measured route under it (BYTE-CELL, "The O(1/K) bytes
a cell question"). `PouwDitherFillC2T_v1{L, CELLS, R, W}`:

    (a, na: Array<L, i8>, b, nb: Array<CELLS, Array<L, i8>>, w: Array<CELLS, Array<W, bit>>) -> ok

over A's row segment, B's CELLS row segments, their noise (public, from the coins, as in `dither`) and the CELLS committed
windows, 1 exactly when each cell's window of sum_k wrap8(a_k + na_k) wrap8(b_ck + nb_ck) is w_c. X is formed once and
shared by the block's cells. The sums run through `segment.PouwSegDot0_v2` and `PouwSegDot_v2`, and each window bit is
checked by `PouwDitherBitEq_v2` (the block's first) and `PouwDitherBitEqAnd_v2` (the rest, one AND each).

**The leaves** (`C2TShape`): per tile (128 × 128 cells, tile tm (n / 128) + tn, cells row-major) and segment, its cells'
windows packed W bits a cell, least significant bit first (cell c's at bits [c W, (c + 1) W) of the tile-segment), 2 W KiB
a tile-segment; the tile-segments in order (tile, segment), cut into rows of `row` bytes, the last zero-padded. A block's
windows are W consecutive bytes of one row. Each row is a `dither.leaf_h` hiding leaf in one binary frame-v3-sha512 tree a
call, under `TAG` and salted as `dither.salts`.

`block_definition` is the verifier's dispatch for this form: it takes only a `C2TShape`, and `dither.block_definition` only a
`dither.Shape`, so neither form's proof block is checked by the other's Definition.
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
from . import segment as SG

TAG = "verity/pouw/dither/c2t/v1"
#: the cells of one row a proof block checks
CELLS = 8
#: the bit SAKHigh-T is stated for: a committed window covers it
PREMISE_BIT = 9
#: the most noise tiles a segment
T_MAX = 16
BIT = BL.BIT
I8 = D.I8
TILE = D.TILE


# -- the shape -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class C2TShape:
    """A C2-T filler call: m pad rows against n rows of B at contraction k, HYB noise of period lp (L′), segments of L inner
    indices (None: one segment, L = K), the committed window bits r .. r + w − 1 of each segment's sum, leaf rows of `row`
    bytes."""

    m: int
    n: int
    k: int
    lp: int
    L: int | None = None
    r: int = PREMISE_BIT
    w: int = 1
    row: int = 16384

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.m % TILE or self.n % TILE:
            raise ValueError(f"m and n are whole tiles of {TILE}, not {self.m} × {self.n}")
        if self.lp <= 0 or self.lp % 32 or self.k % self.lp:
            raise ValueError(f"L′ is a multiple of 32 dividing K, not L′ {self.lp} at K {self.k}")
        if self.k >= D.K_LIMIT:
            raise ValueError(f"K is below {D.K_LIMIT} (`dither.K_LIMIT`), not {self.k}")
        if self.L is not None and (self.L <= 0 or self.L % self.lp or (self.L < self.k and self.k % self.L)):
            raise ValueError(f"L is t L′ (L′ {self.lp}) and at least K or dividing it (K {self.k}), not {self.L}")
        if self.seg_len > T_MAX * self.lp:
            raise ValueError(f"C2-T's t = min(L, K) / L′ is at most {T_MAX} (`dither_c2t.T_MAX`, SAKHigh-T's range), not "
                             f"{self.seg_len} / {self.lp}")
        if not (isinstance(self.r, int) and isinstance(self.w, int) and 0 <= self.r <= PREMISE_BIT < self.r + self.w <= 32):
            raise ValueError(f"C2-T's window covers bit {PREMISE_BIT} (`dither_c2t.PREMISE_BIT`) inside 32 bits, not bits "
                             f"{self.r} .. {self.r + self.w - 1}")
        if self.row <= 0 or self.row % 16 or self.row % self.w:
            raise ValueError(f"a row is a multiple of 16 bytes and of W = {self.w} (a block's bytes), not {self.row}")

    @property
    def seg_len(self) -> int:
        return self.k if self.L is None else min(self.L, self.k)

    @property
    def t(self) -> int:
        return self.seg_len // self.lp

    @property
    def S(self) -> int:
        return self.k // self.seg_len

    @property
    def tiles(self) -> int:
        return (self.m // TILE) * (self.n // TILE)

    @property
    def tile_bytes(self) -> int:
        """A tile-segment's bytes: its 16,384 cells' windows, W bits each."""
        return D.CELLS * self.w // 8

    @property
    def leaves(self) -> int:
        return -(-self.tiles * self.S * self.tile_bytes // self.row)

    def seg(self, s: int) -> tuple[int, int]:
        """Segment s's inner range [a, b) (s = 1 .. S)."""
        if not 1 <= s <= self.S:
            raise ValueError(f"segment {s} is not one of 1 .. {self.S}")
        return (s - 1) * self.seg_len, s * self.seg_len

    def label(self) -> dict:
        return {"m": self.m, "n": self.n, "k": self.k, "lp": self.lp, "t": self.t, "r": self.r, "w": self.w,
                "cells": CELLS, "row": self.row, "leaves": self.leaves}

    def domain(self) -> CommitmentDomain:
        return CommitmentDomain(identity_digest_sha512(TAG, self.label()), -1, RangeIndexedDomain(0, self.leaves),
                                hash="sha512")

    def block(self, i: int, j0: int, s: int) -> tuple[int, int]:
        """(leaf position, byte offset in its row) of proof block (i, j0 .. j0 + 7, s)'s W bytes of windows."""
        self.seg(s)
        if not (0 <= i < self.m and 0 <= j0 < self.n) or j0 % CELLS:
            raise ValueError(f"a block is {CELLS} cells of one row from a column that is a multiple of {CELLS}, not ({i}, {j0}) "
                             f"in {self.m} × {self.n}")
        t, c = (i // TILE) * (self.n // TILE) + j0 // TILE, (i % TILE) * TILE + j0 % TILE
        off = (t * self.S + s - 1) * self.tile_bytes + c * self.w // 8
        return divmod(off, self.row)


# -- the values ------------------------------------------------------------------------------------------------------------------

Rows = list[list[int]]


def window(x: int, r: int, w: int) -> int:
    """floor(x / 2^r) mod 2^w of a signed integer x."""
    return (x >> r) & ((1 << w) - 1)


def _operands(sh: C2TShape, seeds: Mapping[str, bytes], rows: Sequence[Sequence[int]], side: str,
              idx: Sequence[int]) -> Rows:
    noise = D.noise_a(sh, seeds, idx) if side == "A" else D.noise_b(sh, seeds, idx)
    return [[D.wrap8(x + y) for x, y in zip(D._row(f"{side}'s row {r}", v, sh.k), n)] for r, v, n in zip(idx, rows, noise)]


def sums(sh: C2TShape, seeds: Mapping[str, bytes], A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> list[Rows]:
    """Each segment's noised sum T_s (S × m × n, as T[s − 1][i][j], exact and signed) on A (m × K int8, a filler's
    `dither.pad_rows`) and B (n × K int8)."""
    if len(A) != sh.m or len(B) != sh.n:
        raise ValueError(f"A × B is m × n = {sh.m} × {sh.n} rows, not {len(A)} × {len(B)}")
    X, W = _operands(sh, seeds, A, "A", range(sh.m)), _operands(sh, seeds, B, "B", range(sh.n))
    return [[[sum(map(mul, x[a:b], y[a:b])) for y in W] for x in X] for a, b in map(sh.seg, range(1, sh.S + 1))]


def values(sh: C2TShape, seeds: Mapping[str, bytes], A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> list[Rows]:
    """The call's committed windows (S × m × n): `window` of each `sums` entry at the shape's (r, w)."""
    return [[[window(x, sh.r, sh.w) for x in row] for row in T] for T in sums(sh, seeds, A, B)]


def values_buffer(sh: C2TShape, V: Sequence[Sequence[Sequence[int]]]) -> bytes:
    """The call's leaf rows end to end: per tile and segment, its cells' windows, W bits a cell, least significant first,
    cells row-major, the last row zero-padded."""
    bits = bytearray(sh.tiles * sh.S * D.CELLS * sh.w)
    p = 0
    for tm in range(sh.m // TILE):
        for tn in range(sh.n // TILE):
            for s in range(sh.S):
                for r in range(tm * TILE, (tm + 1) * TILE):
                    for v in V[s][r][tn * TILE:(tn + 1) * TILE]:
                        for e in range(sh.w):
                            bits[p] = (int(v) >> e) & 1
                            p += 1
    out = bytearray(sh.leaves * sh.row)
    for q, b in enumerate(bits):
        out[q >> 3] |= b << (q & 7)
    return bytes(out)


def commit(sh: C2TShape, buf: bytes, salt: Callable[[int], bytes]) -> D.Tree:
    """The call's windows (`values_buffer`) under its value root."""
    if len(buf) != sh.leaves * sh.row:
        raise ValueError(f"the call's windows are {sh.leaves * sh.row} bytes, not {len(buf)}")
    return D.tree(sh.domain(), [buf[p * sh.row:(p + 1) * sh.row] for p in range(sh.leaves)], salt)


def block_words(sh: C2TShape, row: bytes, off: int) -> list[int]:
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


def _statics(S: Any) -> None:
    D._length("PouwDitherFillC2T", "L", S.L)
    for what in ("CELLS", "R", "W"):
        if not isinstance(getattr(S, what), int) or isinstance(getattr(S, what), bool):
            raise ValueError(f"PouwDitherFillC2T: {what} is an int, not {getattr(S, what)!r}")
    if S.CELLS < 1 or S.W < 1 or S.R < 0 or S.R + S.W > 32:
        raise ValueError(f"PouwDitherFillC2T: CELLS ≥ 1 and a window of W ≥ 1 bits from R ≥ 0 inside 32, not CELLS {S.CELLS}, "
                         f"R {S.R}, W {S.W}")


def _signature(S: Any) -> tuple:
    _statics(S)
    seg, rows = Array(S.L, I8), Array(S.CELLS, Array(S.L, I8))
    return (("a", seg), ("na", seg), ("b", rows), ("nb", rows), ("w", Array(S.CELLS, Array(S.W, BIT)))), BIT


def _body(B: Any, S: Any, a, na, b, nb, w) -> Any:
    dot0, dot = SG.PouwSegDot0, SG.PouwSegDot
    X = B.batch(SG.PouwI8Add, a, na, axes=(0, 0))
    ok = None
    for c in range(S.CELLS):
        Wc = B.batch(SG.PouwI8Add, b[c], nb[c], axes=(0, 0))
        acc = B.call(dot0, X[0:32], Wc[0:32])
        for q in range(32, S.L, 32):
            acc = B.call(dot, acc, X[q:q + 32], Wc[q:q + 32])
        for e in range(S.W):
            x, y = acc[S.R + e], w[c][e]
            ok = B.call(PouwDitherBitEq, x, y) if ok is None else B.call(PouwDitherBitEqAnd, ok, x, y)
    return ok


PouwDitherFillC2T = CompositeDefinition(
    "PouwDitherFillC2T", 1, ("L", "CELLS", "R", "W"), _signature, _body, conformance="protocol-defined",
    doc="Dither's C2-T filler proof block (`circuit.dither_c2t`): CELLS adjacent cells of one row and one segment of L inner "
        "indices, A's row segment a and B's CELLS row segments b with their noise na and nb (int8 each) and the cells' "
        "committed windows w (W bits each), to 1 exactly when, for every cell c, bits R .. R + W - 1 of the exact sum "
        "sum_k wrap8(a_k + na_k) wrap8(b_ck + nb_ck) are w_c.")


def block_definition(sh: C2TShape, s: int) -> Any:
    """Proof block (·, ·, s)'s Definition for a C2-T filler call: `PouwDitherFillC2T_v1` at the shape's segment, `CELLS` and
    window. Any other form is refused: a served call's blocks are `dither.block_definition`'s."""
    if not isinstance(sh, C2TShape):
        raise ValueError(f"C2-T's blocks are a C2TShape's, not a {type(sh).__name__}'s: a served call's are "
                         f"`dither.block_definition`'s")
    a, b = sh.seg(s)
    return bind(PouwDitherFillC2T, L=b - a, CELLS=CELLS, R=sh.r, W=sh.w)


# -- a proof block in the reference ----------------------------------------------------------------------------------------------

def block_inputs(sh: C2TShape, seeds: Mapping[str, bytes], a_i: Sequence[int], b_rows: Sequence[Sequence[int]], i: int,
                 j0: int, s: int, words: Sequence[int]) -> list[list[int]]:
    """Proof block (i, j0 .. j0 + 7, s)'s inputs to `block_definition`, as word lists: a and na over the segment, b and nb
    (the CELLS rows' segments end to end), and the CELLS committed windows."""
    sh.block(i, j0, s)
    if len(b_rows) != CELLS or len(words) != CELLS:
        raise ValueError(f"a block reads {CELLS} rows of B and {CELLS} windows, not {len(b_rows)} and {len(words)}")
    lo, hi = sh.seg(s)
    jj = range(j0, j0 + CELLS)
    a, bs = D._row(f"A's row {i}", a_i, sh.k), [D._row(f"B's row {j}", v, sh.k) for j, v in zip(jj, b_rows)]
    na, nb = D.noise_a(sh, seeds, [i])[0], D.noise_b(sh, seeds, jj)
    return [D._int8s(a[lo:hi]), D._int8s(na[lo:hi]), [x for v in bs for x in D._int8s(v[lo:hi])],
            [x for v in nb for x in D._int8s(v[lo:hi])], [int(x) & ((1 << sh.w) - 1) for x in words]]


def block_bits(ins: Sequence[Sequence[int]], w: int) -> list[list[int]]:
    """`block_inputs` on bits, as the Definition reads them (each word least significant bit first)."""
    return [[(v >> e) & 1 for v in words for e in range(n)] for words, n in zip(ins, (8, 8, 8, 8, w), strict=True)]


def check_block(ins: Sequence[Sequence[int]], r: int, w: int) -> int:
    """`PouwDitherFillC2T`'s pass bit on `block_inputs`, in the clear."""
    a, na, b, nb, words = ins
    L = len(a)
    x = [D._sx8(p + q) for p, q in zip(a, na, strict=True)]
    for c, want in enumerate(words):
        y = [D._sx8(p + q) for p, q in zip(b[c * L:(c + 1) * L], nb[c * L:(c + 1) * L], strict=True)]
        if window(sum(map(mul, x, y)), r, w) != want:
            return 0
    return 1


__all__ = ["CELLS", "C2TShape", "PREMISE_BIT", "PouwDitherBitEq", "PouwDitherBitEqAnd", "PouwDitherFillC2T", "TAG", "T_MAX",
           "block_bits", "block_definition", "block_inputs", "block_words", "check_block", "commit", "sums", "values",
           "values_buffer", "window"]
