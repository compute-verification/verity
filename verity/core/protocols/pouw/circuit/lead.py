"""The lead's sampled unit (`verity/pouw/v2`): one 16 × 8 MMA tile of one call over one block of at most 32 k32 steps, its
committed words checked against the call's data and the certifier's coins, and on the call's last block the served fold,
at the layout of `Security/Definitions/Pouw/Lead/Layout.lean` (`Pouw.Lead`).

The lead is `tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min`. A call of ``rows`` × K × ``cols`` commits its words over
``m = pad_rows(rows)`` rows and ``n = pad_cols(cols)`` columns (Lean's `padRows`, `padCols`), at S = `steps_for`(K) k32
steps; a padding row has zero data and its tags, coin lanes and coins like any row, a padding column zero data and its
coins. Word (i, t, j) is Lean's `word`:

    word(i, t, j) = -β_j + Σ_{τ ≤ t} (κ(i, τ, j) + coinTerm(i, τ, j))

with κ the data's step dot product and the coin term step 0's T(i)·G[:, j] + a₁[i]·C[:, j] or a later step's 1 + b_t[i],
plus the step's three coin lanes Σ_ℓ A′[i, ℓ] s_t[⌊i/64⌋, ℓ, j], A′ = (127, 1, 1) on even rows and (1, 127, 1) on odd rows.
The unit computes each word as the kernel does, one k32 step of 32 formed lanes:

    step 0   A′ lanes 0-20 A[i, 0..20], 21-23 a₁[i, 0..2], 24-27 T(i), 28-30 rotWeight(i), 31 a₁[i, 3]
             B′ lanes 0-20 B[j, 0..20], 21-23 C[0..2, j], 24-27 G[:, j], 28-30 s_0[·, j],     31 C[3, j]
    t >= 1   A′ lanes 0-27 the step's 28 entries of A, 28-30 rotWeight(i), 31 1 + b_t[i]
             B′ lanes 0-27 the step's 28 entries of B, 28-30 s_t[·, j],     31 1

data lanes past K zero, so Σ_l A′_l B′_l is κ(i, t, j) + coinTerm(i, t, j) (`Call.words`).

**The unit** `PouwLeadTile_v1{K, R0, C0, ROWS, COLS, BLK}` is the tile of rows R0 .. R0 + 15 and columns C0 .. C0 + 7
(16 | R0, 8 | C0), ROWS and COLS of them the call's own, over steps [32 BLK, min(32 BLK + 32, S)). Its inputs, every one a
value, are `Unit.params`: A's ROWS rows and B's COLS rows at the block's K entries; the tile's coins as the certifier's
feed encodes them: its 16 rows' row-word bits at the block (bit q < 4 a₁[i, q], bit t + 3 b_t[i]; on block 0 and on the
call's last block bits 0 .. t₁ + 2, on a block between them the block's steps' bits), on block 0 and on the last block
its 8 columns' G words (byte q the tag weight G[q, j] as an int8), and the coin bytes of its coin tile's lane-coin words
(bytes 0-2 s_t[ℓ, ⌊i/64⌋, j] as int8s, by step, column and lane) at the block's steps, at every step on the last block;
on a block after the first the tile's committed words at the step before it; and the block's committed words, each
step's 128 at Lean's `tileOf` position 8 r + c. They are Layout.lean's `Coins` (g = G + 3, u = (s + 127) / 2) exactly
when every G is in [-3, 3] and every s odd, which the unit checks. It returns one bit, ok, a proof block's outputs being
at most 16 bits:

* **each step** (`PouwLeadStep0`, `PouwLeadStep`): ``w_t ≡ w_{t-1} + Σ_l A′_l B′_l (mod 2^32)``, w_{-1} = -β_j the
  public bias, so the committed words are Lean's read as s32 (each fits s32 up to `MAX_K`, Lean's `Tags.word_exact`);
* **the coins** (`PouwLeadTagWeight`): every G it reads in [-3, 3], `Coins`' Fin 7 less 3, and every s odd, `Coins`'
  Fin 128 as 2u - 127;
* **the fold** (`PouwLeadFold`), on the call's last block, at each of the call's cells: the kernel's epilogue
  (`layout.h`'s `fold_word`) adds to the final word d its row source r_i = 0x4B400000 - walk_i, walk_i = Σ_{t >= 1}
  (1 + b_t[i]) the tag lanes, and its column source c_j = β_j - σ, σ = Σ_t Σ_ℓ A′[i, ℓ] s_t[ℓ, j] the row class's coin
  lanes, in one IADD3, then takes a₁[i] · C[:, j] and T(i) · G[:, j] out by two IDP4A (the bytes -a₁[i], -T(i) against
  C[:, j], G[:, j]). Every coin term leaves, so q = 0x4B400000 + p with p = A_i · B_j mod 2^32, and the epilogue serves
  FMUL s_j (FMUL s_i (FADD(q as fp32, -1.5 · 2^23))): verity_int8's epilogue on A_i · B_j bit for bit while
  |p| <= `LEAD_FOLD_BOUND` = 2^22 (`fold_epilogue`, `int8_epilogue`), and past it another float (2^22 + 1 reads back as
  2^22 + 2) that nothing in the kernel flags. The unit computes r_i (`PouwLeadPop`, `PouwLeadSub32`) and c_j
  (`PouwLeadSub32`) from the coins it reads and checks |p| <= 2^22, so a call with a served word past it is refused as a
  visible failure (`Call.refuse`, `fold_admits`): `Layout.Admits`' counterpart on the private words, which the verifier
  makes through the unit's ok.

ok is the AND of those bits (`verity.ml.circuits.boolean.and_gate`). The tags, codes, biases and row weights are public
constants of the tile's position, and the coins enter as the feed sends them. A cell's step multiplies only its lanes
that vary with both its row and its column: at step 0 the 21 data lanes, a₁[i, q]·C[q, j] and T(i)_q·G[q, j], 29 in all
(`PouwLeadStep0`), and at a later step its 28 data lanes (`PouwLeadStep`), which adds the tag lane 1 + b_t[i] as an
addend; a padding cell's data lanes are zero, so its step multiplies only step 0's 8 coin lanes (`PouwLeadPad0`) and a
later step none (`PouwLeadPad`). The coin lanes' term 127 s_t[i mod 2] + s_t[1 - i mod 2] + s_t[2] depends only on the
row's class and the column, so each step's 16 are `PouwLeadWeigh` once and each cell adds its own. On a tile of the
call's cells alone a step is `PouwLeadStep0Mma` or `PouwLeadStepMma`, 16 rows (`PouwLeadStep0Row`, `PouwLeadStepRow`) of
8 cells each, and on a ragged tile each cell's own piece, batched; the fold's σ is `PouwLeadWeigh` of the three lanes'
sums, trees of `PouwLeadPairS8` then `PouwLeadPair32` over each lane's coins, and each row's walk `PouwLeadPop` of its
tag bits 32 at a time, a tree of `PouwLeadPair32` over them. Each piece's ANDs are `PIECE_ANDS`, which its pin holds,
and a unit's are `unit_ands`.

**Refused** (`bind`, `Unit.refuse`): what `Statement.admits_of_caps` doesn't cover, K outside [1, `MAX_K`], rows past
`MAX_ROWS` or columns past `MAX_COLS`; a tile off the 16 × 8 grid; ROWS outside [1, 16] or COLS outside [1, 8]; and a
block past the call's steps. `Call` is the reference: a call's data and coins, its words, its served bf16s, and a unit's
inputs and verdict.
"""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ZERO, add, compress_columns, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind, primitive
from verity.ml.circuits.refs import Affine, Coll, Strided, concat
from verity.ml.circuits.types import Array, Value

__all__ = [
    "MAX_K", "MAX_ROWS", "MAX_COLS", "LEAD_FOLD_BOUND", "FOLD_MAGIC", "PIECE_ANDS", "PIECES", "Unit", "Call", "unit_ands",
    "steps_for", "pad_rows", "pad_cols", "k_range", "bal13", "row_tag", "col_code", "rot_weight", "bias", "word_bound",
    "fold_word", "fold_admits", "f32", "bf16", "int8_epilogue", "fold_epilogue",
    "PouwLeadStep0", "PouwLeadStep", "PouwLeadPad0", "PouwLeadPad", "PouwLeadWeigh", "PouwLeadTagWeight", "PouwLeadPairS8",
    "PouwLeadPair32", "PouwLeadPop", "PouwLeadSub32", "PouwLeadFold", "PouwLeadStep0Row", "PouwLeadStep0Mma",
    "PouwLeadStepRow", "PouwLeadStepMma", "PouwLeadTile",
]

BIT = BL.BIT
I8, U32 = Array(8, BIT), Array(32, BIT)
M32 = (1 << 32) - 1
#: an MMA tile, Lean's `tileOf`: 16 rows × 8 columns, word (r, c) at position 8 r + c
TM, TN = 16, 8
TILE_WORDS = TM * TN
#: k32 steps a unit
BLOCK = 32
#: rows a coin tile (Lean's `tile`), bits a lane coin (`e`) and coin lanes a step
COIN_TILE, E, LANE_COINS = 64, 7, 3
#: data lanes at step 0 and at a later step; formed lanes a step
LANES0, LANES, K32 = 21, 28, 32
#: the calls the verifier admits (`Statement.admits_of_caps`): K from 1 to 2^16, at most 2^15 rows and 2^20 columns
MAX_K, MAX_ROWS, MAX_COLS = 1 << 16, 1 << 15, 1 << 20
#: the fold's exact range (`serving.LEAD_FOLD_BOUND`): the epilogue reads p = A_i · B_j back exactly while |p| <= 2^22
LEAD_FOLD_BOUND = 1 << 22
#: 1.5 · 2^23 as an fp32's bits: the fold word is FOLD_MAGIC + p, and the epilogue's FADD takes 1.5 · 2^23 off
FOLD_MAGIC, FOLD_OFFSET = 0x4B400000, 12582912.0
#: a step's coin term's bound (`Proofs.Pouw.Lead.Tags.coinTerm_abs`) and a step's data bound with every int8, -128 included
COIN_ABS, DATA_ABS = 16479, LANES * 128 * 128


def steps_for(K: int) -> int:
    """Lean's `stepsFor`: 1 + ⌈(K - 21) / 28⌉, K - 21 truncated at 0."""
    return 1 + (max(K - LANES0, 0) + LANES - 1) // LANES


def pad_rows(m: int) -> int:
    return TM * -(-m // TM)


def pad_cols(n: int) -> int:
    return TN * -(-n // TN)


def k_range(t: int, K: int) -> tuple[int, int]:
    """Step t's data entries [lo, hi) of K: [0, 21) at step 0, then 28 a step."""
    lo = 0 if t == 0 else LANES0 + LANES * (t - 1)
    return min(lo, K), min(lo + (LANES0 if t == 0 else LANES), K)


def bal13(z: int, q: int) -> int:
    """Lean's `bal13`: digit q of z in balanced base 13, (z + 14,280) / 13^q mod 13 - 6, floor division."""
    return (z + 14280) // 13 ** q % 13 - 6


def row_tag(i: int) -> list[int]:
    """T(i), Lean's `rowTag`: the balanced base-13 digits of (⌊i/2⌋ + 1)·(-1)^i."""
    z = i // 2 + 1
    return [bal13(z if i % 2 == 0 else -z, q) for q in range(4)]


def col_code(j: int) -> list[int]:
    """C[:, j], Lean's `colCode`: the balanced base-13 digits of j + 1."""
    return [bal13(j + 1, q) for q in range(4)]


def rot_weight(i: int, l: int) -> int:
    """A′[i, ℓ], Lean's `rotWeight`: 127 on lane i mod 2, else 1."""
    return 127 if l == i % 2 else 1


def bias(S: int, j: int) -> int:
    """β_j, Lean's `bias`: ⌊(Σ_q C[q, j] + 3 (S - 1) + 1) / 2⌋."""
    return (sum(col_code(j)) + 3 * (S - 1) + 1) // 2


def word_bound(S: int) -> int:
    """The largest |word| at S steps for any int8 data: |β| plus S steps of data and coin term."""
    return S * (DATA_ABS + COIN_ABS + 2) + 14


def _sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


# -- the pieces on words ------------------------------------------------------------------------------------------------------

#: step 0's coin product lanes (4 a₁ against C, 4 T against G); the product lanes of step 0 (21 data and those) and of a
#: later step (28 data)
COINS_0 = 8
LANES_0, LANES_T = LANES0 + COINS_0, LANES


def _dot(ab: tuple, n: int) -> int:
    return sum(_sx8(x) * _sx8(y) for x, y in zip(ab[:n], ab[n:2 * n]))


@primitive("PouwLeadStep0", 1, [("prev", Value(32)), ("w", Value(32)), ("a", Array(LANES_0, Value(8))),
                                ("b", Array(LANES_0, Value(8))), ("lane", Value(32))], Value(1),
           conformance="protocol-defined")
def PouwLeadStep0Word(prev: int, w: int, *rest: int) -> int:
    """1 iff w ≡ prev + Σ_l a_l b_l + lane (mod 2^32) over step 0's 29 int8 product lanes, lane its coin lanes' term."""
    return int((prev + _dot(rest, LANES_0) + rest[2 * LANES_0] - w) & M32 == 0)


@primitive("PouwLeadStep", 1, [("prev", Value(32)), ("w", Value(32)), ("a", Array(LANES_T, Value(8))),
                               ("b", Array(LANES_T, Value(8))), ("lane", Value(32)), ("tag", Value(1))], Value(1),
           conformance="protocol-defined")
def PouwLeadStepWord(prev: int, w: int, *rest: int) -> int:
    """1 iff w ≡ prev + Σ_l a_l b_l + lane + 1 + tag (mod 2^32) over a later step's 28 int8 data lanes, lane its coin lanes'
    term and 1 + tag its tag lane."""
    lane, tag = rest[2 * LANES_T], rest[2 * LANES_T + 1]
    return int((prev + _dot(rest, LANES_T) + lane + 1 + (tag & 1) - w) & M32 == 0)


@primitive("PouwLeadPad0", 1, [("prev", Value(32)), ("w", Value(32)), ("a", Array(COINS_0, Value(8))),
                               ("b", Array(COINS_0, Value(8))), ("lane", Value(32))], Value(1),
           conformance="protocol-defined")
def PouwLeadPad0Word(prev: int, w: int, *rest: int) -> int:
    """1 iff w ≡ prev + Σ_l a_l b_l + lane (mod 2^32) over step 0's 8 coin product lanes: a padding cell's step 0, its data
    lanes zero."""
    return int((prev + _dot(rest, COINS_0) + rest[2 * COINS_0] - w) & M32 == 0)


@primitive("PouwLeadPad", 1, [("prev", Value(32)), ("w", Value(32)), ("lane", Value(32)), ("tag", Value(1))], Value(1),
           conformance="protocol-defined")
def PouwLeadPadWord(prev: int, w: int, lane: int, tag: int) -> int:
    """1 iff w ≡ prev + lane + 1 + tag (mod 2^32): a padding cell's later step, its data lanes zero."""
    return int((prev + lane + 1 + (tag & 1) - w) & M32 == 0)


@primitive("PouwLeadWeigh", 1, [("x", Array(3, Value(32)))], Value(32), conformance="protocol-defined")
def PouwLeadWeighWord(x0: int, x1: int, x2: int) -> int:
    """127 x0 + x1 + x2 mod 2^32: three coin lanes, or their sums, under a row's weights A′, its own class's lane first."""
    return (127 * x0 + x1 + x2) & M32


@primitive("PouwLeadTagWeight", 1, [("G", Value(8))], Value(1), conformance="protocol-defined")
def PouwLeadTagWeightWord(G: int) -> int:
    """1 iff the int8 G is in [-3, 3]: a tag weight is `Coins`' Fin 7 coin less 3."""
    return int(-3 <= _sx8(G) <= 3)


@primitive("PouwLeadPairS8", 1, [("x", Array(2, Value(8)))], Value(32), conformance="protocol-defined")
def PouwLeadPairS8Word(x0: int, x1: int) -> int:
    """x0 + x1 of two int8 lane coins, as an s32."""
    return (_sx8(x0) + _sx8(x1)) & M32


@primitive("PouwLeadPair32", 1, [("x", Array(2, Value(32)))], Value(32), conformance="protocol-defined")
def PouwLeadPair32Word(x0: int, x1: int) -> int:
    """x0 + x1 mod 2^32: a lane sum's tree step."""
    return (x0 + x1) & M32


@primitive("PouwLeadPop", 1, [("x", Value(32))], Value(32), conformance="protocol-defined")
def PouwLeadPopWord(x: int) -> int:
    """The number of x's 32 bits that are set: a row's tag bits, 32 at a time."""
    return bin(x & M32).count("1")


@primitive("PouwLeadSub32", 1, [("x", Array(2, Value(32)))], Value(32), conformance="protocol-defined")
def PouwLeadSub32Word(x0: int, x1: int) -> int:
    """x0 - x1 mod 2^32: a fold source, a public constant less a sum of coins."""
    return (x0 - x1) & M32


def _dp4a(a: Any, b: Any, c: int) -> int:
    """IDP4A: c plus the dot product of a's and b's four int8s, mod 2^32."""
    return (c + sum(_sx8(x) * _sx8(y) for x, y in zip(a, b))) & M32


def fold_word(d: int, r: int, c: int, nt: Any, g: Any, na: Any, cj: Any) -> int:
    """`layout.h`'s `fold_word` on a final word d: q = d + r + c (one IADD3 of the row and column sources), then
    q = IDP4A(na, cj, IDP4A(nt, g, q)), which takes a₁ · C and T · G out, all mod 2^32."""
    return _dp4a(na, cj, _dp4a(nt, g, d + r + c))


def fold_admits(p: int) -> bool:
    """Whether the epilogue reads p back exactly: |p| <= `LEAD_FOLD_BOUND`."""
    return -LEAD_FOLD_BOUND <= p <= LEAD_FOLD_BOUND


@primitive("PouwLeadFold", 1, [("w", Value(32)), ("r", Value(32)), ("c", Value(32)), ("nt", Array(4, Value(8))),
                               ("g", Array(4, Value(8))), ("na", Array(4, Value(8))), ("cj", Array(4, Value(8)))],
           Value(1), conformance="protocol-defined")
def PouwLeadFoldWord(w: int, r: int, c: int, *rest: int) -> int:
    """1 iff `fold_word`(w, r, c, nt, g, na, cj) is FOLD_MAGIC + p, mod 2^32, with p `fold_admits`: the served word's fold,
    read as an fp32, is exactly 1.5 · 2^23 + p."""
    q = fold_word(w, r, c, rest[0:4], rest[4:8], rest[8:12], rest[12:16])
    return int((q - FOLD_MAGIC + LEAD_FOLD_BOUND) & M32 <= 2 * LEAD_FOLD_BOUND)


# -- the pieces on bits -------------------------------------------------------------------------------------------------------

#: Baugh-Wooley's constant for one s8 × s8 product: the two negated rows' -(2^14 - 2^7) each
_BW = -(1 << 15) + (1 << 8)


def _products(C, a, b) -> list[list]:
    """32 columns of the Baugh-Wooley partial products of a and b, int8 lane by lane, less their constant len(a) · BW."""
    cols: list[list] = [[] for _ in range(32)]
    for x, y in zip(a, b):
        for i in range(8):
            for j in range(8):
                p = C.AND(x[i], y[j])
                cols[i + j].append(NOT(p) if (i == 7) != (j == 7) else p)
    return cols


def _step_check(C, prev, w, a, b, addends, one):
    """One carry-save reduction mod 2^32 of the Baugh-Wooley products of a and b, prev, the addends, the complement of w
    and the constant n · BW + 1 + one, which is 0 iff the congruence holds; then a NOR of its 32 bits."""
    cols = _products(C, a, b)
    for i in range(32):
        cols[i] += [prev[i], NOT(w[i])]
    for e in addends:
        for i, v in enumerate(e):
            cols[i].append(v)
    for i, v in enumerate(const_bits((len(a) * _BW + 1 + one) & M32, 32)):
        if v.c:
            cols[i].append(v)
    return [NOT(or_reduce(C, compress_columns(C, cols)[:32]))]


def _step0_bits(C, prev, w, *rest):
    n = LANES_0
    return _step_check(C, prev, w, rest[:n], rest[n:2 * n], [rest[2 * n]], 0)


def _step_bits(C, prev, w, *rest):
    n = LANES_T
    return _step_check(C, prev, w, rest[:n], rest[n:2 * n], [rest[2 * n], rest[2 * n + 1]], 1)


def _pad0_bits(C, prev, w, *rest):
    n = COINS_0
    return _step_check(C, prev, w, rest[:n], rest[n:2 * n], [rest[2 * n]], 0)


def _pad_bits(C, prev, w, lane, tag):
    return _step_check(C, prev, w, [], [], [lane, tag], 1)


def _weigh_bits(C, x0, x1, x2):
    """128 x0 - x0 + x1 + x2 by a subtract and two adds mod 2^32."""
    t = sub(C, [ZERO] * 7 + list(x0[:25]), list(x0), 32)[0]
    return add(C, add(C, t, list(x1), width=32), list(x2), width=32)


def _tag_weight_bits(C, G):
    """Bits 2 .. 7 equal (G in [-4, 3]) and not -4 (1111 1100)."""
    equal = NOT(or_reduce(C, [G[k] ^ G[7] for k in range(2, 7)]))
    return [C.AND(equal, NOT(C.AND(G[7], C.AND(NOT(G[1]), NOT(G[0])))))]


def _fold_bits(C, w, r, c, *rest):
    """x = q - FOLD_MAGIC + 2^22 = p + 2^22 in one carry-save reduction mod 2^32 of w, r, c, the Baugh-Wooley products of
    (nt, g) and (na, cj) and the constant 8 · BW - FOLD_MAGIC + 2^22; then x <= 2^23: bits 24 .. 31 clear, and bit 23
    clear or bits 0 .. 22 clear."""
    cols = _products(C, rest[0:4] + rest[8:12], rest[4:8] + rest[12:16])
    for i in range(32):
        cols[i] += [w[i], r[i], c[i]]
    for i, v in enumerate(const_bits((8 * _BW - FOLD_MAGIC + LEAD_FOLD_BOUND) & M32, 32)):
        if v.c:
            cols[i].append(v)
    x = compress_columns(C, cols)[:32]
    high, low = NOT(or_reduce(C, x[24:])), NOT(or_reduce(C, x[:23]))
    return [C.AND(high, NOT(C.AND(x[23], NOT(low))))]


def _pop_bits(C, x):
    """x's set bits counted in one carry-save reduction of its one column."""
    return (compress_columns(C, [list(x)]) + [ZERO] * 32)[:32]


def _pair_s8_bits(C, x0, x1):
    total = add(C, list(x0) + [x0[7]], list(x1) + [x1[7]], width=9)
    return total + [total[8]] * 23


PouwLeadStep0 = definition(
    "PouwLeadStep0", 2, _step0_bits,
    (("prev", U32), ("w", U32), ("a", Array(LANES_0, I8)), ("b", Array(LANES_0, I8)), ("lane", U32)), BIT,
    word=PouwLeadStep0Word, conformance="protocol-defined",
    doc="w ≡ prev + Σ_l a_l b_l + lane (mod 2^32) on bits, step 0's 29 int8 product lanes in one carry-save reduction: "
        "PouwLeadStep0_v1. 3,839 ANDs: 1,856 partial products, 1,952 in the tree and its final ripple, and the NOR's 31")
PouwLeadStep = definition(
    "PouwLeadStep", 2, _step_bits,
    (("prev", U32), ("w", U32), ("a", Array(LANES_T, I8)), ("b", Array(LANES_T, I8)), ("lane", U32), ("tag", BIT)), BIT,
    word=PouwLeadStepWord, conformance="protocol-defined",
    doc="w ≡ prev + Σ_l a_l b_l + lane + 1 + tag (mod 2^32) on bits, a later step's 28 int8 data lanes in one carry-save "
        "reduction: PouwLeadStep_v1. 3,711 ANDs: 1,792 partial products, 1,888 in the tree and its final ripple, and the "
        "NOR's 31")
PouwLeadPad0 = definition(
    "PouwLeadPad0", 2, _pad0_bits,
    (("prev", U32), ("w", U32), ("a", Array(COINS_0, I8)), ("b", Array(COINS_0, I8)), ("lane", U32)), BIT,
    word=PouwLeadPad0Word, conformance="protocol-defined",
    doc="w ≡ prev + Σ_l a_l b_l + lane (mod 2^32) on bits, a padding cell's step 0 over its 8 coin product lanes in one "
        "carry-save reduction: PouwLeadPad0_v1. 1,150 ANDs: 512 partial products, 607 in the tree and its final ripple, and "
        "the NOR's 31")
PouwLeadPad = definition(
    "PouwLeadPad", 2, _pad_bits, (("prev", U32), ("w", U32), ("lane", U32), ("tag", BIT)), BIT, word=PouwLeadPadWord,
    conformance="protocol-defined",
    doc="w ≡ prev + lane + 1 + tag (mod 2^32) on bits, a padding cell's later step in one carry-save reduction: "
        "PouwLeadPad_v1. 123 ANDs: 92 in the tree and its final ripple, and the NOR's 31")
PouwLeadWeigh = definition(
    "PouwLeadWeigh", 2, _weigh_bits, (("x", Array(3, U32)),), U32, word=PouwLeadWeighWord, conformance="protocol-defined",
    doc="127 x0 + x1 + x2 mod 2^32 on bits: PouwLeadWeigh_v1. 92 ANDs over its subtract and two adds")
PouwLeadTagWeight = definition(
    "PouwLeadTagWeight", 2, _tag_weight_bits, (("G", I8),), BIT, word=PouwLeadTagWeightWord, conformance="protocol-defined",
    doc="G in [-3, 3] on bits, bits 2 .. 7 equal and G not -4: PouwLeadTagWeight_v1. 7 ANDs")
PouwLeadPairS8 = definition(
    "PouwLeadPairS8", 2, _pair_s8_bits, (("x", Array(2, I8)),), U32, word=PouwLeadPairS8Word, conformance="protocol-defined",
    doc="x0 + x1 of two int8 lane coins as an s32 on bits, a 9-bit ripple add of their sign extensions, sign-extended: "
        "PouwLeadPairS8_v1. 8 ANDs")
PouwLeadPair32 = definition(
    "PouwLeadPair32", 2, lambda C, x0, x1: add(C, list(x0), list(x1), width=32), (("x", Array(2, U32)),), U32,
    word=PouwLeadPair32Word, conformance="protocol-defined",
    doc="x0 + x1 mod 2^32 on bits, a ripple add: PouwLeadPair32_v1. 31 ANDs")
PouwLeadPop = definition(
    "PouwLeadPop", 2, _pop_bits, (("x", U32),), U32, word=PouwLeadPopWord, conformance="protocol-defined",
    doc="x's set bits counted on bits, one carry-save reduction of its one column: PouwLeadPop_v1. 31 ANDs")
PouwLeadSub32 = definition(
    "PouwLeadSub32", 2, lambda C, x0, x1: sub(C, list(x0), list(x1), 32)[0], (("x", Array(2, U32)),), U32,
    word=PouwLeadSub32Word, conformance="protocol-defined",
    doc="x0 - x1 mod 2^32 on bits, a ripple subtract: PouwLeadSub32_v1. 31 ANDs")
PouwLeadFold = definition(
    "PouwLeadFold", 2, _fold_bits,
    (("w", U32), ("r", U32), ("c", U32), ("nt", Array(4, I8)), ("g", Array(4, I8)), ("na", Array(4, I8)),
     ("cj", Array(4, I8))), BIT, word=PouwLeadFoldWord, conformance="protocol-defined",
    doc="fold_word(w, r, c, nt, g, na, cj) - 0x4B400000 within [-2^22, 2^22] mod 2^32 on bits, one carry-save reduction of "
        "the IADD3 and the two IDP4A's 8 int8 products, then its range: PouwLeadFold_v1. 1,142 ANDs: 512 partial "
        "products, 599 in the tree and its final ripple, and the range's 31")

#: each piece's ANDs, which its pin holds
PIECE_ANDS = {"PouwLeadStep0_v2": 3839, "PouwLeadStep_v2": 3711, "PouwLeadPad0_v2": 1150, "PouwLeadPad_v2": 123,
              "PouwLeadWeigh_v2": 92, "PouwLeadTagWeight_v2": 7, "PouwLeadPairS8_v2": 8, "PouwLeadPair32_v2": 31,
              "PouwLeadPop_v2": 31, "PouwLeadSub32_v2": 31, "PouwLeadFold_v2": 1142}
PIECES = (PouwLeadStep0, PouwLeadStep, PouwLeadPad0, PouwLeadPad, PouwLeadWeigh, PouwLeadTagWeight, PouwLeadPairS8,
          PouwLeadPair32, PouwLeadPop, PouwLeadSub32, PouwLeadFold)

WORDS_ROW, WORDS_TILE = Array(TN, U32), Array(TM, Array(TN, U32))


def _row_mma(name: str, piece: Any, n: int, tag: bool) -> tuple[CompositeDefinition, CompositeDefinition]:
    """A step's row (8 ``piece``, the row's lanes a and tag shared by its 8 columns') and MMA tile (16 rows, the columns'
    lanes b shared by its 16 rows'), each cell's lane term its own."""
    a_row, b_tile = Array(n, I8), Array(TN, Array(n, I8))
    row_params = [("prev", WORDS_ROW), ("w", WORDS_ROW), ("a", a_row), ("b", b_tile), ("lane", WORDS_ROW)]
    tile_params = [("prev", WORDS_TILE), ("w", WORDS_TILE), ("a", Array(TM, a_row)), ("b", b_tile), ("lane", WORDS_TILE)]
    row_axes, tile_axes = [0, 0, None, 0, 0], [0, 0, 0, None, 0]
    if tag:
        row_params.append(("tag", BIT))
        tile_params.append(("tag", Array(TM, BIT)))
        row_axes.append(None)
        tile_axes.append(0)
    ands = f"{PIECE_ANDS[piece.id]:,}"
    row = CompositeDefinition(
        f"PouwLead{name}Row", 1, (), lambda S_: (tuple(row_params), Array(TN, BIT)),
        lambda B, S_, *args: B.batch(piece, *args, axes=row_axes), conformance="protocol-defined",
        doc=f"One row of an MMA tile at {'a later step' if tag else 'step 0'}: 8 `{piece.name}`, the row's lanes shared by "
            f"its 8 columns'. 8 · {ands} ANDs")
    tile = CompositeDefinition(
        f"PouwLead{name}Mma", 1, (), lambda S_: (tuple(tile_params), Array(TM, Array(TN, BIT))),
        lambda B, S_, *args: B.batch(row, *args, axes=tile_axes), conformance="protocol-defined",
        doc=f"One 16 × 8 MMA tile at {'a later step' if tag else 'step 0'}: 16 `PouwLead{name}Row`, the columns' lanes "
            f"shared by its 16 rows'. 128 · {ands} ANDs")
    return row, tile


PouwLeadStep0Row, PouwLeadStep0Mma = _row_mma("Step0", PouwLeadStep0, LANES_0, False)
PouwLeadStepRow, PouwLeadStepMma = _row_mma("Step", PouwLeadStep, LANES_T, True)


# -- the unit -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """One unit's shape: K, the tile's first row and column, its rows and columns of the call, and its block of steps."""

    K: int
    R0: int
    C0: int
    ROWS: int
    COLS: int
    BLK: int

    @property
    def S(self) -> int:
        return steps_for(self.K)

    @property
    def t0(self) -> int:
        return BLOCK * self.BLK

    @property
    def t1(self) -> int:
        return min(self.t0 + BLOCK, self.S)

    @property
    def steps(self) -> int:
        return self.t1 - self.t0

    @property
    def last(self) -> bool:
        return self.t1 == self.S

    @property
    def k0(self) -> int:
        return k_range(self.t0, self.K)[0]

    @property
    def kb(self) -> int:
        """The block's data entries of K."""
        return k_range(self.t1 - 1, self.K)[1] - self.k0

    @property
    def row_bits(self) -> tuple[int, int]:
        """The bits [lo, hi) of each row's row words the unit reads: a₁ at bits 0 .. 3 on block 0 and on the call's last
        block, and b_t at bit t + 3 for the block's steps past 0, on the last block every step's."""
        return (0 if self.BLK == 0 or self.last else self.t0 + 3), self.t1 + 3

    @property
    def reads_g(self) -> bool:
        """Whether the unit reads its columns' G words: on block 0 (step 0's lanes) and on the call's last block (the
        fold)."""
        return self.BLK == 0 or self.last

    @property
    def u0(self) -> int:
        """The first step of the lane coins the unit reads: every step on the call's last block, else its own."""
        return 0 if self.last else self.t0

    def refuse(self) -> str | None:
        if not 1 <= self.K <= MAX_K:
            return f"K = {self.K} is outside [1, {MAX_K}], the inner dimensions the verifier admits"
        if self.R0 < 0 or self.R0 % TM or self.C0 < 0 or self.C0 % TN:
            return f"the tile at row {self.R0}, column {self.C0} is not on the {TM} × {TN} grid"
        if self.R0 + TM > MAX_ROWS or self.C0 + TN > MAX_COLS:
            return f"the tile at row {self.R0}, column {self.C0} passes the {MAX_ROWS} rows and {MAX_COLS} columns the " \
                   "verifier admits"
        if not (1 <= self.ROWS <= TM and 1 <= self.COLS <= TN):
            return f"a tile holds 1 .. {TM} of the call's rows and 1 .. {TN} of its columns, not {self.ROWS} and {self.COLS}"
        if not 0 <= self.t0 < self.S:
            return f"block {self.BLK} starts past the call's {self.S} steps"
        return None

    def params(self) -> tuple:
        """The unit's parameters, in order."""
        lo, hi = self.row_bits
        out = [("a", Array(self.ROWS, Array(self.kb, I8))), ("b", Array(self.COLS, Array(self.kb, I8))),
               ("row", Array(TM, Array(hi - lo, BIT)))]
        if self.reads_g:
            out.append(("g", Array(TN, Array(4, I8))))
        out.append(("s", Array(self.t1 - self.u0, Array(TN, Array(LANE_COINS, I8)))))
        if self.BLK > 0:
            out.append(("prev", Array(TILE_WORDS, U32)))
        out.append(("w", Array(self.steps, Array(TILE_WORDS, U32))))
        return tuple(out)


def _unit(S: Any) -> Unit:
    return Unit(S.K, S.R0, S.C0, S.ROWS, S.COLS, S.BLK)


def _refuse(bindings: dict) -> str | None:
    if set(bindings) != {"K", "R0", "C0", "ROWS", "COLS", "BLK"}:
        return None
    return Unit(**bindings).refuse()


def _signature(S: Any) -> tuple:
    u = _unit(S)
    if (why := u.refuse()) is not None:
        raise ValueError(why)
    return u.params(), BIT


def _lanes(s: Coll, n: int, cols: int) -> Coll:
    """The lane coins s (n steps × 8 columns × 3 lanes, the feed's order) of the first ``cols`` columns lane by lane, lane
    ℓ · cols + c its n steps."""
    refs = s.refs
    if not isinstance(refs, Affine):
        raise TypeError(f"the lane coins are one run of a parameter's leaves, not {refs!r}")
    step = TN * LANE_COINS * 8
    return Coll(Array(LANE_COINS * cols * n, I8), concat([
        Strided(refs.space, refs.idx, refs.base + (c * LANE_COINS + ell) * 8, 8, step, 1, 8 * n)
        for ell in range(LANE_COINS) for c in range(cols)]))


def _wide(refs: Any, width: int) -> Any:
    """An s32 by wiring: an int8's sign bit repeated, a 32-bit word as it is."""
    return refs if width == 32 else concat([refs] + [refs.slice(7, 8)] * 24)


def _sums(B: Any, x: Coll, lanes: int, n: int, width: int) -> Coll:
    """Each of ``lanes`` lanes' sum of its n ``width``-bit terms, a lane's run after the one before, as an s32: a tree of
    `PouwLeadPairS8` on int8 terms, then `PouwLeadPair32`, a level's odd one out carried up."""
    piece = PouwLeadPairS8 if width == 8 else PouwLeadPair32
    while n > 1:
        half, odd = n // 2, n % 2
        pairs = concat([x[k * n:k * n + 2 * half].refs for k in range(lanes)])
        y = B.batch(piece, Coll(Array(lanes * half, Array(2, Array(width, BIT))), pairs), axes=(0,))
        x = Coll(Array(lanes * (half + odd), U32), concat([part for k in range(lanes) for part in (
            [y[k * half:(k + 1) * half].refs] + ([_wide(x[k * n + n - 1].refs, width)] if odd else []))]))
        n, piece, width = half + odd, PouwLeadPair32, 32
    return Coll(Array(lanes, U32), concat([_wide(x[k].refs, width) for k in range(lanes)]))


def _body(B: Any, S: Any, *args: Coll) -> Coll:
    u = _unit(S)
    p = dict(zip((name for name, _t in u.params()), args))
    bits: dict = {}

    def const(v: int, width: int):
        for b in {v >> k & 1 for k in range(width)} - set(bits):
            bits[b] = B.call(BL.constant(b)).refs
        return concat([bits[v >> k & 1] for k in range(width)])

    def each(piece: Any, members: list, *fields: Any) -> Any:
        """``piece`` once a member, its k-th parameter ``fields[k]`` of the member."""
        params, _ret = bind(piece).signature()
        return B.batch(piece, *[Coll(Array(len(members), t), concat([f(*x) for x in members]))
                                for (_n, t), f in zip(params, fields)], axes=(0,) * len(fields)).refs

    su, lo = u.t1 - u.u0, u.row_bits[0]
    srefs = p["s"].refs
    oks = [Strided(srefs.space, srefs.idx, srefs.base, 1, 8, 1, LANE_COINS * TN * su)]
    if u.reads_g:
        oks.append(B.batch(PouwLeadTagWeight, p["g"].reshape(Array(4 * TN, I8)), axes=(0,)).refs)
    weighed = B.batch(PouwLeadWeigh, Coll(Array(u.steps * 2 * TN, Array(LANE_COINS, U32)), concat([
        _wide(p["s"][t - u.u0][c][ell].refs, 8)
        for t in range(u.t0, u.t1) for cls in range(2) for c in range(TN) for ell in (cls, 1 - cls, 2)])), axes=(0,))

    def lane(t: int, cell: int):
        """Step t's lane term at a cell, its row's class's (`PouwLeadWeigh`)."""
        return weighed[((t - u.t0) * 2 + cell // TN % 2) * TN + cell % TN].refs

    def before(t: int, cell: int):
        """The cell's word before step t: -β_j before step 0, else the committed word."""
        if t == 0:
            return const(-bias(u.S, u.C0 + cell % TN) & M32, 32)
        return p["w"][t - 1 - u.t0][cell].refs if t > u.t0 else p["prev"][cell].refs

    def data(side: str, x: int, t: int) -> list:
        """Row or column x's data lanes at step t, past K zero."""
        k_lo, k_hi = k_range(t, u.K)
        pad = (LANES0 if t == 0 else LANES) - (k_hi - k_lo)
        return [p[side][x][k_lo - u.k0:k_hi - u.k0].refs] + ([const(0, 8 * pad)] if pad else [])

    def coins_a(r: int) -> list:
        """Row r's step-0 coin product lanes: a₁[i, q] as an int8, then T(i)."""
        return [concat([p["row"][r][q].refs, const(0, 7)]) for q in range(4)] + \
            [const(v & 0xFF, 8) for v in row_tag(u.R0 + r)]

    def coins_b(c: int) -> list:
        """Column c's step-0 coin product lanes: C[:, j], then G[:, j]."""
        return [const(v & 0xFF, 8) for v in col_code(u.C0 + c)] + [p["g"][c].refs]

    later = range(max(u.t0, 1), u.t1)
    real = [cell for cell in range(TILE_WORDS) if cell // TN < u.ROWS and cell % TN < u.COLS]
    if u.ROWS == TM and u.COLS == TN:
        if u.t0 == 0:
            oks.append(B.call(
                PouwLeadStep0Mma, Coll(WORDS_TILE, concat([before(0, cell) for cell in range(TILE_WORDS)])),
                p["w"][0].reshape(WORDS_TILE),
                Coll(Array(TM, Array(LANES_0, I8)), concat([x for r in range(TM) for x in data("a", r, 0) + coins_a(r)])),
                Coll(Array(TN, Array(LANES_0, I8)), concat([x for c in range(TN) for x in data("b", c, 0) + coins_b(c)])),
                Coll(WORDS_TILE, concat([lane(0, cell) for cell in range(TILE_WORDS)]))).refs)
        if later:
            n, tiles = len(later), Array(len(later), WORDS_TILE)
            oks.append(B.batch(
                PouwLeadStepMma, Coll(tiles, concat([before(t, cell) for t in later for cell in range(TILE_WORDS)])),
                Coll(tiles, p["w"][u.steps - n:].refs),
                Coll(Array(n, Array(TM, Array(LANES_T, I8))),
                     concat([x for t in later for r in range(TM) for x in data("a", r, t)])),
                Coll(Array(n, Array(TN, Array(LANES_T, I8))),
                     concat([x for t in later for c in range(TN) for x in data("b", c, t)])),
                Coll(tiles, concat([lane(t, cell) for t in later for cell in range(TILE_WORDS)])),
                Coll(Array(n, Array(TM, BIT)), concat([p["row"].take(1, t + 3 - lo).refs for t in later])),
                axes=(0,) * 6).refs)
    else:
        pad = [cell for cell in range(TILE_WORDS) if cell not in real]
        if u.t0 == 0:
            prev, w, step0 = (lambda cell: before(0, cell)), (lambda cell: p["w"][0][cell].refs), (lambda cell: lane(0, cell))
            oks.append(each(PouwLeadStep0, [(cell,) for cell in real], prev, w,
                            lambda cell: concat(data("a", cell // TN, 0) + coins_a(cell // TN)),
                            lambda cell: concat(data("b", cell % TN, 0) + coins_b(cell % TN)), step0))
            oks.append(each(PouwLeadPad0, [(cell,) for cell in pad], prev, w, lambda cell: concat(coins_a(cell // TN)),
                            lambda cell: concat(coins_b(cell % TN)), step0))
        if later:
            prev, w, tag = before, (lambda t, cell: p["w"][t - u.t0][cell].refs), \
                (lambda t, cell: p["row"][cell // TN][t + 3 - lo].refs)
            oks.append(each(PouwLeadStep, [(t, cell) for t in later for cell in real], prev, w,
                            lambda t, cell: concat(data("a", cell // TN, t)), lambda t, cell: concat(data("b", cell % TN, t)),
                            lane, tag))
            oks.append(each(PouwLeadPad, [(t, cell) for t in later for cell in pad], prev, w, lane, tag))
    if u.last:
        cols, classes = u.COLS, min(u.ROWS, 2)
        sums = _sums(B, _lanes(p["s"], su, cols), LANE_COINS * cols, su, 8)
        sigma = B.batch(PouwLeadWeigh, Coll(Array(classes * cols, Array(LANE_COINS, U32)), concat([
            sums[ell * cols + c].refs for cls in range(classes) for c in range(cols) for ell in (cls, 1 - cls, 2)])),
            axes=(0,))
        col_src = B.batch(PouwLeadSub32, Coll(Array(classes * cols, Array(2, U32)), concat([
            x for cls in range(classes) for c in range(cols)
            for x in (const(bias(u.S, u.C0 + c) & M32, 32), sigma[cls * cols + c].refs)])), axes=(0,))
        tags, chunks = u.S - 1, -(-(u.S - 1) // 32)
        if tags:
            pops = B.batch(PouwLeadPop, Coll(Array(u.ROWS * chunks, U32), concat([
                x for r in range(u.ROWS) for k in range(chunks)
                for x in [p["row"][r][4 + 32 * k:4 + min(32 * k + 32, tags)].refs]
                + ([const(0, 32 * k + 32 - tags)] if 32 * k + 32 > tags else [])])), axes=(0,))
            walks = _sums(B, Coll(Array(u.ROWS * chunks, U32), pops.refs), u.ROWS, chunks, 32)
            row_src = B.batch(PouwLeadSub32, Coll(Array(u.ROWS, Array(2, U32)), concat([
                x for r in range(u.ROWS) for x in (const((FOLD_MAGIC - tags) & M32, 32), walks[r].refs)])), axes=(0,))
        oks.append(each(
            PouwLeadFold, [(cell,) for cell in real], lambda cell: p["w"][u.steps - 1][cell].refs,
            lambda cell: row_src[cell // TN].refs if tags else const(FOLD_MAGIC, 32),
            lambda cell: col_src[cell // TN % 2 * cols + cell % TN].refs,
            lambda cell: concat([const(-v & 0xFF, 8) for v in row_tag(u.R0 + cell // TN)]),
            lambda cell: p["g"][cell % TN].refs,
            lambda cell: concat([p["row"][cell // TN][q].refs for q in range(4) for _ in range(8)]),
            lambda cell: concat([const(v & 0xFF, 8) for v in col_code(u.C0 + cell % TN)])))
    every = concat(oks)
    return B.call(BL.and_gate(every.count), Coll(Array(every.count, BIT), every))


PouwLeadTile = CompositeDefinition(
    "PouwLeadTile", 1, ("K", "R0", "C0", "ROWS", "COLS", "BLK"), _signature, _body, conformance="protocol-defined",
    refuse=_refuse,
    doc="The lead's sampled unit (`circuit.lead`): one 16 × 8 MMA tile at row R0 and column C0 of a call at contraction K, "
        "ROWS and COLS of its rows and columns the call's, over steps [32 BLK, 32 BLK + 32), to ok: every committed word is "
        "the step's formed k32 lanes added to the word before, every tag weight is in [-3, 3] and every lane coin odd, and "
        "on the call's last block each served word's fold reads back A_i · B_j exactly, |A_i · B_j| <= 2^22. Its ANDs are "
        "`unit_ands`.")


def _tree_ands(n: int, lanes: int, piece: str) -> int:
    ands = 0
    while n > 1:
        ands += lanes * (n // 2) * PIECE_ANDS[piece]
        n, piece = n // 2 + n % 2, "PouwLeadPair32_v2"
    return ands


def unit_ands(K: int, BLK: int, ROWS: int = TM, COLS: int = TN) -> int:
    """`PouwLeadTile_v1`'s ANDs at K and block BLK with ROWS × COLS of its 128 cells the call's: on block 0 step 0's
    `PouwLeadStep0` a cell of the call's and `PouwLeadPad0` a padding cell; its later steps' `PouwLeadStep` and
    `PouwLeadPad` likewise; each step's 16 lane terms' `PouwLeadWeigh`; 32 tag weights' `PouwLeadTagWeight` on a block that
    reads G; on the call's last block, over the call's columns and rows only, its columns' lane sums, a `PouwLeadWeigh`
    and a column source's `PouwLeadSub32` for each column and row class, each row's walk (`PouwLeadPop` 32 tag bits at a
    time, summed) and row source's `PouwLeadSub32`, and a `PouwLeadFold` a cell of the call's; and the AND of its oks, its
    lane coins' low bits among them."""
    u, real = Unit(K, 0, 0, ROWS, COLS, BLK), ROWS * COLS
    su, pad = u.t1 - u.u0, TILE_WORDS - real
    ands = (u.t1 - max(u.t0, 1)) * (real * PIECE_ANDS["PouwLeadStep_v2"] + pad * PIECE_ANDS["PouwLeadPad_v2"]) \
        + u.steps * 2 * TN * PIECE_ANDS["PouwLeadWeigh_v2"]
    oks = u.steps * TILE_WORDS + LANE_COINS * TN * su
    if BLK == 0:
        ands += real * PIECE_ANDS["PouwLeadStep0_v2"] + pad * PIECE_ANDS["PouwLeadPad0_v2"]
    if u.reads_g:
        ands += 4 * TN * PIECE_ANDS["PouwLeadTagWeight_v2"]
        oks += 4 * TN
    if u.last:
        ands += _tree_ands(su, LANE_COINS * COLS, "PouwLeadPairS8_v2") + real * PIECE_ANDS["PouwLeadFold_v2"] \
            + min(ROWS, 2) * COLS * (PIECE_ANDS["PouwLeadWeigh_v2"] + PIECE_ANDS["PouwLeadSub32_v2"])
        if (chunks := -(-(u.S - 1) // 32)):
            ands += ROWS * (chunks * PIECE_ANDS["PouwLeadPop_v2"] + PIECE_ANDS["PouwLeadSub32_v2"]) \
                + _tree_ands(chunks, ROWS, "PouwLeadPair32_v2")
        oks += real
    return ands + oks - 1


# -- the reference ------------------------------------------------------------------------------------------------------------

def f32(x: float) -> float:
    """x rounded to the nearest fp32, ties to even (finite x)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x: float) -> int:
    """The bf16 pack of the fp32 x, rounded to nearest even (finite x): its bits."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u + 0x7FFF + (u >> 16 & 1)) >> 16 & 0xFFFF


def int8_epilogue(y: int, si: float, sj: float) -> int:
    """verity_int8's epilogue on its int32 product y (`int8_exact.dequantize`): RN_bf16(RN32(RN32(RN32(y) s_i) s_j)), the
    activation's scale first."""
    return bf16(f32(f32(f32(y) * si) * sj))


def fold_epilogue(q: int, si: float, sj: float) -> int:
    """The lead's epilogue on its fold word q (`layout.h`'s `xu_fold`, then the bf16 pack): v the fp32 whose bits are q,
    FADD(v, -1.5 · 2^23), FMUL s_i, FMUL s_j. It is `int8_epilogue` on p bit for bit while q = FOLD_MAGIC + p with
    `fold_admits`(p): v is then 1.5 · 2^23 + p exactly. Past it v is another float (2^22 + 1 reads back as 2^22 + 2)."""
    v = struct.unpack("<f", struct.pack("<I", q & M32))[0]
    return bf16(f32(f32(f32(v - FOLD_OFFSET) * si) * sj))


@dataclass
class Call:
    """A call's data and the certifier's coins, as Layout.lean's `Coins` orders them: A (rows × K) and B (cols × K) int8,
    tag bits b[i][t] and step-1 bits a1[i][q] for each of the m padded rows, tag weights' coins g[q][j] in [0, 7) for each
    of the n padded columns, and lane coins u[t][k][ℓ][j] in [0, 128) a step, coin tile, lane and column."""

    A: list[list[int]]
    B: list[list[int]]
    b: list[list[int]]
    a1: list[list[int]]
    g: list[list[int]]
    u: list[list[list[list[int]]]]

    @property
    def rows(self) -> int:
        return len(self.A)

    @property
    def cols(self) -> int:
        return len(self.B)

    @property
    def K(self) -> int:
        return len(self.A[0])

    @property
    def m(self) -> int:
        return pad_rows(self.rows)

    @property
    def n(self) -> int:
        return pad_cols(self.cols)

    @property
    def S(self) -> int:
        return steps_for(self.K)

    @classmethod
    def draw(cls, rng: random.Random, rows: int, cols: int, K: int) -> "Call":
        """A call of random int8 data, -128 included, and its coins drawn as the certifier draws them."""
        m, n, S = pad_rows(rows), pad_cols(cols), steps_for(K)
        tiles = -(-m // COIN_TILE)
        return cls([[rng.randrange(-128, 128) for _ in range(K)] for _ in range(rows)],
                   [[rng.randrange(-128, 128) for _ in range(K)] for _ in range(cols)],
                   [[rng.randrange(2) for _ in range(S)] for _ in range(m)],
                   [[rng.randrange(2) for _ in range(4)] for _ in range(m)],
                   [[rng.randrange(7) for _ in range(n)] for _ in range(4)],
                   [[[[rng.randrange(128) for _ in range(n)] for _ in range(LANE_COINS)] for _ in range(tiles)]
                    for _ in range(S)])

    def s(self, t: int, i: int, ell: int, j: int) -> int:
        """Row i's lane coin s_t[ℓ, ⌊i/64⌋, j] = 2u - 127."""
        return 2 * self.u[t][i // COIN_TILE][ell][j] - 127

    def row_word(self, i: int) -> int:
        """Row i's bits in the feed's row words: a₁[i, q] at bit q, b_t[i] at bit t + 3 for t >= 1."""
        return sum(v << q for q, v in enumerate(self.a1[i])) | sum(self.b[i][t] << (t + 3) for t in range(1, self.S))

    def lanes(self, i: int, t: int, j: int) -> tuple[list[int], list[int]]:
        """Row i's and column j's formed lanes A′ and B′ at step t, a padding row's or column's data zero."""
        lo, hi = k_range(t, self.K)
        width = LANES0 if t == 0 else LANES
        pad = [0] * (width - (hi - lo))
        a = (self.A[i][lo:hi] if i < self.rows else [0] * (hi - lo)) + pad
        b = (self.B[j][lo:hi] if j < self.cols else [0] * (hi - lo)) + pad
        w = [rot_weight(i, ell) for ell in range(LANE_COINS)]
        s = [self.s(t, i, ell, j) for ell in range(LANE_COINS)]
        if t == 0:
            G, C = [self.g[q][j] - 3 for q in range(4)], col_code(j)
            return a + self.a1[i][:3] + row_tag(i) + w + [self.a1[i][3]], b + C[:3] + G + s + [C[3]]
        return a + w + [1 + self.b[i][t]], b + s + [1]

    def words(self) -> list[list[list[int]]]:
        """The committed words W[t][i][j], each step's formed lanes' dot product added to the word before, from -β_j."""
        out, acc = [], [[-bias(self.S, j) for j in range(self.n)] for _ in range(self.m)]
        for t in range(self.S):
            for i in range(self.m):
                for j in range(self.n):
                    a, b = self.lanes(i, t, j)
                    acc[i][j] += sum(x * y for x, y in zip(a, b))
            out.append([row[:] for row in acc])
        return out

    def product(self, i: int, j: int) -> int:
        """A_i · B_j, the int32 product verity_int8 serves."""
        return sum(x * y for x, y in zip(self.A[i], self.B[j]))

    def refuse(self) -> str | None:
        """Why the verifier refuses the call, else None: a served word whose fold the epilogue can't read back exactly
        (`fold_admits`), which the unit of its tile's last block outputs as a visible failure."""
        for i in range(self.rows):
            for j in range(self.cols):
                if not fold_admits(p := self.product(i, j)):
                    return f"the served word at row {i}, column {j} is A_i · B_j = {p}, past the fold's exact range " \
                           f"|p| <= {LEAD_FOLD_BOUND}"
        return None

    def sources(self, i: int, j: int) -> tuple[int, int]:
        """Row i's and column j's fold sources: the row's FOLD_MAGIC less its walk Σ_{t >= 1} (1 + b_t[i]) and the column's
        β_j less σ, the row class's coin lanes Σ_t Σ_ℓ A′[i, ℓ] s_t[ℓ, j]."""
        walk = sum(1 + self.b[i][t] for t in range(1, self.S))
        sigma = sum(rot_weight(i, ell) * self.s(t, i, ell, j) for t in range(self.S) for ell in range(LANE_COINS))
        return (FOLD_MAGIC - walk) & M32, (bias(self.S, j) - sigma) & M32

    def fold(self, i: int, j: int, words: list | None = None) -> int:
        """The kernel's fold word q of the final word (i, j) (`fold_word`): the IADD3 of its sources, then -a₁[i] against
        C[:, j] and -T(i) against G[:, j]."""
        W = self.words() if words is None else words
        r, c = self.sources(i, j)
        return fold_word(W[-1][i][j], r, c, [-v & 0xFF for v in row_tag(i)], [self.g[q][j] - 3 & 0xFF for q in range(4)],
                         [-v & 0xFF for v in self.a1[i]], [v & 0xFF for v in col_code(j)])

    def served(self, i: int, j: int, si: float, sj: float, words: list | None = None) -> int:
        """The bf16 the lead serves at row i and column j under scales s_i and s_j (`fold_epilogue` on its fold)."""
        return fold_epilogue(self.fold(i, j, words), si, sj)

    def unit(self, R0: int, C0: int, BLK: int) -> Unit:
        return Unit(self.K, R0, C0, min(self.rows - R0, TM), min(self.cols - C0, TN), BLK)

    def unit_args(self, R0: int, C0: int, BLK: int, words: list | None = None) -> list[list[list[int]]]:
        """The unit's inputs at the tile at row R0 and column C0 and block BLK, `evaluate_words`' arrays of one instance:
        the call's data, its coins in the feed's encodings (row bits, G and s as int8s) and the committed words ``words``
        (`words` by default)."""
        u, W = self.unit(R0, C0, BLK), self.words() if words is None else words
        lo, hi = u.k0, u.k0 + u.kb
        cells = [(R0 + cell // TN, C0 + cell % TN) for cell in range(TILE_WORDS)]
        rlo, rhi = u.row_bits
        args = {"a": [x & 0xFF for i in range(R0, R0 + u.ROWS) for x in self.A[i][lo:hi]],
                "b": [x & 0xFF for j in range(C0, C0 + u.COLS) for x in self.B[j][lo:hi]],
                "row": [self.row_word(R0 + r) >> rlo & (1 << (rhi - rlo)) - 1 for r in range(TM)],
                "g": [self.g[q][C0 + c] - 3 & 0xFF for c in range(TN) for q in range(4)] if u.reads_g else [],
                "s": [self.s(t, R0, ell, C0 + c) & 0xFF for t in range(u.u0, u.t1) for c in range(TN)
                      for ell in range(LANE_COINS)],
                "prev": [W[u.t0 - 1][i][j] & M32 for i, j in cells] if u.BLK else [],
                "w": [W[t][i][j] & M32 for t in range(u.t0, u.t1) for i, j in cells]}
        return [[args[name]] for name, _t in u.params()]

    def unit_ok(self, R0: int, C0: int, BLK: int, words: list | None = None) -> int:
        """The unit's verdict on the call's committed words ``words`` (`words` by default)."""
        from verity.ml.circuits.evaluation.bits import evaluate_words

        u = self.unit(R0, C0, BLK)
        fn = bind(PouwLeadTile, K=u.K, R0=R0, C0=C0, ROWS=u.ROWS, COLS=u.COLS, BLK=BLK)
        return int(evaluate_words(fn, *self.unit_args(R0, C0, BLK, words)).reshape(-1)[0])
