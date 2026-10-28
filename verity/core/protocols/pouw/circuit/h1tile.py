"""`eps1-sm120-h1fx1` as a compile-pass scheme (`compile.SCHEMES`): sm_120's H1 + fx1 (decisions 82 and 83, E2E-120's
`segment.PouwCallCheck_v1{FOLD "fx1"}`) as one self-contained r × c tile unit (decisions 105 and 106) with the warp
checkpoint receipt CP-W (decision 104, #2438's `PouwWarpCheck_v2` body).

The unit is ``(a: r rows of X, b: c rows of W, salt, position, roots) -> {z, receipt}`` (`compile.Scheme`), on bits:

* **Seed and noise.** seed_j and σ_R are one call of `seeds.PouwUnitSeeds` on the unit's salt, position and roots. The
  noise is seed_j's stream in counter mode, each row's and column's from blocks of its own (`Layout`): row i of X takes
  blocks ``[i RB, (i + 1) RB)``, its dither bits b (K) then its digit coins z (32 (m - 1)); column j of W blocks
  ``r RB + [j CB, (j + 1) CB)``, its coin bits c (K) then U's 32 bytes (`PouwU129`: -64 + byte mod 129, H1's keyed word in
  ±64). Lane l's fold keys (duo's y_0..7, z_0..7), its chain's start, are block l of σ_R's stream (decision 121): no key
  is a function of seed_j. A kernel derives any row's or column's noise from seed_j alone, and any lane's keys from σ_R.
  In a live window the salt is round r's, `live_salt` (`window.round_key` of round r's coin and root, its stream's first
  64 bytes), and every unit computes its own noise from it in its own proof: no noise is a step that several units read.
* **Lanes.** The tile is G × Q lanes, G = r / 4 and Q = c / 8 (at 32 × 32 a warp of cgemm.cu, lane 4 g + q): lane
  (g, q) holds rows ``g + G s`` (row slot s < 4) and columns ``2 q + 2 Q (s >> 1) + (s & 1)`` (column slot s < 8), and its
  entry e at row slot `segment.row_slot`(e) and column slot `segment.col_slot`(e), so the lanes' entries are the tile's
  cells once each.
* **Sets.** Each lane's H1 sets (`segment.call_words`) over X = A + b and W' = B + c + U: T_0 .. T_(KS-1), O' and
  D_1 .. D_(m-1), m = `segment.n_digits`(KS). X, W', U, the residue sums and the digits are computed once a row or column.
* **Receipt** (CP-W): each lane folds its sets from its keys with fx1 (`PouwFx1Absorb_v2`); after set s, receipt word 2 s is
  the XOR over lanes and chains of y, and word 2 s + 1 of z (`PouwXinFinal_v2` on each half of the state, `PouwXor32_v2`
  across lanes): 2 (KS + m) words. A lane's own words before that XOR are its partial (`fold`, `lane_partials`), which a
  lane sub-statement returns (`h1zy.PouwH1LaneZy`).
* **z** is each cell's exact product, the served Call's own Definition on the cell's rows of A and B, which no noise reaches.

**The one-stream unit**, `PouwH1Tile_v3` (`SCHEME_V3`, `eps1-sm120-h1fx1-os`), is v1 with z taken from the words the
receipt folds instead of a product of its own: each cell's z is cgemm.cu's served output O = O' - sum_d 128^d D_d
(`PouwOneStream2` at m = 2, `PouwOneStream3` at m = 3) from the cell's lane's O' and D_1 .. D_(m-1), plus the exact wrap
correction 256 X (wb⁺ - wb⁻)ᵀ (`PouwWrapSeg` over each segment of K), minus the dither peel X cᵀ + (b - 256 xa) Bᵀ
(`PouwPeelSeg` over each segment), mod 2^32. Its noise, lanes, sets and receipt are v1's bit for bit.

* **Wrap.** W' = B + c + U is int8, so it wraps where B + c + U leaves [-128, 127]: wb⁺ = [B + c + U > 127] and
  wb⁻ = [B + c + U < -128], each column's from its B (registered) and its noise, computed in the tile (`PouwWrapBits`),
  never supplied. Since T_last = X W'ᵀ with W' = Ŵ + U - 256 wb⁺ + 256 wb⁻, O' = T_last - A_0 Uᵀ and D_d = A_d Uᵀ, with
  sum_d 128^d A_d the residue sums of X, the recombination plus the correction is X Ŵᵀ, the product of the noised operands
  X = A + b (the int8 the MMA reads) and Ŵ = B + c, for every int8 B.
* **Peel.** X = A + b wraps too, at A = 127 with b = 1 (X = -128): xa = [A = 127] AND b, each row's from its A and its
  dither bits, computed in the tile (`PouwXWrap`). With X = A + b - 256 xa, X Ŵᵀ - X cᵀ - (b - 256 xa) Bᵀ = A Bᵀ (the
  same as - b Bᵀ - A cᵀ - b cᵀ + 256 xa Ŵᵀ), so z is A Bᵀ, the served product, exact in s32 for every int8 A and B, and
  equal to v1's z.

**The gated unit**, `PouwH1Tile_v4`, is v3 with one more input, ``admit``: its r rows' admit bits, the queue's admission
(`PouwQueueAdmit`'s words, the bits the credit tally counts), a committed input like A and never a function of the rows. A
dead tile, every admit bit 0, has the canonical empty receipt, zero words: each receipt word is masked by the tile's live
bit, the OR of the admit bits (`PouwOrBit`, `PouwMask32`). Its z, and a live tile's receipt, are v3's. So a dead tile's
served values need no noise: its receipt is the constant 0 and its rows, which admission holds to the canonical empty row,
give z = A Bᵀ = 0; its credit, the tally of its admit bits, is 0.

**The J10-R units**, `PouwH1Tile_v5` (v1's z) and `PouwH1Tile_v6` (v3's one stream), are v1 and v3 under J10-R's noise
(`seeds`' docstring), with two more public
inputs: ``call``, the position of its call's first unit, and ``blocks``, the first block of its call row R I and of its call
column C J in the call-scale `Layout(M, N, K)` (`j10r_blocks`). Its noise is seed_c's (`seeds.call_seed` of the salt,
``call`` and the roots), row i from block ``blocks[0] + i RB`` and column j from ``blocks[1] + j CB``: the call-scale
``noise(seed_c, Layout(M, N, K))``'s rows and columns of the unit (`j10r_noise`), each block derived once a call by whoever
serves it (`j10r_compressions`). Its fold keys are σ_R's at its own position, v1's. No scheme takes v5 or v6.

**J10-R v2**, `PouwH1Tile_v7` (v5's z) and `PouwH1Tile_v8` (v6's one stream), is v5 and v6 under noise version 2
(`Layout.v`): U is fresh at every entry of K, column j's K bytes after its coin bits (``col_bits`` = 9 K, read U[t]),
where v1 draws 32 and reads U[t % 32]. Under v1, X Uᵀ = Y Uᵀ for Y X's residue sums, so a prover with A = B = 0 gets
b Uᵀ by folding b (about 32 / K of a product) and b cᵀ by a binary product; under v2, b (c + U)ᵀ is a product of fresh
random matrices. X Uᵀ has no fold, so v2's sets drop the digits: T_0 .. T_(KS-1), then O' = T_last - X Uᵀ over every
segment (m = 1, KS + 1 sets, 2 (KS + 1) receipt words; a row's noise its K dither bits alone), and the one stream's z is
O' itself with v6's wrap correction and peel. Serving it costs one more int8 product, X Uᵀ, the size of the served one.
Seeds, block counters and fold keys are v5's. Version 1 stays the recorded runs'; no scheme takes v7 or v8.

So z exists only through the MMAs whose words are credited, and A, B and the noise: a kernel serving y = dequant(z)
serves the receipted words. The compile pass replaces its ``noise-in-z`` rule for a one-stream scheme
(`compile.Scheme.stream`, here `STREAM`) with ``stream``: every z leaf is made by those Definitions from leaves the
receipt's cone reads.

The scheme takes the Boolean s8 kinds (`KINDS`) on whole-lane grid tiles: a full r × c grid with 4 | r and 8 | c, 32 | K
and KS = K / 32 >= 2; any other dot or tile is `CompileError("scheme")`. `scheme(tile, name)` builds it at another tile;
`SCHEME` is registered at a warp, 32 × 32, and `PouwH1Tile` as a Definition, when `segment` (E2E-120) is on the tree;
`SCHEME_V3` (`scheme_v3`) and `PouwH1Tile3` the same. `PouwH1Tile4` is a Definition only: the compile pass has no admit
input to give it. `reference` is the unit on words (``stream=True``: v3's; with ``gated``, the admit bits: v4's).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, add, and_reduce, compress_columns, const_bits, or_reduce
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind, primitive
from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Record, Value

from . import compile as C
from . import seeds as SD
from .boolean import PouwXor32

try:
    from . import segment as SG
except ImportError:  # H1's receipts and their checks (E2E-120, `circuit.segment`) are not on this tree
    SG = None

try:
    from verity.ml.circuits import ports as _ports
except ImportError:  # port types (circuits PROTOCOL.md §12) are not on this tree
    _ports = None

SCHEME = "eps1-sm120-h1fx1"
#: the one-stream unit's scheme (`PouwH1Tile3`)
SCHEME_V3 = "eps1-sm120-h1fx1-os"
FOLD = "fx1"
#: the dot kinds the unit takes: the s8 families on bits
KINDS = ("GemmAccumulatorS8_v2", "S8Dot_v2")
BIT = BL.BIT
I8, U32 = Array(8, BIT), Array(32, BIT)
M32 = (1 << 32) - 1
#: noise bits a column takes past its coin bits under noise version 1: U's 32 bytes
U_BITS = 256
#: the noise versions (`Layout.v`): 1, U's 32 bytes a column read at t % 32; 2, J10-R v2's, U fresh at every entry of K
NOISE_VERSIONS = (1, 2)
#: the products' noise on a row of X (its dither bits b, an entry each; the digit coins are the receipt's) and on a column
#: of W (its coin bits c, an entry each, and U): the s8 kinds' form for the work floor (`compile.Scheme.noise`)
H1_ROWS = C.RowNoise(a=(0, 1), b=(U_BITS, 1))
#: a lane's fold keys: duo's y_0..7, z_0..7, one block
KEY_BITS = 512
#: the salt port's bytes, a SHA-512 digest
SALT_BYTES = 64


@dataclass(frozen=True)
class Layout:
    """One r × c tile at contraction K under noise version v (`NOISE_VERSIONS`): its lanes, seed_j's noise blocks in
    counter mode, and σ_R's key blocks."""

    r: int
    c: int
    k: int
    v: int = 1

    def __post_init__(self) -> None:
        if self.v not in NOISE_VERSIONS:
            raise ValueError(f"noise version {self.v!r} is none of {NOISE_VERSIONS}")

    @property
    def ks(self) -> int:
        return self.k // 32

    @property
    def m(self) -> int:
        """The peel's words a cell: O' and D_1 .. D_(m-1) under v1; O' alone under v2 (1), whose U has no fold."""
        return SG.n_digits(self.ks) if self.v == 1 else 1

    @property
    def sets(self) -> int:
        """The sets a lane absorbs: KS prefix sets, O', D_1 .. D_(m-1)."""
        return self.ks + self.m

    @property
    def u(self) -> int:
        """U's bytes a column: 32 under v1, read at t % 32; K under v2, read at t."""
        return 32 if self.v == 1 else self.k

    @property
    def row_bits(self) -> int:
        return self.k + 32 * (self.m - 1)

    @property
    def col_bits(self) -> int:
        return self.k + 8 * self.u

    @property
    def rb(self) -> int:
        return -(-self.row_bits // SD.BLOCK_BITS)

    @property
    def cb(self) -> int:
        return -(-self.col_bits // SD.BLOCK_BITS)

    @property
    def lanes(self) -> int:
        return (self.r // 4) * (self.c // 8)

    @property
    def blocks(self) -> int:
        """seed_j's noise blocks: the rows' and the columns'."""
        return self.r * self.rb + self.c * self.cb

    def row(self, i: int) -> int:
        """Row i's first block."""
        return i * self.rb

    def col(self, j: int) -> int:
        return self.r * self.rb + j * self.cb

    def lane(self, ln: int) -> int:
        """Lane ln's block of σ_R's stream: its fold keys."""
        return ln

    def rows(self, ln: int) -> tuple[int, ...]:
        """Lane ln's tile rows, by row slot."""
        G = self.r // 4
        g = ln // (self.c // 8)
        return tuple(g + G * s for s in range(4))

    def cols(self, ln: int) -> tuple[int, ...]:
        """Lane ln's tile columns, by column slot."""
        Q = self.c // 8
        q = ln % Q
        return tuple(2 * q + 2 * Q * (s >> 1) + (s & 1) for s in range(8))


# -- U's bytes ---------------------------------------------------------------------------------------------------------------

def u129(x: int) -> int:
    """A noise byte as H1's keyed word: -64 + x mod 129, as its int8 encoding."""
    return (-64 + (x & 0xFF) % 129) & 0xFF


PouwU129Word = PrimitiveDefinition("PouwU129", 1, (("x", Value(8)),), Value(8), u129, conformance="protocol-defined")


def _u129_bits(C_, x):
    ge = C_.AND(x[7], or_reduce(C_, list(x[:7])))
    return add(C_, list(x), [ge] * 6 + [NOT(ge)] * 2, width=8)


PouwU129 = definition("PouwU129", 2, _u129_bits, (("x", I8),), I8, word=PouwU129Word, conformance="protocol-defined",
                      doc="-64 + x mod 129 on bits (x + 192, or x + 63 where x >= 129): PouwU129_v1")


# -- the one stream ----------------------------------------------------------------------------------------------------------

def one_stream(op: int, d: Sequence[int]) -> int:
    """cgemm.cu's served output from a cell's absorbed words: O' - sum_d 128^d D_d mod 2^32 (d = 1 .. m - 1)."""
    return (op - sum(w << 7 * (t + 1) for t, w in enumerate(d))) & M32


@primitive("PouwOneStream2", 1, [("op", Value(32)), ("d1", Value(32))], Value(32), conformance="protocol-defined")
def PouwOneStream2Word(op: int, d1: int) -> int:
    """O' - 128 D_1 mod 2^32."""
    return one_stream(op, [d1])


@primitive("PouwOneStream3", 1, [("op", Value(32)), ("d1", Value(32)), ("d2", Value(32))], Value(32),
           conformance="protocol-defined")
def PouwOneStream3Word(op: int, d1: int, d2: int) -> int:
    """O' - 128 D_1 - 128² D_2 mod 2^32."""
    return one_stream(op, [d1, d2])


def _one_stream_bits(C_, op, *d):
    acc = list(op)
    for t, w in enumerate(d):
        sh = 7 * (t + 1)
        acc = add(C_, acc, [ONE] * sh + [NOT(v) for v in w[:32 - sh]], ONE, width=32)
    return acc


PouwOneStream2 = definition("PouwOneStream2", 2, _one_stream_bits, (("op", U32), ("d1", U32)), U32, word=PouwOneStream2Word,
                            conformance="protocol-defined", doc="O' - 128 D_1 mod 2^32 on bits: PouwOneStream2_v1")
PouwOneStream3 = definition("PouwOneStream3", 2, _one_stream_bits, (("op", U32), ("d1", U32), ("d2", U32)), U32,
                            word=PouwOneStream3Word, conformance="protocol-defined",
                            doc="O' - 128 D_1 - 128² D_2 mod 2^32 on bits: PouwOneStream3_v1")
#: the recombination at m digits
ONE_STREAM = {2: PouwOneStream2, 3: PouwOneStream3}


def _sx(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def wrap_bits(b: int, c: int, u: int) -> int:
    """W' = B + c + U's int8 wrap at one entry: 1 (wb⁺) where B + c + U > 127, 2 (wb⁻) where it is < -128, else 0."""
    s = _sx(b) + (c & 1) + _sx(u)
    return 1 if s > 127 else 2 if s < -128 else 0


def wrap_seg(acc: int, x: Sequence[int], wb: Sequence[int]) -> int:
    """acc + 256 sum_e x_e (wb⁺_e - wb⁻_e) mod 2^32 over one segment: the wrap correction of a cell's one stream, x the int8
    encodings of X and wb `wrap_bits`' (both bits set: 0), so a flipped bit moves the sum by ±256 x_e."""
    return (acc + (sum(_sx(xe) * ((w & 1) - (w >> 1 & 1)) for xe, w in zip(x, wb)) << 8)) & M32


PouwWrapBitsWord = PrimitiveDefinition("PouwWrapBits", 1, (("b", Value(8)), ("c", Value(1)), ("u", Value(8))), Value(2),
                                       wrap_bits, conformance="protocol-defined")


@primitive("PouwWrapSeg", 1, [("acc", Value(32)), ("x", Array(32, Value(8))), ("wb", Array(32, Value(2)))], Value(32),
           conformance="protocol-defined")
def PouwWrapSegWord(acc: int, *xw: int) -> int:
    """acc + 256 sum_e x_e (wb⁺_e - wb⁻_e) mod 2^32 (`wrap_seg`)."""
    return wrap_seg(acc, xw[:32], xw[32:])


def _wrap_bits_bits(C_, b, c, u):
    s = add(C_, list(b) + [b[7]], list(u) + [u[7]], c[0], width=9)
    ov = s[7] ^ s[8]
    p = C_.AND(ov, s[7])
    return [p, ov ^ p]


def _wrap_seg_bits(C_, acc, *xw):
    """Each entry's term x (wb⁺ - wb⁻) as the 9-bit (sel x) XOR m plus m (sel = wb⁺ XOR wb⁻, m = wb⁻ AND NOT wb⁺), its
    sign row inverted against a constant, shifted by 8 into one carry-save tree with acc's bits."""
    x, wb = xw[:32], xw[32:]
    cols: list[list] = [[] for _ in range(32)]
    for i, v in enumerate(acc):
        cols[i].append(v)
    for xe, (p, n) in zip(x, wb):
        sel, m = p ^ n, C_.AND(n, NOT(p))
        xs = [C_.AND(v, sel) for v in xe]
        for i in range(8):
            cols[8 + i].append(xs[i] ^ m)
        cols[16].append(NOT(xs[7] ^ m))
        cols[8].append(m)
    for i, v in enumerate(const_bits((-32 << 16) & M32, 32)):
        if v.c:
            cols[i].append(v)
    return compress_columns(C_, cols)[:32]


PouwWrapBits = definition("PouwWrapBits", 2, _wrap_bits_bits, (("b", I8), ("c", BIT), ("u", I8)), Array(2, BIT),
                          word=PouwWrapBitsWord, conformance="protocol-defined",
                          doc="B + c + U's int8 wrap bits [wb⁺, wb⁻] from its 9-bit sum: PouwWrapBits_v1")
PouwWrapSeg = definition("PouwWrapSeg", 2, _wrap_seg_bits,
                         (("acc", U32), ("x", Array(32, I8)), ("wb", Array(32, Array(2, BIT)))), U32, word=PouwWrapSegWord,
                         conformance="protocol-defined",
                         doc="acc + 256 sum_e x_e (wb⁺_e - wb⁻_e) mod 2^32 on bits: PouwWrapSeg_v1")
def x_wrap(a: int, b: int) -> int:
    """X = A + b's int8 wrap at one entry: 1 where A = 127 and its dither bit b is set (X = -128), else 0."""
    return int((a & 0xFF) == 0x7F and b & 1)


def peel_seg(acc: int, x: Sequence[int], bd: Sequence[int], xa: Sequence[int], w: Sequence[int], c: Sequence[int]) -> int:
    """acc - sum_e (c_e x_e + bd_e w_e - 256 xa_e w_e) mod 2^32 over one segment: the dither peel of a cell's one stream,
    x the int8 encodings of X, w of B, bd and c the row's and the column's dither bits, xa `x_wrap`'s. Since
    X = A + b - 256 xa and Ŵ = B + c, X Ŵᵀ - X cᵀ - (b - 256 xa) Bᵀ = A Bᵀ."""
    return (acc - sum(_sx(xe) * (ce & 1) + _sx(we) * ((be & 1) - 256 * (ae & 1))
                      for xe, be, ae, we, ce in zip(x, bd, xa, w, c))) & M32


PouwXWrapWord = PrimitiveDefinition("PouwXWrap", 1, (("a", Value(8)), ("b", Value(1))), Value(1), x_wrap,
                                    conformance="protocol-defined")


@primitive("PouwPeelSeg", 1, [("acc", Value(32)), ("x", Array(32, Value(8))), ("bd", Value(32)), ("xa", Value(32)),
                              ("w", Array(32, Value(8))), ("c", Value(32))], Value(32), conformance="protocol-defined")
def PouwPeelSegWord(acc: int, *v: int) -> int:
    """acc - sum_e (c_e x_e + bd_e w_e - 256 xa_e w_e) mod 2^32 (`peel_seg`), bd, xa and c a bit an entry."""
    bd, xa, c = ([m >> e & 1 for e in range(32)] for m in (v[32], v[33], v[66]))
    return peel_seg(acc, v[:32], bd, xa, v[34:66], c)


def _x_wrap_bits(C_, a, b):
    return [C_.AND(C_.AND(b[0], NOT(a[7])), and_reduce(C_, list(a[:7])))]


def _signed_row(cols: list[list], g: Sequence, shift: int, neg: bool) -> int:
    """Add ±2^shift times the int8 g into `cols`, each negative-weight bit as its complement; the constant it leaves."""
    k = 0
    for i, v in enumerate(g):
        if (i == 7) != neg:
            cols[shift + i].append(NOT(v))
            k -= 1 << shift + i
        else:
            cols[shift + i].append(v)
    return k


def _peel_seg_bits(C_, acc, *v):
    """Each entry's -c x, -bd w and +256 xa w as gated int8 rows, their negative-weight bits complemented against one
    constant, in one carry-save tree with acc's bits."""
    x, (bd, xa), w, (c,) = v[:32], v[32:34], v[34:66], v[66:]
    cols: list[list] = [[] for _ in range(32)]
    for i, b in enumerate(acc):
        cols[i].append(b)
    k = 0
    for xe, be, ae, we, ce in zip(x, bd, xa, w, c):
        k += _signed_row(cols, [C_.AND(b, ce) for b in xe], 0, True)
        k += _signed_row(cols, [C_.AND(b, be) for b in we], 0, True)
        k += _signed_row(cols, [C_.AND(b, ae) for b in we], 8, False)
    for i, b in enumerate(const_bits(k & M32, 32)):
        if b.c:
            cols[i].append(b)
    return compress_columns(C_, cols)[:32]


PouwXWrap = definition("PouwXWrap", 2, _x_wrap_bits, (("a", I8), ("b", BIT)), BIT, word=PouwXWrapWord,
                       conformance="protocol-defined", doc="[A = 127] AND b on bits: PouwXWrap_v1")
PouwPeelSeg = definition("PouwPeelSeg", 2, _peel_seg_bits,
                         (("acc", U32), ("x", Array(32, I8)), ("bd", Array(32, BIT)), ("xa", Array(32, BIT)),
                          ("w", Array(32, I8)), ("c", Array(32, BIT))), U32, word=PouwPeelSegWord,
                         conformance="protocol-defined",
                         doc="acc - sum_e (c_e x_e + bd_e w_e - 256 xa_e w_e) mod 2^32 on bits: PouwPeelSeg_v1")
#: what a one-stream z is made of: the recombinations, the wrap correction and the dither peel
STREAM = (PouwOneStream2, PouwOneStream3, PouwWrapSeg, PouwWrapBits, PouwPeelSeg, PouwXWrap)


# -- a dead tile -------------------------------------------------------------------------------------------------------------

@primitive("PouwOrBit", 1, [("a", Value(1)), ("b", Value(1))], Value(1), conformance="protocol-defined")
def PouwOrBitWord(a: int, b: int) -> int:
    return (a | b) & 1


@primitive("PouwMask32", 1, [("w", Value(32)), ("live", Value(1))], Value(32), conformance="protocol-defined")
def PouwMask32Word(w: int, live: int) -> int:
    """w where live is 1, else 0."""
    return w & M32 if live & 1 else 0


PouwOrBit = definition("PouwOrBit", 2, lambda C_, a, b: [OR(C_, a[0], b[0])], (("a", BIT), ("b", BIT)), BIT,
                       word=PouwOrBitWord, conformance="protocol-defined", doc="a OR b on bits: PouwOrBit_v1")
PouwMask32 = definition("PouwMask32", 2, lambda C_, w, live: [C_.AND(v, live[0]) for v in w], (("w", U32), ("live", BIT)),
                        U32, word=PouwMask32Word, conformance="protocol-defined", doc="w AND live, on bits: PouwMask32_v1")


# -- the reference -----------------------------------------------------------------------------------------------------------

def _stream(seed_j: bytes, block: int, n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        out += SD.leaves(SD.noise_block(seed_j, block), True)
        block += 1
    return out[:n]


def _value(bits: Sequence[int]) -> int:
    return sum((b & 1) << i for i, b in enumerate(bits))


def _noise_at(seed: bytes, lay: Layout, row0: int, col0: int) -> tuple[list, list]:
    k, m = lay.k, lay.m
    rows = []
    for i in range(lay.r):
        s = _stream(seed, row0 + i * lay.rb, lay.row_bits)
        rows.append((s[:k], [s[k + 32 * d:k + 32 * d + 32] for d in range(m - 1)]))
    cols = []
    for j in range(lay.c):
        s = _stream(seed, col0 + j * lay.cb, lay.col_bits)
        cols.append((s[:k], [u129(_value(s[k + 8 * t:k + 8 * t + 8])) for t in range(lay.u)]))
    return rows, cols


def noise(seed_j: bytes, lay: Layout) -> tuple[list, list]:
    """The unit's noise from seed_j: per row (dither bits, digit coins), per column (coin bits, U's int8 encodings: 32
    under noise version 1, K under v2)."""
    return _noise_at(seed_j, lay, lay.row(0), lay.col(0))


def live_salt(commitment, coin: bytes, r: int, root: bytes) -> bytes:
    """A live window's unit salt in round r (`window`, "Live windows"): the first `SALT_BYTES` of `window.round_key`'s
    stream from round r's opened coin and the round's root.  It is the unit's salt port; each unit derives its own seed_j,
    σ_R and noise from it, its public position and its roots inside the unit, so no unit reads noise another computed."""
    from .. import window as Wd
    return Wd.round_key(commitment, coin, r, root).stream(0, SALT_BYTES)


def fold_keys(sigma_r: bytes, lay: Layout) -> list[list[int]]:
    """Each lane's 16 fold keys, its chain's start, from σ_R (`seeds.receipt_seed`)."""
    keys = []
    for ln in range(lay.lanes):
        s = _stream(sigma_r, lay.lane(ln), KEY_BITS)
        keys.append([_value(s[32 * w:32 * w + 32]) for w in range(16)])
    return keys


def lane_sets(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], rows: list, cols: list, lay: Layout,
              ln: int) -> list[list[int]]:
    """Lane ln's absorbed sets on words (`segment.call_words`, under noise version 2 `call_words_v2`): T_0 .. T_(KS-1),
    O', D_1 .. D_(m-1), 32 entries each."""
    R, Cs = lay.rows(ln), lay.cols(ln)
    seg = [range(32 * s, 32 * s + 32) for s in range(lay.ks)]
    xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
    bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
    wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
    cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
    if lay.v == 2:
        return call_words_v2(xr, bb, wr, cc, [[[cols[j][1][t] for t in ts] for j in Cs] for ts in seg])
    return SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs], [[rows[i][1][d] for i in R] for d in range(lay.m - 1)])


def call_words_v2(xr, b, wr, c, u) -> list[list[int]]:
    """One lane's absorbed sets under noise version 2, by segment as `segment.call_words` takes them, with ``u`` its 8
    columns' U at each segment (fresh at every entry of K): T_0 .. T_(KS-1) over X = A + b and W' = B + c + U, then
    O' = T_last - X Uᵀ entry by entry over every segment. X Uᵀ has no fold, so there are no digits."""
    X = [[[(v + (bb & 1)) & 0xFF for v, bb in zip(xs[r], bs[r])] for r in range(4)] for xs, bs in zip(xr, b, strict=True)]
    Wp = [[[(v + (cc & 1) + uu) & 0xFF for v, cc, uu in zip(ws[q], cs[q], us[q])] for q in range(8)]
          for ws, cs, us in zip(wr, c, u, strict=True)]
    T = SG.prefixes(X, Wp)
    op = list(T[-1])
    for xs, us in zip(X, u, strict=True):
        neg = [[(-v) & 0xFF for v in col] for col in us]
        op = [SG.seg_dot(op[e], neg[SG.col_slot(e)], xs[SG.row_slot(e)]) for e in range(32)]
    return T + [op]


def col_wrap(b_j: Sequence[int], col: tuple) -> list[int]:
    """Column j's wrap bits at every entry of K (`wrap_bits` of B, its coin bit and U, U under noise version 1 repeating
    every 32 entries)."""
    return [wrap_bits(v, col[0][t], col[1][t % len(col[1])]) for t, v in enumerate(b_j)]


def row_wrap(a_i: Sequence[int], row: tuple) -> list[int]:
    """Row i's X wrap bits at every entry of K (`x_wrap` of A and its dither bit)."""
    return [x_wrap(v, row[0][t]) for t, v in enumerate(a_i)]


def stream_cell(op: int, d: Sequence[int], x_i: Sequence[int], wb_j: Sequence[int], peel_i: tuple, peel_j: tuple) -> int:
    """A cell's v3 z: `one_stream` of its O' and D words, then `wrap_seg` over each segment of its row's X and its column's
    wrap bits (X Ŵᵀ), then `peel_seg` over each segment with its row's (dither bits, X wrap bits) `peel_i` and its column's
    (B, coin bits) `peel_j`: A Bᵀ mod 2^32 for every int8 A and B."""
    z = one_stream(op, d)
    for s in range(0, len(x_i), 32):
        z = wrap_seg(z, x_i[s:s + 32], wb_j[s:s + 32])
    for s in range(0, len(x_i), 32):
        z = peel_seg(z, x_i[s:s + 32], *(v[s:s + 32] for v in (*peel_i, *peel_j)))
    return z


def reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes, position: int,
              roots: tuple[bytes, bytes], pairs: Sequence[tuple[int, int]] | None = None,
              stream: bool = False, gated: Sequence[int] | None = None) -> tuple[list[int], list[int]]:
    """The unit on words: (z, receipt) for A's r rows and B's c rows (int8 encodings), its salt, position and roots, z at
    `pairs` (the full grid in row-major order by default). `stream`: v3's z, each cell's `one_stream` of its lane's O' and
    D words with its wrap correction and its dither peel (`stream_cell`); else v1's, the exact A Bᵀ cell. Both are A Bᵀ.
    `gated`: the tile's r admit bits (v4, ``stream=True``): a dead tile, every bit 0, has the zero receipt and its exact
    z, its noise never derived; a live tile's z and receipt are the ungated ones."""
    lay = Layout(len(a), len(b), len(a[0]))
    if pairs is None:
        pairs = [(i, j) for i in range(lay.r) for j in range(lay.c)]
    if gated is not None:
        if isinstance(gated, (bool, int)) or len(gated) != lay.r:
            raise ValueError(f"gated is the tile's {lay.r} admit bits, not {gated!r}")
        if not any(g & 1 for g in gated):
            return [SG.seg_dot(0, a[i], b[j]) for i, j in pairs], [0] * (2 * lay.sets)
    rows, cols = noise(SD.seed(salt, position, *roots), lay)
    return _unit(a, b, rows, cols, fold_keys(SD.receipt_seed(salt, position, *roots), lay), lay, pairs, stream)


def _unit(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], rows: list, cols: list, keys: list[list[int]],
          lay: Layout, pairs: Sequence[tuple[int, int]], stream: bool) -> tuple[list[int], list[int]]:
    xs = [[(a[i][t] + rows[i][0][t]) & 0xFF for t in range(lay.k)] for i in range(lay.r)] if stream else None
    wbs = [col_wrap(b[j], cols[j]) for j in range(lay.c)] if stream else None
    pis = [(rows[i][0][:lay.k], row_wrap(a[i], rows[i])) for i in range(lay.r)] if stream else None
    pjs = [(b[j], cols[j][0][:lay.k]) for j in range(lay.c)] if stream else None
    ks = lay.ks
    cp = [0] * (2 * lay.sets)
    served: dict[tuple[int, int], int] = {}
    for ln in range(lay.lanes):
        sets = lane_sets(a, b, rows, cols, lay, ln)
        for w, v in enumerate(fold(keys[ln], sets)):
            cp[w] ^= v
        for e in range(32 if stream else 0):
            cell = (lay.rows(ln)[SG.row_slot(e)], lay.cols(ln)[SG.col_slot(e)])
            served[cell] = stream_cell(sets[ks][e], [d[e] for d in sets[ks + 1:]], xs[cell[0]], wbs[cell[1]],
                                       pis[cell[0]], pjs[cell[1]])
    z = [served[p] for p in pairs] if stream else [SG.seg_dot(0, a[i], b[j]) for i, j in pairs]
    return z, cp


def fold(key: Sequence[int], sets: Sequence[Sequence[int]]) -> list[int]:
    """One lane's part of the receipt on words, its partial: from its 16 fold keys, after each of its sets (`lane_sets`) the
    XOR of its fx1 state's y half (word 2 s) and of its z half (word 2 s + 1), 2 (KS + m) words. The receipt is the lanes'
    partials XORed word by word."""
    st, out = list(key), []
    for p in sets:
        st = SG.absorb(FOLD, st, p)
        out += [_xor_words(st[0:8]), _xor_words(st[8:16])]
    return out


def _xor_words(ws: Sequence[int]) -> int:
    x = 0
    for w in ws:
        x ^= w
    return x & M32


def lane_partials(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes, position: int,
                  roots: tuple[bytes, bytes], gated: Sequence[int] | None = None) -> list[list[int]]:
    """Each lane's partial (`fold`) of the unit's receipt on words, lane by lane, for A's r rows and B's c rows: the words
    whose XOR is `reference`'s receipt, the rows a lane sub-statement checks (`h1zy.PouwH1LaneZy`). `gated`: the tile's r
    admit bits (v4's): a dead tile's partials, every bit 0, are zero words, whose XOR is its zero receipt."""
    lay = Layout(len(a), len(b), len(a[0]))
    if gated is not None:
        if isinstance(gated, (bool, int)) or len(gated) != lay.r:
            raise ValueError(f"gated is the tile's {lay.r} admit bits, not {gated!r}")
        if not any(g & 1 for g in gated):
            return [[0] * (2 * lay.sets) for _ in range(lay.lanes)]
    rows, cols = noise(SD.seed(salt, position, *roots), lay)
    keys = fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    return [fold(keys[ln], lane_sets(a, b, rows, cols, lay, ln)) for ln in range(lay.lanes)]


# -- J10-R -------------------------------------------------------------------------------------------------------------------

def j10r_blocks(m: int, k: int, tile: tuple[int, int], I: int, J: int, v: int = 1) -> tuple[int, int]:
    """Unit (I, J)'s ``blocks`` under J10-R: the first block of call row R I and of call column C J in its call's one
    stream at call scale, `Layout(m, ·, k, v)` (row i at i RB, column j at m RB + j CB), for a call of ``m`` A rows (its
    slot) and an R × C ``tile``, at noise version ``v``. The verifier derives them from the call's structure. Refused
    (ValueError): a slot of no whole row tiles, a row tile outside it, a negative column tile."""
    r, c = tile
    if m < r or m % r or not 0 <= I < m // r or J < 0:
        raise ValueError(f"unit ({I}, {J}) is no R × C = {r} × {c} tile of a call of {m} rows")
    call = Layout(m, c, k, v)
    return call.row(r * I), call.col(c * J)


def j10r_noise(seed_c: bytes, lay: Layout, blocks: tuple[int, int]) -> tuple[list, list]:
    """A unit's noise under J10-R, `noise`'s format: row i's blocks from ``blocks[0] + i RB`` and column j's from
    ``blocks[1] + j CB`` of the call's seed_c (`seeds.call_seed`), the call-scale `noise(seed_c, Layout(M, N, K))`'s rows
    R I .. R I + R - 1 and columns C J .. C J + C - 1 at `j10r_blocks`' (I, J)."""
    return _noise_at(seed_c, lay, *blocks)


def j10r_unit_blocks(lay: Layout, blocks: tuple[int, int]) -> list[int]:
    """The counters of seed_c's blocks a unit reads under J10-R, rows' then columns', as its gates derive them."""
    return ([blocks[0] + i * lay.rb + t for i in range(lay.r) for t in range(lay.rb)]
            + [blocks[1] + j * lay.cb + t for j in range(lay.c) for t in range(lay.cb)])


def j10r_reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes, position: int,
                   roots: tuple[bytes, bytes], call: int, blocks: tuple[int, int],
                   pairs: Sequence[tuple[int, int]] | None = None, stream: bool = False,
                   v: int = 1) -> tuple[list[int], list[int]]:
    """The J10-R unit on words (`PouwH1Tile5`, with ``stream`` `PouwH1Tile6`; at noise version ``v`` 2 `PouwH1Tile7` and
    `PouwH1Tile8`): `reference`'s (z, receipt) with its noise `j10r_noise` of `seeds.call_seed`(salt, ``call``, roots) at
    ``blocks`` (`j10r_blocks` at ``v``), ``call`` its call's first unit position, and its fold keys today's, σ_R at its own
    ``position``."""
    lay = Layout(len(a), len(b), len(a[0]), v)
    if pairs is None:
        pairs = [(i, j) for i in range(lay.r) for j in range(lay.c)]
    rows, cols = j10r_noise(SD.call_seed(salt, call, *roots), lay, blocks)
    return _unit(a, b, rows, cols, fold_keys(SD.receipt_seed(salt, position, *roots), lay), lay, pairs, stream)


def unit_compressions(k: int, tile: tuple[int, int] = (32, 32), v: int = 1) -> int:
    """SHA-512 compressions a unit's noise, keys and seeds take under today's form (`PouwUnitSeeds`' four, its noise
    blocks, a block a lane's keys), at noise version ``v``: 612 at a warp and K = 4096 under v1."""
    lay = Layout(*tile, k, v)
    return lay.blocks + lay.lanes + 4


def j10r_compressions(m: int, n: int, k: int, tile: tuple[int, int] = (32, 32), v: int = 1) -> Fraction:
    """SHA-512 compressions a unit under J10-R (at noise version ``v``) when a call of ``m`` rows and ``n`` columns
    derives each block once: the distinct counters of seed_c its units read (`j10r_unit_blocks` at `j10r_blocks`,
    counted, not computed), seed_c's two compressions once a call, and each unit's own σ_R (two) and its lanes' key
    blocks, over the call's units."""
    r, c = tile
    if n % c:
        raise ValueError(f"a call of {n} columns is no whole tiles of {c}")
    lay, read = Layout(r, c, k, v), set()
    for I in range(m // r):
        for J in range(n // c):
            read.update(j10r_unit_blocks(lay, j10r_blocks(m, k, tile, I, J, v)))
    units = (m // r) * (n // c)
    return Fraction(len(read) + 2, units) + 2 + lay.lanes


# -- the unit ----------------------------------------------------------------------------------------------------------------

def _layout(S: Any, v: int = 1) -> Layout:
    return Layout(S.R, S.C, S.EXACT.signature()[0][0][1].n, v)


def _signature(S: Any, admit: bool = False, v: int = 1) -> tuple:
    (_, a), (_, b) = S.EXACT.signature()[0][:2]
    D, P = SD.types(True)
    return ((("a", Array(S.R, a)), ("b", Array(S.C, b)), *((("admit", Array(S.R, BIT)),) if admit else ()), ("salt", D),
             ("position", P), ("roots", Array(2, D))),
            Record(("z", Array(len(S.PAIRS), S.EXACT.signature()[1])), ("receipt", Array(2 * _layout(S, v).sets, U32))))


def _signature_v4(S: Any) -> tuple:
    return _signature(S, admit=True)


def _digits(B: Any, x: Coll, z: list[Coll], lay: Layout) -> list[Coll]:
    """A row's digits A_0 .. A_(m-1) of its residue sums (`segment._call_sets`' per row)."""
    Y = B.batch(SG.PouwSx8, x[0:32], axes=(0,))
    for s in range(1, lay.ks):
        Y = B.batch(SG.PouwResAdd, Y, x[32 * s:32 * s + 32], axes=(0, 0))
    return _sum_digits(B, Y, z, lay.m)


def _sum_digits(B: Any, Y: Coll, z: list[Coll], m: int) -> list[Coll]:
    """A row's digits A_0 .. A_(m-1) of its residue sums Y themselves (32 u32 words)."""
    out, R = [], Y
    for d in range(m - 1):
        zd = B.batch(SG.PouwDigitZ, R, z[d], axes=(0, 0)) if d == m - 2 else z[d]
        out.append(B.batch(SG.PouwDigitLo, R, zd, axes=(0, 0)))
        R = B.batch(SG.PouwDigitHi, R, zd, axes=(0, 0))
    out.append(B.batch(SG.PouwLow8, R, axes=(0,)))
    return out


def _xor(B: Any, words: list[Coll]) -> Coll:
    while len(words) > 1:
        words = [B.call(PouwXor32, words[i], words[i + 1]) if i + 1 < len(words) else words[i]
                 for i in range(0, len(words), 2)]
    return words[0]


def _row(B: Any, a_i: Coll, nz: Coll, lay: Layout) -> tuple[Coll, list[Coll], tuple[Coll, Coll]]:
    """A row's X = A + b, its digits A_0 .. A_(m-1) (none under noise version 2) and X's addends (A, dither bits), from
    its codes and its noise bits."""
    k = lay.k
    x = B.batch(SG.PouwI8AddBit, a_i, nz[0:k], axes=(0, 0))
    digits = _digits(B, x, [nz[k + 32 * d:k + 32 * d + 32] for d in range(lay.m - 1)], lay) if lay.v == 1 else []
    return x, digits, (a_i, nz[0:k])


def _col(B: Any, b_j: Coll, nz: Coll, lay: Layout) -> tuple[Coll, Coll, Coll, tuple[Coll, Coll, Coll]]:
    """A column's U, -U, W' = B + c + U and W''s addends (B, coin bits, U over K: v1's 32 repeated, v2's K), from its
    codes and its noise bits."""
    k = lay.k
    u = B.batch(PouwU129, Coll(Array(lay.u, I8), nz[k:k + 8 * lay.u].refs), axes=(0,))
    nu = B.batch(SG.PouwI8Neg, u, axes=(0,))
    wc = B.batch(SG.PouwI8AddBit, b_j, nz[0:k], axes=(0, 0))
    ur = Coll(Array(k, I8), concat([u.refs] * (k // lay.u)))
    return u, nu, B.batch(SG.PouwI8Add, wc, ur, axes=(0, 0)), (b_j, nz[0:k], ur)


def _chain(B: Any, lay: Layout, X: list, W: list, U: list, NU: list, A: list, key: Coll) -> tuple[list, list]:
    """One lane's sets and its fx1 chain: X and A its 4 rows' (by row slot), W, U and NU its 8 columns' (by column slot),
    ``key`` its block of σ_R's stream. Per set, the state's two receipt words (`PouwXinFinal` of its y half, then of its z
    half), the lane's part of the receipt; per entry, its O' and D_1 .. D_(m-1). Under noise version 2, O' is T_last
    less X Uᵀ over every segment, and there are no D."""
    rs, cs = [SG.row_slot(e) for e in range(32)], [SG.col_slot(e) for e in range(32)]
    T, prev = [], None
    for s in range(lay.ks):
        xs = [x[32 * s:32 * s + 32] for x in X]
        ws = [w[32 * s:32 * s + 32] for w in W]
        prev = [B.call(SG.PouwSegDot0, xs[rs[e]], ws[cs[e]]) if prev is None else
                B.call(SG.PouwSegDot, prev[e], xs[rs[e]], ws[cs[e]]) for e in range(32)]
        T.append(prev)
    if lay.v == 1:
        op = [B.call(SG.PouwSegDot, T[-1][e], NU[cs[e]], A[rs[e]][0]) for e in range(32)]
    else:
        op = T[-1]
        for s in range(lay.ks):
            op = [B.call(SG.PouwSegDot, op[e], NU[cs[e]][32 * s:32 * s + 32], X[rs[e]][32 * s:32 * s + 32])
                  for e in range(32)]
    D = [[B.call(SG.PouwSegDot0, U[cs[e]], A[rs[e]][d]) for e in range(32)] for d in range(1, lay.m)]
    state, cp = Coll(Array(KEY_BITS, BIT), key.refs), []
    for p in [*T, op, *D]:
        state = B.call(SG.PouwFx1Absorb, state, array_of(p))
        cp.append((B.call(SG.PouwXinFinal, state[0:256]), B.call(SG.PouwXinFinal, state[256:512])))
    return cp, [[op[e]] + [d[e] for d in D] for e in range(32)]


def _bits(blocks: Sequence[Coll], n: int) -> Coll:
    return Coll(Array(n, BIT), concat([blk.refs for blk in blocks[:-(-n // SD.BLOCK_BITS)]]).slice(0, n))


def _sets(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> tuple[list, dict, list, list, list]:
    """The receipt's refs; per cell (i, j) its lane's O' and D_1 .. D_(m-1) words, as the receipt folds them; each row's X;
    each column's (B, coin bits, U repeated over K), W''s addends; and each row's (A, dither bits), X's."""
    lay = _layout(S)
    seed_j, sigma_r = SD.unit_seeds_call(B, salt, position, roots, True)
    blocks = SD.noise_blocks(B, seed_j, lay.blocks, True)
    keys = SD.noise_blocks(B, sigma_r, lay.lanes, True)
    return _lanes(B, lay, a, b, [_bits(blocks[lay.row(i):], lay.row_bits) for i in range(lay.r)],
                  [_bits(blocks[lay.col(j):], lay.col_bits) for j in range(lay.c)], keys)


def _sets_j10r(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll, call: Coll,
               blocks: Coll, v: int = 1) -> tuple[list, dict, list, list, list]:
    """`_sets` under J10-R at noise version ``v``: seed_c and σ_R from one `seeds.PouwCallSeeds`, row i's blocks of
    seed_c from counter ``blocks[0] + i RB`` and column j's from ``blocks[1] + j CB``, and the lanes' keys σ_R's as
    today."""
    lay = _layout(S, v)
    seed_c, sigma_r = SD.call_seeds_call(B, salt, position, call, roots, True)
    rows, cols = SD.noise_blocks_at(B, seed_c, [(blocks[0], range(lay.r * lay.rb)), (blocks[1], range(lay.c * lay.cb))],
                                    True)
    keys = SD.noise_blocks(B, sigma_r, lay.lanes, True)
    return _lanes(B, lay, a, b, [_bits(rows[i * lay.rb:], lay.row_bits) for i in range(lay.r)],
                  [_bits(cols[j * lay.cb:], lay.col_bits) for j in range(lay.c)], keys)


def _lanes(B: Any, lay: Layout, a: Coll, b: Coll, row_bits: list[Coll], col_bits: list[Coll],
           keys: list[Coll]) -> tuple[list, dict, list, list, list]:
    X, A, XIN = [], [], []
    for i in range(lay.r):
        x, d, xin = _row(B, a[i], row_bits[i], lay)
        X.append(x)
        A.append(d)
        XIN.append(xin)
    W, U, NU, WIN = [], [], [], []
    for j in range(lay.c):
        u, nu, w, win = _col(B, b[j], col_bits[j], lay)
        U.append(u)
        NU.append(nu)
        W.append(w)
        WIN.append(win)
    rs, cs = [SG.row_slot(e) for e in range(32)], [SG.col_slot(e) for e in range(32)]
    cps: list[list[tuple[Coll, Coll]]] = [[] for _ in range(lay.sets)]
    words: dict[tuple[int, int], list[Coll]] = {}
    for ln in range(lay.lanes):
        R, Cs = lay.rows(ln), lay.cols(ln)
        cp, w = _chain(B, lay, [X[i] for i in R], [W[j] for j in Cs], [U[j] for j in Cs], [NU[j] for j in Cs],
                       [A[i] for i in R], keys[lay.lane(ln)])
        for s, pair in enumerate(cp):
            cps[s].append(pair)
        for e in range(32):
            words[(R[rs[e]], Cs[cs[e]])] = w[e]
    return [_xor(B, [w[h] for w in ws]).refs for ws in cps for h in range(2)], words, X, WIN, XIN


def _body(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    receipt = _sets(B, S, a, b, salt, position, roots)[0]
    z = [B.call(S.EXACT, a[i], b[j]).refs for i, j in S.PAIRS]
    return Coll(_signature(S)[1], concat(z + receipt))


def _body_v3(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    z, receipt = _one_stream(B, S, a, b, salt, position, roots)
    return Coll(_signature(S)[1], concat(z + receipt))


def _body_v4(B: Any, S: Any, a: Coll, b: Coll, admit: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    z, receipt = _one_stream(B, S, a, b, salt, position, roots)
    live = [admit[i] for i in range(S.R)]
    while len(live) > 1:
        live = [B.call(PouwOrBit, live[i], live[i + 1]) if i + 1 < len(live) else live[i] for i in range(0, len(live), 2)]
    receipt = [B.call(PouwMask32, Coll(U32, w), live[0]).refs for w in receipt]
    return Coll(_signature_v4(S)[1], concat(z + receipt))


def _one_stream(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> tuple[list, list]:
    """v3's z refs (cell by cell of PAIRS) and receipt refs (word by word)."""
    return _stream_z(B, S, _sets(B, S, a, b, salt, position, roots))


def _signature_j10r(S: Any, v: int = 1) -> tuple:
    params, ret = _signature(S, v=v)
    return (*params, ("call", SD.POS_BITS), ("blocks", Array(2, SD.POS_BITS))), ret


def _signature_j10r_v2(S: Any) -> tuple:
    return _signature_j10r(S, 2)


def _body_exact_j10r(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll, call: Coll,
                     blocks: Coll, v: int) -> Coll:
    receipt = _sets_j10r(B, S, a, b, salt, position, roots, call, blocks, v)[0]
    z = [B.call(S.EXACT, a[i], b[j]).refs for i, j in S.PAIRS]
    return Coll(_signature(S, v=v)[1], concat(z + receipt))


def _body_stream_j10r(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll, call: Coll,
                      blocks: Coll, v: int) -> Coll:
    z, receipt = _stream_z(B, S, _sets_j10r(B, S, a, b, salt, position, roots, call, blocks, v), v)
    return Coll(_signature(S, v=v)[1], concat(z + receipt))


def _body_v5(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll, call: Coll, blocks: Coll) -> Coll:
    return _body_exact_j10r(B, S, a, b, salt, position, roots, call, blocks, 1)


def _body_v6(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll, call: Coll, blocks: Coll) -> Coll:
    return _body_stream_j10r(B, S, a, b, salt, position, roots, call, blocks, 1)


def _body_v7(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll, call: Coll, blocks: Coll) -> Coll:
    return _body_exact_j10r(B, S, a, b, salt, position, roots, call, blocks, 2)


def _body_v8(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll, call: Coll, blocks: Coll) -> Coll:
    return _body_stream_j10r(B, S, a, b, salt, position, roots, call, blocks, 2)


def _stream_z(B: Any, S: Any, sets: tuple, v: int = 1) -> tuple[list, list]:
    receipt, words, X, WIN, XIN = sets
    lay = _layout(S, v)
    wb = {j: B.batch(PouwWrapBits, *WIN[j], axes=(0, 0, 0)) for j in sorted({j for _i, j in S.PAIRS})}
    xa = {i: B.batch(PouwXWrap, *XIN[i], axes=(0, 0)) for i in sorted({i for i, _j in S.PAIRS})}
    z = [_cell_z(B, lay, words[(i, j)], X[i], XIN[i], xa[i], WIN[j], wb[j]).refs for i, j in S.PAIRS]
    return z, receipt


def _cell_z(B: Any, lay: Layout, words: list, x_i: Coll, xin_i: tuple, xa_i: Coll, win_j: tuple, wb_j: Coll) -> Coll:
    """A cell's one-stream z: `ONE_STREAM` of its O' and D words, its wrap correction over each segment of its row's X and
    its column's wrap bits, then its dither peel over each segment (`stream_cell`). At m = 1 (noise version 2) O' is
    the recombination."""
    acc = B.call(ONE_STREAM[lay.m], *words) if lay.m > 1 else words[0]
    for s in range(lay.ks):
        acc = B.call(PouwWrapSeg, acc, x_i[32 * s:32 * s + 32], wb_j[32 * s:32 * s + 32])
    for s in range(lay.ks):
        seg = slice(32 * s, 32 * s + 32)
        acc = B.call(PouwPeelSeg, acc, x_i[seg], xin_i[1][seg], xa_i[seg], win_j[0][seg], win_j[1][seg])
    return acc


def _unit_ports(S: Any) -> Any:
    position = {"position": _ports.In(kind="position")} if "position" in _ports.KINDS else {}
    return _ports.Ports(params={"salt": _ports.In(kind="coin"), **position}, ret={"receipt": _ports.Out(sink=True)})


PouwH1Tile = CompositeDefinition(
    "PouwH1Tile", 1, ("EXACT", "R", "C", "PAIRS"), _signature, _body, conformance="protocol-defined",
    register=SG is not None,
    doc="One r × c tile of H1 + fx1 under CP-W (`circuit.h1tile`): A's R rows and B's C rows, the salt, the unit's position "
        "and its seed group's roots, to {z: EXACT on each cell of PAIRS, receipt: per set the lanes' XOR of y, then of z}.",
    **({"ports": _unit_ports} if _ports is not None else {}))

PouwH1Tile3 = CompositeDefinition(
    "PouwH1Tile", 3, ("EXACT", "R", "C", "PAIRS"), _signature, _body_v3, conformance="protocol-defined",
    register=SG is not None,
    doc="The one-stream H1 + fx1 tile (`circuit.h1tile`): v1's noise, sets and receipt, and z on each cell of PAIRS its "
        "lane's O' - sum_d 128^d D_d plus its wrap correction 256 X (wb⁺ - wb⁻)ᵀ, less its dither peel "
        "X cᵀ + (b - 256 xa) Bᵀ, mod 2^32: the served product A Bᵀ for every int8 A and B (EXACT's type, never called).",
    **({"ports": _unit_ports} if _ports is not None else {}))

PouwH1Tile4 = CompositeDefinition(
    "PouwH1Tile", 4, ("EXACT", "R", "C", "PAIRS"), _signature_v4, _body_v4, conformance="protocol-defined",
    register=SG is not None,
    doc="v3's one-stream H1 + fx1 tile (`circuit.h1tile`) with its R rows' admit bits: a dead tile, every admit bit 0, has "
        "the zero receipt (each word AND the OR of the bits); z and a live tile's receipt are v3's.",
    **({"ports": _unit_ports} if _ports is not None else {}))

PouwH1Tile5 = CompositeDefinition(
    "PouwH1Tile", 5, ("EXACT", "R", "C", "PAIRS"), _signature_j10r, _body_v5, conformance="protocol-defined",
    register=SG is not None,
    doc="v1's H1 + fx1 tile (`circuit.h1tile`) under J10-R's noise: seed_c = `seeds.call_seed` at its call's first "
        "position `call`, row i's noise from block blocks[0] + i RB of seed_c's stream and column j's from blocks[1] + j CB "
        "(`j10r_blocks`, the call-scale stream's), the fold keys σ_R's at the unit's own position; z EXACT on each cell.",
    **({"ports": _unit_ports} if _ports is not None else {}))

PouwH1Tile6 = CompositeDefinition(
    "PouwH1Tile", 6, ("EXACT", "R", "C", "PAIRS"), _signature_j10r, _body_v6, conformance="protocol-defined",
    register=SG is not None,
    doc="v3's one-stream H1 + fx1 tile (`circuit.h1tile`) under v5's J10-R noise: z each cell's O' - sum_d 128^d D_d with "
        "its wrap correction and dither peel, A Bᵀ for every int8 A and B.",
    **({"ports": _unit_ports} if _ports is not None else {}))

PouwH1Tile7 = CompositeDefinition(
    "PouwH1Tile", 7, ("EXACT", "R", "C", "PAIRS"), _signature_j10r_v2, _body_v7, conformance="protocol-defined",
    register=SG is not None,
    doc="v5's J10-R H1 + fx1 tile (`circuit.h1tile`) under noise version 2: column j's U fresh at every entry of K (its K "
        "bytes after its coin bits), so the lane's sets are T_0 .. T_(KS-1) of X (B + c + U)ᵀ and O' = T_last - X Uᵀ, with "
        "no digits; seeds, blocks and fold keys v5's; z EXACT on each cell.",
    **({"ports": _unit_ports} if _ports is not None else {}))

PouwH1Tile8 = CompositeDefinition(
    "PouwH1Tile", 8, ("EXACT", "R", "C", "PAIRS"), _signature_j10r_v2, _body_v8, conformance="protocol-defined",
    register=SG is not None,
    doc="v6's one-stream J10-R H1 + fx1 tile (`circuit.h1tile`) under v7's noise version 2: z each cell's O' with its wrap "
        "correction (U read at every entry of K) and dither peel, A Bᵀ for every int8 A and B.",
    **({"ports": _unit_ports} if _ports is not None else {}))


@dataclass(frozen=True)
class _Unit:
    """The scheme's factory: `compile.Scheme.units`' per kind."""

    name: str
    defn: Any = PouwH1Tile

    def __call__(self, dot: C.Dot, tile: C.Tile) -> Any:
        if dot.family not in KINDS or dot.a != Array(dot.k, I8) or dot.b != Array(dot.k, I8):
            raise C.CompileError("scheme", f"{self.name}'s H1 tile takes the s8 kinds on bits ({', '.join(KINDS)}), not "
                                           f"{dot.family} on {dot.a!r}")
        if not tile.grid or tile.r % 4 or tile.c % 8:
            raise C.CompileError("scheme", f"{self.name}'s H1 tile is whole lanes, a full r × c grid with 4 | r and 8 | c, not "
                                           f"{len(tile.pairs)} cells of {tile.r} × {tile.c}")
        if dot.k % 32 or dot.k < 64:
            raise C.CompileError("scheme", f"{self.name}'s H1 tile needs 32 | K and KS = K / 32 >= 2 (a digit word), "
                                           f"not K = {dot.k}")
        if self.defn is PouwH1Tile3 and SG.n_digits(dot.k // 32) not in ONE_STREAM:
            raise C.CompileError("scheme", f"{self.name}'s one stream recombines m = 2 or 3 digits, not "
                                           f"m = {SG.n_digits(dot.k // 32)} at K = {dot.k}")
        return bind(self.defn, EXACT=dot.exact, R=tile.r, C=tile.c, PAIRS=tile.pairs)


def scheme(tile: tuple[int, int] = (32, 32), name: str = SCHEME) -> C.Scheme:
    """H1 + fx1 as a compile-pass scheme at `tile` (register it with `compile.register`)."""
    f = _Unit(name)
    return C.Scheme(name, tile, {kind: f for kind in KINDS}, noise={kind: H1_ROWS for kind in KINDS})


def scheme_v3(tile: tuple[int, int] = (32, 32), name: str = SCHEME_V3) -> C.Scheme:
    """The one stream (`PouwH1Tile3`) as a compile-pass scheme at `tile`, whose z the pass checks is made by `STREAM` from
    folded words."""
    f = _Unit(name, PouwH1Tile3)
    return C.Scheme(name, tile, {kind: f for kind in KINDS}, stream=STREAM)


if SG is not None:
    C.register(scheme())
    C.register(scheme_v3())

__all__ = ["FOLD", "H1_ROWS", "KINDS", "Layout", "NOISE_VERSIONS", "PouwH1Tile", "PouwH1Tile5", "PouwH1Tile6", "PouwH1Tile7",
           "PouwH1Tile8", "PouwU129", "PouwU129Word", "SALT_BYTES", "SCHEME", "call_words_v2", "fold", "fold_keys",
           "j10r_blocks", "j10r_compressions", "j10r_noise", "j10r_reference",
           "j10r_unit_blocks", "lane_partials", "live_salt", "noise", "reference", "scheme", "u129", "unit_compressions"]
