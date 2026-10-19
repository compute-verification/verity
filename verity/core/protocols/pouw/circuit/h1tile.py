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
* **Lanes.** The tile is G × Q lanes, G = r / 4 and Q = c / 8 (at 32 × 32 a warp of cgemm.cu, lane 4 g + q): lane
  (g, q) holds rows ``g + G s`` (row slot s < 4) and columns ``2 q + 2 Q (s >> 1) + (s & 1)`` (column slot s < 8), and its
  entry e at row slot `segment.row_slot`(e) and column slot `segment.col_slot`(e), so the lanes' entries are the tile's
  cells once each.
* **Sets.** Each lane's H1 sets (`segment.call_words`) over X = A + b and W' = B + c + U: T_0 .. T_(KS-1), O' and
  D_1 .. D_(m-1), m = `segment.n_digits`(KS). X, W', U, the residue sums and the digits are computed once a row or column.
* **Receipt** (CP-W): each lane folds its sets from its keys with fx1 (`PouwFx1Absorb_v2`); after set s, receipt word 2 s is
  the XOR over lanes and chains of y, and word 2 s + 1 of z (`PouwXinFinal_v2` on each half of the state, `PouwXor32_v2`
  across lanes): 2 (KS + m) words.
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

So z exists only through the MMAs whose words are credited, and A, B and the noise: a kernel serving y = dequant(z)
serves the receipted words. The compile pass replaces its ``noise-in-z`` rule for a one-stream scheme
(`compile.Scheme.stream`, here `STREAM`) with ``stream``: every z leaf is made by those Definitions from leaves the
receipt's cone reads.

The scheme takes the Boolean s8 kinds (`KINDS`) on whole-lane grid tiles: a full r × c grid with 4 | r and 8 | c, 32 | K
and KS = K / 32 >= 2; any other dot or tile is `CompileError("scheme")`. `scheme(tile, name)` builds it at another tile;
`SCHEME` is registered at a warp, 32 × 32, and `PouwH1Tile` as a Definition, when `segment` (E2E-120) is on the tree;
`SCHEME_V3` and `PouwH1Tile3` the same. `reference` is the unit on words (``stream=True``: v3's).
"""
from __future__ import annotations

from dataclasses import dataclass
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
#: noise bits a column takes past its coin bits: U's 32 bytes
U_BITS = 256
#: a lane's fold keys: duo's y_0..7, z_0..7, one block
KEY_BITS = 512


@dataclass(frozen=True)
class Layout:
    """One r × c tile at contraction K: its lanes, seed_j's noise blocks in counter mode, and σ_R's key blocks."""

    r: int
    c: int
    k: int

    @property
    def ks(self) -> int:
        return self.k // 32

    @property
    def m(self) -> int:
        return SG.n_digits(self.ks)

    @property
    def sets(self) -> int:
        """The sets a lane absorbs: KS prefix sets, O', D_1 .. D_(m-1)."""
        return self.ks + self.m

    @property
    def row_bits(self) -> int:
        return self.k + 32 * (self.m - 1)

    @property
    def col_bits(self) -> int:
        return self.k + U_BITS

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


# -- the reference -----------------------------------------------------------------------------------------------------------

def _stream(seed_j: bytes, block: int, n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        out += SD.leaves(SD.noise_block(seed_j, block), True)
        block += 1
    return out[:n]


def _value(bits: Sequence[int]) -> int:
    return sum((b & 1) << i for i, b in enumerate(bits))


def noise(seed_j: bytes, lay: Layout) -> tuple[list, list]:
    """The unit's noise from seed_j: per row (dither bits, digit coins), per column (coin bits, U's int8 encodings)."""
    k, m = lay.k, lay.m
    rows = []
    for i in range(lay.r):
        s = _stream(seed_j, lay.row(i), lay.row_bits)
        rows.append((s[:k], [s[k + 32 * d:k + 32 * d + 32] for d in range(m - 1)]))
    cols = []
    for j in range(lay.c):
        s = _stream(seed_j, lay.col(j), lay.col_bits)
        cols.append((s[:k], [u129(_value(s[k + 8 * t:k + 8 * t + 8])) for t in range(32)]))
    return rows, cols


def fold_keys(sigma_r: bytes, lay: Layout) -> list[list[int]]:
    """Each lane's 16 fold keys, its chain's start, from σ_R (`seeds.receipt_seed`)."""
    keys = []
    for ln in range(lay.lanes):
        s = _stream(sigma_r, lay.lane(ln), KEY_BITS)
        keys.append([_value(s[32 * w:32 * w + 32]) for w in range(16)])
    return keys


def lane_sets(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], rows: list, cols: list, lay: Layout,
              ln: int) -> list[list[int]]:
    """Lane ln's absorbed sets on words (`segment.call_words`): T_0 .. T_(KS-1), O', D_1 .. D_(m-1), 32 entries each."""
    R, Cs = lay.rows(ln), lay.cols(ln)
    seg = [range(32 * s, 32 * s + 32) for s in range(lay.ks)]
    xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
    bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
    wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
    cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
    return SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs], [[rows[i][1][d] for i in R] for d in range(lay.m - 1)])


def col_wrap(b_j: Sequence[int], col: tuple) -> list[int]:
    """Column j's wrap bits at every entry of K (`wrap_bits` of B, its coin bit and U, U repeating every 32 entries)."""
    return [wrap_bits(v, col[0][t], col[1][t % 32]) for t, v in enumerate(b_j)]


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
              stream: bool = False) -> tuple[list[int], list[int]]:
    """The unit on words: (z, receipt) for A's r rows and B's c rows (int8 encodings), its salt, position and roots, z at
    `pairs` (the full grid in row-major order by default). `stream`: v3's z, each cell's `one_stream` of its lane's O' and
    D words with its wrap correction and its dither peel (`stream_cell`); else v1's, the exact A Bᵀ cell. Both are A Bᵀ."""
    lay = Layout(len(a), len(b), len(a[0]))
    rows, cols = noise(SD.seed(salt, position, *roots), lay)
    xs = [[(a[i][t] + rows[i][0][t]) & 0xFF for t in range(lay.k)] for i in range(lay.r)] if stream else None
    wbs = [col_wrap(b[j], cols[j]) for j in range(lay.c)] if stream else None
    pis = [(rows[i][0][:lay.k], row_wrap(a[i], rows[i])) for i in range(lay.r)] if stream else None
    pjs = [(b[j], cols[j][0][:lay.k]) for j in range(lay.c)] if stream else None
    keys = fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    ks = lay.ks
    cp = [[0, 0] for _ in range(lay.sets)]
    served: dict[tuple[int, int], int] = {}
    for ln in range(lay.lanes):
        sets = lane_sets(a, b, rows, cols, lay, ln)
        st = list(keys[ln])
        for s, p in enumerate(sets):
            st = SG.absorb(FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
        for e in range(32 if stream else 0):
            cell = (lay.rows(ln)[SG.row_slot(e)], lay.cols(ln)[SG.col_slot(e)])
            served[cell] = stream_cell(sets[ks][e], [d[e] for d in sets[ks + 1:]], xs[cell[0]], wbs[cell[1]],
                                       pis[cell[0]], pjs[cell[1]])
    if pairs is None:
        pairs = [(i, j) for i in range(lay.r) for j in range(lay.c)]
    z = [served[p] for p in pairs] if stream else [SG.seg_dot(0, a[i], b[j]) for i, j in pairs]
    return z, [w & M32 for s in cp for w in s]


# -- the unit ----------------------------------------------------------------------------------------------------------------

def _layout(S: Any) -> Layout:
    return Layout(S.R, S.C, S.EXACT.signature()[0][0][1].n)


def _signature(S: Any) -> tuple:
    (_, a), (_, b) = S.EXACT.signature()[0][:2]
    D, P = SD.types(True)
    return ((("a", Array(S.R, a)), ("b", Array(S.C, b)), ("salt", D), ("position", P), ("roots", Array(2, D))),
            Record(("z", Array(len(S.PAIRS), S.EXACT.signature()[1])), ("receipt", Array(2 * _layout(S).sets, U32))))


def _digits(B: Any, x: Coll, z: list[Coll], lay: Layout) -> list[Coll]:
    """A row's digits A_0 .. A_(m-1) of its residue sums (`segment._call_sets`' per row)."""
    Y = B.batch(SG.PouwSx8, x[0:32], axes=(0,))
    for s in range(1, lay.ks):
        Y = B.batch(SG.PouwResAdd, Y, x[32 * s:32 * s + 32], axes=(0, 0))
    out, R = [], Y
    for d in range(lay.m - 1):
        zd = B.batch(SG.PouwDigitZ, R, z[d], axes=(0, 0)) if d == lay.m - 2 else z[d]
        out.append(B.batch(SG.PouwDigitLo, R, zd, axes=(0, 0)))
        R = B.batch(SG.PouwDigitHi, R, zd, axes=(0, 0))
    out.append(B.batch(SG.PouwLow8, R, axes=(0,)))
    return out


def _xor(B: Any, words: list[Coll]) -> Coll:
    while len(words) > 1:
        words = [B.call(PouwXor32, words[i], words[i + 1]) if i + 1 < len(words) else words[i]
                 for i in range(0, len(words), 2)]
    return words[0]


def _sets(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> tuple[list, dict, list, list, list]:
    """The receipt's refs; per cell (i, j) its lane's O' and D_1 .. D_(m-1) words, as the receipt folds them; each row's X;
    each column's (B, coin bits, U repeated over K), W''s addends; and each row's (A, dither bits), X's."""
    lay = _layout(S)
    k, ks = lay.k, lay.ks
    seed_j, sigma_r = SD.unit_seeds_call(B, salt, position, roots, True)
    blocks = SD.noise_blocks(B, seed_j, lay.blocks, True)
    keys = SD.noise_blocks(B, sigma_r, lay.lanes, True)

    def bits(first: int, n: int) -> Coll:
        return Coll(Array(n, BIT), concat([blocks[first + t].refs for t in range(-(-n // SD.BLOCK_BITS))]).slice(0, n))

    X, A, XIN = [], [], []
    for i in range(lay.r):
        nz = bits(lay.row(i), lay.row_bits)
        X.append(B.batch(SG.PouwI8AddBit, a[i], nz[0:k], axes=(0, 0)))
        XIN.append((a[i], nz[0:k]))
        A.append(_digits(B, X[-1], [nz[k + 32 * d:k + 32 * d + 32] for d in range(lay.m - 1)], lay))
    W, U, NU, WIN = [], [], [], []
    for j in range(lay.c):
        nz = bits(lay.col(j), lay.col_bits)
        U.append(B.batch(PouwU129, Coll(Array(32, I8), nz[k:k + U_BITS].refs), axes=(0,)))
        NU.append(B.batch(SG.PouwI8Neg, U[-1], axes=(0,)))
        wc = B.batch(SG.PouwI8AddBit, b[j], nz[0:k], axes=(0, 0))
        ur = Coll(Array(k, I8), concat([U[-1].refs] * ks))
        W.append(B.batch(SG.PouwI8Add, wc, ur, axes=(0, 0)))
        WIN.append((b[j], nz[0:k], ur))
    rs, cs = [SG.row_slot(e) for e in range(32)], [SG.col_slot(e) for e in range(32)]
    cps: list[list[tuple[Coll, Coll]]] = [[] for _ in range(lay.sets)]
    words: dict[tuple[int, int], list[Coll]] = {}
    for ln in range(lay.lanes):
        R, Cs = lay.rows(ln), lay.cols(ln)
        T, prev = [], None
        for s in range(ks):
            xs = [X[i][32 * s:32 * s + 32] for i in R]
            ws = [W[j][32 * s:32 * s + 32] for j in Cs]
            prev = [B.call(SG.PouwSegDot0, xs[rs[e]], ws[cs[e]]) if prev is None else
                    B.call(SG.PouwSegDot, prev[e], xs[rs[e]], ws[cs[e]]) for e in range(32)]
            T.append(prev)
        op = [B.call(SG.PouwSegDot, T[-1][e], NU[Cs[cs[e]]], A[R[rs[e]]][0]) for e in range(32)]
        D = [[B.call(SG.PouwSegDot0, U[Cs[cs[e]]], A[R[rs[e]]][d]) for e in range(32)] for d in range(1, lay.m)]
        state = Coll(Array(KEY_BITS, BIT), keys[lay.lane(ln)].refs)
        for s, p in enumerate([*T, op, *D]):
            state = B.call(SG.PouwFx1Absorb, state, array_of(p))
            cps[s].append((B.call(SG.PouwXinFinal, state[0:256]), B.call(SG.PouwXinFinal, state[256:512])))
        for e in range(32):
            words[(R[rs[e]], Cs[cs[e]])] = [op[e]] + [d[e] for d in D]
    return [_xor(B, [w[h] for w in ws]).refs for ws in cps for h in range(2)], words, X, WIN, XIN


def _body(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    receipt = _sets(B, S, a, b, salt, position, roots)[0]
    z = [B.call(S.EXACT, a[i], b[j]).refs for i, j in S.PAIRS]
    return Coll(_signature(S)[1], concat(z + receipt))


def _body_v3(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    receipt, words, X, WIN, XIN = _sets(B, S, a, b, salt, position, roots)
    lay = _layout(S)
    f = ONE_STREAM[lay.m]
    wb = {j: B.batch(PouwWrapBits, *WIN[j], axes=(0, 0, 0)) for j in sorted({j for _i, j in S.PAIRS})}
    xa = {i: B.batch(PouwXWrap, *XIN[i], axes=(0, 0)) for i in sorted({i for i, _j in S.PAIRS})}
    z = []
    for i, j in S.PAIRS:
        acc = B.call(f, *words[(i, j)])
        for s in range(lay.ks):
            acc = B.call(PouwWrapSeg, acc, X[i][32 * s:32 * s + 32], wb[j][32 * s:32 * s + 32])
        for s in range(lay.ks):
            seg = slice(32 * s, 32 * s + 32)
            acc = B.call(PouwPeelSeg, acc, X[i][seg], XIN[i][1][seg], xa[i][seg], WIN[j][0][seg], WIN[j][1][seg])
        z.append(acc.refs)
    return Coll(_signature(S)[1], concat(z + receipt))


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


def scheme(tile: tuple[int, int] = (32, 32), name: str = SCHEME, defn: Any = PouwH1Tile) -> C.Scheme:
    """H1 + fx1 as a compile-pass scheme at `tile` (register it with `compile.register`); `defn` `PouwH1Tile3` for the one
    stream (`SCHEME_V3`), whose z the pass checks is recombined from folded words."""
    f = _Unit(name, defn)
    stream = STREAM if defn is PouwH1Tile3 else ()
    return C.Scheme(name, tile, {kind: f for kind in KINDS}, stream=stream)


if SG is not None:
    C.register(scheme())
    C.register(scheme(name=SCHEME_V3, defn=PouwH1Tile3))

__all__ = ["FOLD", "KINDS", "Layout", "ONE_STREAM", "PouwH1Tile", "PouwH1Tile3", "PouwOneStream2", "PouwOneStream2Word",
           "PouwOneStream3", "PouwOneStream3Word", "PouwU129", "PouwU129Word", "PouwWrapBits", "PouwWrapBitsWord", "PouwWrapSeg",
           "PouwWrapSegWord", "PouwPeelSeg", "PouwPeelSegWord", "PouwXWrap", "PouwXWrapWord", "SCHEME", "SCHEME_V3", "STREAM",
           "col_wrap", "fold_keys", "lane_sets", "noise", "one_stream", "peel_seg", "reference", "row_wrap", "scheme",
           "stream_cell", "u129", "wrap_bits", "wrap_seg", "x_wrap"]
