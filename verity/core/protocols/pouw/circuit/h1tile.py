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

The scheme takes the Boolean s8 kinds (`KINDS`) on whole-lane grid tiles: a full r × c grid with 4 | r and 8 | c, 32 | K
and KS = K / 32 >= 2; any other dot or tile is `CompileError("scheme")`. `scheme(tile, name)` builds it at another tile;
`SCHEME` is registered at a warp, 32 × 32, and `PouwH1Tile` as a Definition, when `segment` (E2E-120) is on the tree.
`reference` is the unit on words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, add, or_reduce
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
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
FOLD = "fx1"
#: the dot kinds the unit takes: the s8 families on bits
KINDS = ("GemmAccumulatorS8_v2", "S8Dot_v2")
BIT = BL.BIT
I8, U32 = Array(8, BIT), Array(32, BIT)
M32 = (1 << 32) - 1
#: noise bits a column takes past its coin bits: U's 32 bytes
U_BITS = 256
#: the products' noise on a row of X (its dither bits b, an entry each; the digit coins are the receipt's) and on a column
#: of W (its coin bits c, an entry each, and U): the s8 kinds' form for the work floor (`compile.Scheme.noise`)
H1_ROWS = C.RowNoise(a=(0, 1), b=(U_BITS, 1))
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


def reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes, position: int,
              roots: tuple[bytes, bytes], pairs: Sequence[tuple[int, int]] | None = None) -> tuple[list[int], list[int]]:
    """The unit on words: (z, receipt) for A's r rows and B's c rows (int8 encodings), its salt, position and roots, z at
    `pairs` (the full grid in row-major order by default)."""
    lay = Layout(len(a), len(b), len(a[0]))
    rows, cols = noise(SD.seed(salt, position, *roots), lay)
    keys = fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    ks, m = lay.ks, lay.m
    cp = [[0, 0] for _ in range(lay.sets)]
    for ln in range(lay.lanes):
        R, Cs = lay.rows(ln), lay.cols(ln)
        seg = [range(32 * s, 32 * s + 32) for s in range(ks)]
        xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
        bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
        wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
        cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
        st = list(keys[ln])
        for s, p in enumerate(SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs],
                                            [[rows[i][1][d] for i in R] for d in range(m - 1)])):
            st = SG.absorb(FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
    if pairs is None:
        pairs = [(i, j) for i in range(lay.r) for j in range(lay.c)]
    return [SG.seg_dot(0, a[i], b[j]) for i, j in pairs], [w & M32 for s in cp for w in s]


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


def _body(B: Any, S: Any, a: Coll, b: Coll, salt: Coll, position: Coll, roots: Coll) -> Coll:
    lay = _layout(S)
    k, ks = lay.k, lay.ks
    seed_j, sigma_r = SD.unit_seeds_call(B, salt, position, roots, True)
    blocks = SD.noise_blocks(B, seed_j, lay.blocks, True)
    keys = SD.noise_blocks(B, sigma_r, lay.lanes, True)

    def bits(first: int, n: int) -> Coll:
        return Coll(Array(n, BIT), concat([blocks[first + t].refs for t in range(-(-n // SD.BLOCK_BITS))]).slice(0, n))

    X, A = [], []
    for i in range(lay.r):
        nz = bits(lay.row(i), lay.row_bits)
        X.append(B.batch(SG.PouwI8AddBit, a[i], nz[0:k], axes=(0, 0)))
        A.append(_digits(B, X[-1], [nz[k + 32 * d:k + 32 * d + 32] for d in range(lay.m - 1)], lay))
    W, U, NU = [], [], []
    for j in range(lay.c):
        nz = bits(lay.col(j), lay.col_bits)
        U.append(B.batch(PouwU129, Coll(Array(32, I8), nz[k:k + U_BITS].refs), axes=(0,)))
        NU.append(B.batch(SG.PouwI8Neg, U[-1], axes=(0,)))
        wc = B.batch(SG.PouwI8AddBit, b[j], nz[0:k], axes=(0, 0))
        W.append(B.batch(SG.PouwI8Add, wc, Coll(Array(k, I8), concat([U[-1].refs] * ks)), axes=(0, 0)))
    rs, cs = [SG.row_slot(e) for e in range(32)], [SG.col_slot(e) for e in range(32)]
    cps: list[list[tuple[Coll, Coll]]] = [[] for _ in range(lay.sets)]
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
    receipt = [_xor(B, [w[h] for w in words]).refs for words in cps for h in range(2)]
    z = [B.call(S.EXACT, a[i], b[j]).refs for i, j in S.PAIRS]
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


@dataclass(frozen=True)
class _Unit:
    """The scheme's factory: `compile.Scheme.units`' per kind."""

    name: str

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
        return bind(PouwH1Tile, EXACT=dot.exact, R=tile.r, C=tile.c, PAIRS=tile.pairs)


def scheme(tile: tuple[int, int] = (32, 32), name: str = SCHEME) -> C.Scheme:
    """H1 + fx1 as a compile-pass scheme at `tile` (register it with `compile.register`)."""
    f = _Unit(name)
    return C.Scheme(name, tile, {kind: f for kind in KINDS}, noise={kind: H1_ROWS for kind in KINDS})


if SG is not None:
    C.register(scheme())

__all__ = ["FOLD", "H1_ROWS", "KINDS", "Layout", "PouwH1Tile", "PouwU129", "PouwU129Word", "SCHEME", "fold_keys", "noise",
           "reference", "scheme", "u129"]
