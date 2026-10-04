"""A hidden unit circuit's share of the lincheck fold, in-circuit (recursion): the inner verifier's `folded` (C-Flock's
PROTOCOL.md §10, §16.1) on the committed nonzeros of the unit's `A_0` and `B_0`, at one inner session's coins as constants.

    CombEntry_v1{SL, POINT}      (e: Array<W, Value<1>>) -> Array<REPS, F128>                         one entry's value
    CombChunk_v1{SL, E, POINT}   (acc: Array<REPS, F128>, row: Array<WORDS, Array<16, Value<1>>>) -> Array<REPS, F128>
                                 acc plus the E entries of one com_P row
    CombColumn_v1{SL, COL, POINT}  (a, b: Array<2^SL, Value<1>>) -> Array<REPS, F128>                the dense column's share
        W = 2 SL - 4 bits an entry, WORDS = ceil(E W / 16), F128 = Array<8, Array<16, Value<1>>> (bit k of the element is
        bit k % 16 of word k // 16, `Flock.F128`'s bit order), REPS the points (fast100's two reps), POINT their coins
        (`point_static`)

**The share.** At each rep's coins the inner verifier's fold of a unit circuit of slot log SL (`Flock.Stmt.partialRanges`)
is, at ``s < 64``,

    share[s] = Σ_{c mod 64 = s} eq(r, c >> 6) (α (A_0ᵀ e)[c] + (B_0ᵀ e)[c]),   e[i] = lw[i mod 64] eq(ρ, i >> 6),

with ``ρ`` and ``r`` (the lincheck's ``ts`` reversed) cut to their first SL - 6 coordinates and ``eq(x, b) = Π_j (1 + x_j +
b_j)``, times its ranges' weight. `local_point` folds the weight into ``lw`` (the share is linear in ``lw``), so a `Point` is
the unit's local coins and the share is the template's whole contribution to ``folded`` (`template_fold`; `fold_dense` is
the verifier's literal fallback). It is a sum over the nonzeros. An *entry* is a cell ``(i, c)`` with a nonzero in A, in B or
in both: its ``side`` (bit 0: A, bit 1: B), the row ``i`` and the column's high bits ``u = c >> 6``, W bits LSB first. The
column's low six bits are its *bucket*, public: the entry of ``share`` it adds to. An entry is worth

    E(e) = (α side_0 + side_1) lw[i mod 64] eq(ρ, i >> 6) eq(r, u),

zero at side 0 (padding). The unit's constant column ``COL``, read by many of its rows, is held densely instead: its A and
B bits per row (`CombColumn`), whose share is linear.

**com_P's rows** (`Shape`, `pack`, `unpack`): whole SHA-512 blocks of 64 u16 words, as every registered row. Per hidden
unit circuit, a class-fixed shape (SL, R rows a bucket, COL, ``blocks`` a row, E = `per_row` entries a row): rows ``R s ..
R s + R - 1`` hold bucket ``s``'s entries, at most ``R E``, padded with zero entries, entry ``k`` of a row at its bits ``k W
.. k W + W - 1`` (LSB first, word ``j`` its bits ``16 j ..``); then the dense column, its ``2^SL`` A bits and ``2^SL`` B
bits. Every bit string is a matrix: equal cells cancel (F2), a zero side is no entry. Registered as a registered value
(`verity_flock.com_p`), each chunk unit of V* reads one row at a public position, so the statement hashes it in-circuit and
the verifier opens its path: the entries a chunk reads are the registered ones. A row's hm96 slot (2^18 rows) is paid once
a row, so rows of many blocks bind an entry for less (13 blocks, 605 entries, at SL = 13).

**In-circuit.** The coins are constants, so a coin times a witness bit is linear. ``E`` is a public function of the entry's
bits. Its first factor, over the side and the row's low ``d`` bits (`low_bits`), is linear in the monomials of those bits
(their algebraic normal form): ``2^d - d - 1`` ANDs for the monomials, ``2 (2^d - 1)`` for their products with the two side
bits, shared by every rep. Each remaining bit ``b`` of the row (coin ``ρ_j``) and of ``u`` (coin ``r_j``) is one step ``X <-
X (1 + c) + b X`` (eq(c, b) = 1 + c + b in characteristic 2): the constant product linear, ``b X`` 128 ANDs. So
(`and_count`)

    ANDs per entry = (2^d - d - 1) + 2 (2^d - 1) + 128 REPS (2 SL - 6 - d),   d = 7 at two reps (3,702 at SL = 13)

The builders commit X before every ``every``-th step but the first (128 copy rows a rep), which bounds the forms' length
(their nonzeros) at the price of rows: C-Flock's rows are the ANDs plus the copies (`and_count` counts the ANDs). A chunk is
its accumulator plus E entries' values (linear); a column is linear (no AND). The IR bodies are traced from the builders
(`trace`), where a copy is no gate; `chunk_circuit` and `column_circuit` are the builders' circuits with their copies, for
C-Flock's lowering (`verity_flock.ir_lower.Units.builder`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from operator import xor
from typing import Mapping, Sequence

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.refs import Coll, concat, strided_view
from verity.ir.types import Array

from . import trace as TR
from .forms import ONE, ZERO, Circuit, L

__all__ = ["BLOCK_WORDS", "EVERY", "MOD", "CombChunk", "CombColumn", "CombEntry", "Point", "Shape", "and_count", "build_chunk",
           "build_column", "build_entry", "chunk", "chunk_circuit", "chunk_definition", "column", "column_circuit",
           "column_definition", "decode", "encode", "entries", "entry_bits", "entry_definition", "entry_value", "eq", "eq_table",
           "f128_of", "f128_words", "fold_dense", "gmul", "local_point", "low_bits", "pack", "per_row", "point_static", "points",
           "range_weight", "row_bits", "row_words", "template_fold", "unpack", "values", "words_of"]

BIT = BL.BIT
#: GF(2^128) = F2[x]/(x^128 + x^7 + x^2 + x + 1), bit k of an int the coefficient of x^k (`Flock.F128`)
MOD = (1 << 128) | 0x87
MASK = (1 << 128) - 1
WORD = 16
F128 = Array(8, Array(WORD, BIT))
#: the skip bits: a row's low six index its Lagrange weight, a column's low six are its bucket
SKIP = 6
#: a SHA-512 block of a registered row, in u16 words (`registered.BLOCK`): a com_P row is a whole number of them
BLOCK_WORDS = 64
#: the builders' commit spacing: one copy of X every two steps (rows ~1.4x the ANDs, ~70 nonzeros a row at SL = 13)
EVERY = 2


def gmul(a: int, b: int) -> int:
    p = 0
    while b:
        low = b & -b
        p ^= a * low
        b ^= low
    while p >> 128:
        hi = p >> 128
        p = (p & MASK) ^ hi ^ (hi << 1) ^ (hi << 2) ^ (hi << 7)
    return p


def eq(x: Sequence[int], b: int) -> int:
    """``eq(x, b)`` at the bits of ``b``, bit ``j`` against ``x[j]``."""
    out = 1
    for j, xj in enumerate(x):
        out = gmul(out, xj ^ 1 ^ ((b >> j) & 1))
    return out


def eq_table(x: Sequence[int]) -> list[int]:
    """``eq(x, b)`` for every ``b < 2^|x|`` (`Flock.eqTable`)."""
    t = [1]
    for xj in x:
        t = [gmul(v, xj ^ 1) for v in t] + [gmul(v, xj) for v in t]
    return t


@dataclass(frozen=True)
class Point:
    """One rep's coins as a unit circuit of slot log ``len(rho) + 6`` sees them: α, the 64 skip weights (times the unit's
    ranges' weight, `local_point`), the row coins (the zerocheck's first SL - 6) and the column coins (``ts`` reversed, its
    first SL - 6)."""

    alpha: int
    lw: tuple[int, ...]
    rho: tuple[int, ...]
    rr: tuple[int, ...]

    def __post_init__(self) -> None:
        for k in ("lw", "rho", "rr"):
            object.__setattr__(self, k, tuple(int(v) for v in getattr(self, k)))
        object.__setattr__(self, "alpha", int(self.alpha))
        if len(self.lw) != 1 << SKIP or len(self.rho) != len(self.rr):
            raise ValueError("a point is 64 skip weights and as many row coins as column coins")
        if any(not 0 <= v <= MASK for v in (self.alpha, *self.lw, *self.rho, *self.rr)):
            raise ValueError("a coin is a GF(2^128) element, an int below 2^128")

    @property
    def SL(self) -> int:
        return len(self.rho) + SKIP

    def e(self, i: int) -> int:
        """The row weight ``e[i] = lw[i mod 64] eq(ρ, i >> 6)``."""
        return _tables(self)[0][i]


@lru_cache(maxsize=64)
def _tables(p: Point) -> tuple[list[int], list[int], list[int]]:
    """``e``, ``α e`` over the 2^SL rows and ``eq(r, u)`` over the 2^(SL - 6) column high bits."""
    er = eq_table(p.rho)
    e = [gmul(p.lw[i & 63], er[i >> SKIP]) for i in range(1 << p.SL)]
    return e, [gmul(p.alpha, v) for v in e], eq_table(p.rr)


def local_point(alpha: int, lw: Sequence[int], rho: Sequence[int], ts: Sequence[int], SL: int,
                ranges: Sequence[tuple[int, int]]) -> Point:
    """A unit circuit's point at a block's lincheck coins (α, ``lw`` = the skip weights at z, the zerocheck's ``ρ`` and the
    lincheck's ``ts``, k_log - 6 each), its ranges ``(pos0, count)`` of slots of log SL: ``lw`` times the ranges' weight."""
    rr = tuple(reversed(ts))
    wt = reduce(xor, (range_weight(rho, rr, SL, p0, n) for p0, n in ranges), 0)
    return Point(alpha, [gmul(wt, v) for v in lw], tuple(rho[:SL - SKIP]), rr[:SL - SKIP])


def point_static(ps: Sequence[Point]) -> dict:
    """The points as the ``POINT`` static: per rep, its coins as 32 hex digits (canonical JSON holds no 128-bit integer)."""
    if not ps or len({p.SL for p in ps}) != 1:
        raise ValueError("at least one point, all of one slot log")
    h = lambda v: f"{v:032x}"  # noqa: E731
    return {"alpha": [h(p.alpha) for p in ps], "lw": [[h(v) for v in p.lw] for p in ps],
            "rho": [[h(v) for v in p.rho] for p in ps], "rr": [[h(v) for v in p.rr] for p in ps]}


def points(POINT: Mapping) -> tuple[Point, ...]:
    """The points a ``POINT`` static names (`point_static`'s inverse)."""
    i = lambda s: int(s, 16)  # noqa: E731
    n = len(POINT["alpha"])
    if any(len(POINT[k]) != n for k in ("lw", "rho", "rr")):
        raise ValueError("POINT: one alpha, lw, rho and rr per rep")
    return tuple(Point(i(POINT["alpha"][r]), tuple(map(i, POINT["lw"][r])), tuple(map(i, POINT["rho"][r])),
                       tuple(map(i, POINT["rr"][r]))) for r in range(n))


# --- entries and com_P's rows ------------------------------------------------------------------------------------------------

def _check_sl(SL: int) -> None:
    if isinstance(SL, bool) or not isinstance(SL, int) or not SKIP <= SL <= 27:
        raise ValueError(f"a slot log is an int from {SKIP} to 27 (C-Flock's k_log bound), not {SL!r}")


def entry_bits(SL: int) -> int:
    """W: side (2), row (SL), column high bits (SL - 6)."""
    _check_sl(SL)
    return 2 * SL - 4


def per_row(SL: int, blocks: int = 1) -> int:
    """The entries a com_P row of ``blocks`` SHA-512 blocks holds."""
    return blocks * BLOCK_WORDS * WORD // entry_bits(SL)


def encode(SL: int, side: int, i: int, c: int) -> int:
    """Cell ``(i, c)``'s entry (its bucket ``c mod 64`` is not in it)."""
    if not 0 <= side < 4 or not 0 <= i < 1 << SL or not 0 <= c < 1 << SL:
        raise ValueError(f"an entry is a side below 4 and a cell of a 2^{SL} block")
    return side | i << 2 | (c >> SKIP) << (2 + SL)


def decode(SL: int, e: int) -> tuple[int, int, int]:
    """``(side, i, u)``."""
    return e & 3, (e >> 2) & ((1 << SL) - 1), e >> (2 + SL)


def entry_value(p: Point, e: int) -> int:
    side, i, u = decode(p.SL, e)
    w = (p.alpha if side & 1 else 0) ^ (side >> 1)
    return gmul(gmul(w, p.e(i)), eq(p.rr, u)) if w else 0


def values(ps: Sequence[Point], es: Sequence[int]) -> list[int]:
    """Each rep's sum of the entries' values (`CombEntry` summed), column by column: ``Σ_u eq(r, u) Σ_{entries at u} ...``."""
    out = []
    for p in ps:
        e, ae, er = _tables(p)
        acc = [0] * len(er)
        for x in es:
            side, i, u = decode(p.SL, x)
            if side & 1:
                acc[u] ^= ae[i]
            if side & 2:
                acc[u] ^= e[i]
        out.append(reduce(xor, (gmul(a, b) for a, b in zip(acc, er) if a), 0))
    return out


def row_words(SL: int, E: int) -> int:
    """The words E entries take (`CombChunk`'s row)."""
    return -(-E * entry_bits(SL) // WORD)


def row_bits(words: Sequence[int]) -> list[int]:
    return [(int(w) >> j) & 1 for w in words for j in range(WORD)]


def words_of(bits: Sequence[int], n: int) -> list[int]:
    """``bits`` (LSB first) as ``n`` u16 words, zero past the bits."""
    if len(bits) > WORD * n:
        raise ValueError(f"{len(bits)} bits do not fit {n} words")
    out = [0] * n
    for k, b in enumerate(bits):
        out[k // WORD] |= (b & 1) << (k % WORD)
    return out


def _row_entries(SL: int, E: int, words: Sequence[int]) -> list[int]:
    W, bits = entry_bits(SL), row_bits(words)
    return [sum(bits[k * W + j] << j for j in range(W)) for k in range(E)]


def chunk(ps: Sequence[Point], E: int, acc: Sequence[int], words: Sequence[int]) -> list[int]:
    """`CombChunk`'s reference: per rep, ``acc`` plus the row's first E entries' values."""
    return [a ^ v for a, v in zip(acc, values(ps, _row_entries(ps[0].SL, E, words)))]


def column(ps: Sequence[Point], COL: int, a: Sequence[int], b: Sequence[int]) -> list[int]:
    """`CombColumn`'s reference: ``eq(r, COL >> 6) Σ_i (α a_i + b_i) e[i]`` per rep."""
    out = []
    for p in ps:
        e, ae, er = _tables(p)
        acc = 0
        for i, (ai, bi) in enumerate(zip(a, b)):
            if ai & 1:
                acc ^= ae[i]
            if bi & 1:
                acc ^= e[i]
        out.append(gmul(acc, er[COL >> SKIP]))
    return out


def entries(rows_a: Sequence[Sequence[int]], rows_b: Sequence[Sequence[int]], SL: int,
            COL: int | None = None) -> tuple[list[list[int]], list[int], list[int]]:
    """A unit circuit's matrices (row ``i`` of A is the columns ``rows_a[i]`` reads, each an odd number of times) as its 64
    buckets of entries, in (row, column) order, and its column ``COL``'s A and B bits (all zero for none)."""
    n = 1 << SL
    if len(rows_a) > n or len(rows_b) > n:
        raise ValueError(f"more than 2^{SL} rows")
    cells: dict[tuple[int, int], int] = {}
    for side, rows in ((1, rows_a), (2, rows_b)):
        for i, r in enumerate(rows):
            for c in r:
                c = int(c)
                if not 0 <= c < n:
                    raise ValueError(f"row {i} reads column {c}, outside the 2^{SL} block")
                cells[(i, c)] = cells.get((i, c), 0) ^ side
    a, b = [0] * n, [0] * n
    buckets: list[list[int]] = [[] for _ in range(1 << SKIP)]
    for (i, c), side in sorted(cells.items()):
        if not side:
            continue
        if c == COL:
            a[i], b[i] = side & 1, side >> 1
        else:
            buckets[c & 63].append(encode(SL, side, i, c))
    return buckets, a, b


@dataclass(frozen=True)
class Shape:
    """A hidden unit circuit's class-fixed shape: slot log, rows a bucket, its dense column (or none), SHA-512 blocks a row,
    entries a row (default: as many as fit)."""

    SL: int
    R: int
    COL: int | None = None
    blocks: int = 1
    E: int | None = None

    def __post_init__(self) -> None:
        _check_sl(self.SL)
        if isinstance(self.blocks, bool) or not isinstance(self.blocks, int) or self.blocks < 1:
            raise ValueError(f"a row is at least one SHA-512 block, not {self.blocks!r}")
        if self.E is None:
            object.__setattr__(self, "E", per_row(self.SL, self.blocks))
        top = per_row(self.SL, self.blocks)
        if not 1 <= self.E <= top or self.R < 0 or (self.COL is not None and not 0 <= self.COL < 1 << self.SL):
            raise ValueError(f"a shape is 1 to {top} entries a row, rows a bucket, and a column inside the block")

    @property
    def W(self) -> int:
        return entry_bits(self.SL)

    @property
    def words(self) -> int:
        return self.blocks * BLOCK_WORDS

    @property
    def bound(self) -> int:
        """Entries a bucket."""
        return self.R * self.E

    @property
    def column_rows(self) -> int:
        return 0 if self.COL is None else -(-(2 << self.SL) // (WORD * self.words))

    @property
    def rows(self) -> int:
        return (1 << SKIP) * self.R + self.column_rows

    @property
    def nnz(self) -> int:
        """The hidden entries as `(side, row, column, value)` with one side each (`verity_flock.com_p.hidden_entries`), padding
        included: two a slot, two a column row."""
        return 2 * (1 << SKIP) * self.bound + (0 if self.COL is None else 2 << self.SL)


def pack(shape: Shape, buckets: Sequence[Sequence[int]], a: Sequence[int] = (), b: Sequence[int] = ()) -> list[list[int]]:
    """com_P's rows: each bucket's entries padded to the bound, E a row, then the column's A bits and B bits."""
    W, E = shape.W, shape.E
    if len(buckets) != 1 << SKIP:
        raise ValueError("64 buckets")
    out = []
    for s, es in enumerate(buckets):
        if len(es) > shape.bound:
            raise ValueError(f"bucket {s} holds {len(es)} entries, over the bound {shape.bound}")
        es = list(es) + [0] * (shape.bound - len(es))
        for r in range(shape.R):
            out.append(words_of([(e >> j) & 1 for e in es[r * E:(r + 1) * E] for j in range(W)], shape.words))
    if shape.COL is not None:
        bits = [x & 1 for x in a] + [x & 1 for x in b]
        if len(bits) != 2 << shape.SL:
            raise ValueError(f"the column's bits are 2^{shape.SL} A bits and 2^{shape.SL} B bits")
        step = WORD * shape.words
        out += [words_of(bits[k:k + step], shape.words) for k in range(0, len(bits), step)]
    elif any(a) or any(b):
        raise ValueError("a shape without a dense column holds no column bits")
    return out


def unpack(shape: Shape, rows: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int], list[int]]:
    """The buckets (every slot, padding included) and the column bits a sequence of com_P's rows holds."""
    if len(rows) != shape.rows or any(len(r) != shape.words for r in rows):
        raise ValueError(f"a shape's rows are {shape.rows} of {shape.words} words")
    R = shape.R
    buckets = [[e for r in rows[s * R:(s + 1) * R] for e in _row_entries(shape.SL, shape.E, r)] for s in range(1 << SKIP)]
    n = 1 << shape.SL
    bits = [x for r in rows[(1 << SKIP) * R:] for x in row_bits(r)]
    a, b = (bits[:n], bits[n:2 * n]) if shape.COL is not None else ([0] * n, [0] * n)
    return buckets, a, b


# --- the fold --------------------------------------------------------------------------------------------------------------

def template_fold(p: Point, sums: Sequence[int], col: int = 0, COL: int | None = None) -> list[int]:
    """The unit circuit's share at one rep: bucket ``s``'s sum at ``s``, the dense column's share (`column`) at ``COL mod
    64``."""
    out = [int(v) for v in sums]
    if COL is not None:
        out[COL & 63] ^= col
    return out


def range_weight(rho: Sequence[int], rr: Sequence[int], SL: int, pos0: int, count: int) -> int:
    """A range's weight at a block's coins (``rho``, ``rr`` its k_log - 6 each): ``Σ_k eq(ρ[SL - 6 ..], q) eq(r[SL - 6 ..], q)``
    over its slots ``q = pos0 + k`` (`Flock.Stmt.partialRanges`' ``wt``)."""
    h = SL - SKIP
    acc = 0
    for q in range(pos0, pos0 + count):
        acc ^= gmul(eq(rho[h:], q), eq(rr[h:], q))
    return acc


def fold_dense(alpha: int, beta: int, lw: Sequence[int], rho: Sequence[int], ts: Sequence[int], rows_a: Sequence[Sequence[int]],
               rows_b: Sequence[Sequence[int]], pin: int) -> list[int]:
    """The lincheck fold as the verifier computes it on a block it does not read the structure of (`Flock.Stmt.folded`'s
    fallback): ``comb = α A_0ᵀ e + B_0ᵀ e`` over the block's ``2^(|ρ| + 6)`` columns, β added at the pin, halved at each ``t`` of
    ``ts`` in turn, top half onto bottom (`Flock.halve`)."""
    n = 1 << (len(rho) + SKIP)
    er = eq_table(rho)
    e = [gmul(lw[i & 63], er[i >> SKIP]) for i in range(n)]
    comb = [0] * n
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        ae = gmul(alpha, e[i])
        for c in ra:
            comb[c] ^= ae
        for c in rb:
            comb[c] ^= e[i]
    comb[pin] ^= beta
    for t in ts:
        h = len(comb) // 2
        comb = [comb[i] ^ gmul(t, comb[i + h] ^ comb[i]) for i in range(h)]
    return comb


# --- the circuits ----------------------------------------------------------------------------------------------------------

def _ands(SL: int, reps: int, d: int) -> int:
    return (1 << d) - d - 1 + 2 * ((1 << d) - 1) + 128 * reps * (2 * SL - SKIP - d)


def low_bits(SL: int, reps: int) -> int:
    """d: the row's low bits the first factor takes (at least the six of its Lagrange weight), the fewest ANDs."""
    _check_sl(SL)
    return min(range(SKIP, SL + 1), key=lambda d: (_ands(SL, reps, d), d))


def and_count(SL: int, reps: int) -> int:
    """`CombEntry`'s ANDs (the module docstring), the copies apart."""
    return _ands(SL, reps, low_bits(SL, reps))


@lru_cache(maxsize=None)
def _times(c: int) -> tuple[tuple[int, ...], ...]:
    """Multiplication by ``c`` as F2 rows: output bit k is the XOR of input bits ``_times(c)[k]``."""
    cols = [gmul(c, 1 << l) for l in range(128)]
    return tuple(tuple(l for l in range(128) if (cols[l] >> k) & 1) for k in range(128))


def _sum(fs):
    fs = list(fs)
    return reduce(xor, fs) if fs else ZERO


def _lin(fs):
    """The XOR of many forms in one pass (an untracked form: the trace writes it as a balanced tree over its bits)."""
    s, c = set(), 0
    for f in fs:
        s ^= f.s
        c ^= f.c
    return L(frozenset(s), c)


def _commit(C, X):
    return [x if x.is_const() or (len(x.s) == 1 and x.c == 0) else C._new("and", (x, ONE)) for x in X]


def _anf(table: Sequence[int]) -> list[int]:
    """The algebraic normal form of a table over ``log2 len`` bits: coefficient ``S`` is the XOR of the table over ``S``'s
    subsets."""
    t = list(table)
    h = 1
    while h < len(t):
        for x in range(len(t)):
            if x & h:
                t[x] ^= t[x ^ h]
        h <<= 1
    return t


def build_entry(C, SL: int, ps: Sequence[Point], e: Sequence, every: int = EVERY) -> list[list]:
    """Each point's ``E(e)``, 128 forms LSB first, from the entry's W forms; X is committed before every ``every``-th step but
    the first."""
    W, d = entry_bits(SL), low_bits(SL, len(ps))
    if len(e) != W or any(p.SL != SL for p in ps):
        raise ValueError(f"an entry of {W} bits at points of slot log {SL}")
    s0, s1, row, u = e[0], e[1], list(e[2:2 + SL]), list(e[2 + SL:])
    m = [ONE] * (1 << d)
    for S in range(1, 1 << d):
        top = S.bit_length() - 1
        rest = S ^ (1 << top)
        m[S] = row[top] if not rest else C.AND(m[rest], row[top])
    m0, m1 = [C.AND(s0, x) for x in m], [C.AND(s1, x) for x in m]
    out = []
    for p in ps:
        g = _anf([gmul(p.lw[x & 63], eq(p.rho[:d - SKIP], x >> SKIP)) for x in range(1 << d)])
        ga = [gmul(p.alpha, v) for v in g]
        X = [_sum([m0[S] for S in range(1 << d) if (ga[S] >> k) & 1] + [m1[S] for S in range(1 << d) if (g[S] >> k) & 1])
             for k in range(128)]
        steps = [(row[j], p.rho[j - SKIP]) for j in range(d, SL)] + [(u[j], p.rr[j]) for j in range(SL - SKIP)]
        for j, (b, c) in enumerate(steps):
            if j and j % every == 0:
                X = _commit(C, X)
            M = _times(c ^ 1)
            X = [_sum(X[l] for l in M[k]) ^ C.AND(b, X[k]) for k in range(128)]
        out.append(X)
    return out


def build_chunk(C, SL: int, ps: Sequence[Point], acc: Sequence, bits: Sequence, E: int, every: int = EVERY) -> list[list]:
    """Each point's ``acc`` plus the values of the E entries in ``bits`` (``E W`` forms or more, LSB first)."""
    W = entry_bits(SL)
    if len(acc) != 128 * len(ps) or len(bits) < E * W:
        raise ValueError(f"an accumulator of {128 * len(ps)} bits and at least {E * W} entry bits")
    out = [list(acc[128 * r:128 * r + 128]) for r in range(len(ps))]
    for k in range(E):
        for r, X in enumerate(build_entry(C, SL, ps, bits[k * W:(k + 1) * W], every)):
            out[r] = [x ^ y for x, y in zip(out[r], X)]
    return out


def build_column(C, SL: int, ps: Sequence[Point], COL: int, a: Sequence, b: Sequence) -> list[list]:
    """Each point's ``eq(r, COL >> 6) Σ_i (α a_i + b_i) e[i]``, 128 forms LSB first (linear)."""
    if len(a) != 1 << SL or len(b) != 1 << SL or not 0 <= COL < 1 << SL:
        raise ValueError(f"a column is 2^{SL} A bits and 2^{SL} B bits, at a column of the block")
    out = []
    for p in ps:
        e, ae, er = _tables(p)
        w = er[COL >> SKIP]
        ew, aew = [gmul(w, v) for v in e], [gmul(w, v) for v in ae]
        out.append([_lin([a[i] for i in range(1 << SL) if (aew[i] >> k) & 1] + [b[i] for i in range(1 << SL) if (ew[i] >> k) & 1])
                    for k in range(128)])
    return out


def _words16(X: list) -> list[list]:
    return [X[WORD * w:WORD * w + WORD] for w in range(8)]


def chunk_circuit(SL: int, E: int, ps: Sequence[Point], every: int = EVERY) -> Circuit:
    """`CombChunk` as a C-Flock unit circuit, its copies kept: inputs ``in0 .. in{8 REPS - 1}`` the accumulator's u16 words,
    then ``in{8 REPS} ..`` the row's ``row_words(SL, E)``; outputs ``out0 .. out{8 REPS - 1}``."""
    C, n = Circuit(cse=True), 8 * len(ps)
    acc = [x for j in range(n) for x in C.input(f"in{j}", WORD)]
    bits = [x for j in range(row_words(SL, E)) for x in C.input(f"in{n + j}", WORD)]
    out = [x for X in build_chunk(C, SL, ps, acc, bits, E, every) for x in X]
    for j in range(n):
        C.output(f"out{j}", out[WORD * j:WORD * j + WORD])
    return C


def column_circuit(SL: int, COL: int, ps: Sequence[Point]) -> Circuit:
    """`CombColumn` as a C-Flock unit circuit: inputs ``in0 .. in{2^(SL + 1) / 16 - 1}`` the A bits' then the B bits' u16
    words; outputs ``out0 .. out{8 REPS - 1}``."""
    C, n = Circuit(cse=True), 1 << (SL - 4)
    ws = [C.input(f"in{j}", WORD) for j in range(2 * n)]
    a, b = [x for w in ws[:n] for x in w], [x for w in ws[n:] for x in w]
    out = [x for X in build_column(C, SL, ps, COL, a, b) for x in X]
    for j in range(8 * len(ps)):
        C.output(f"out{j}", out[WORD * j:WORD * j + WORD])
    return C


def _check(S, *names) -> tuple[Point, ...]:
    _check_sl(S.SL)
    ps = points(S.POINT)
    if not ps or any(p.SL != S.SL for p in ps):
        raise ValueError(f"POINT's coins are not slot log {S.SL}'s ({S.SL - SKIP} row and column coins a rep)")
    for k in names:
        v = getattr(S, k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{k} is an int >= 1, not {v!r}")
    return ps


def _key(POINT: Mapping) -> tuple:
    return tuple((k, tuple(tuple(x) if isinstance(x, list) else x for x in POINT[k])) for k in ("alpha", "lw", "rho", "rr"))


def _unkey(key: tuple) -> dict:
    return {k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in key}


def _entry_sig(S):
    ps = _check(S)
    return (("e", Array(entry_bits(S.SL), BIT)),), Array(len(ps), F128)


@lru_cache(maxsize=None)
def _traced_entry(SL: int, key: tuple) -> TR.Traced:
    ps = points(_unkey(key))
    return TR.trace(lambda C, e: [w for X in build_entry(C, SL, ps, e) for w in _words16(X)], [Array(entry_bits(SL), BIT)],
                    Array(len(ps), F128))


def _entry_body(B, S, e):
    params, ret = _entry_sig(S)
    return TR.emit(_traced_entry(S.SL, _key(S.POINT)), [t for _, t in params], ret, B, e)


CombEntry = CompositeDefinition(
    "CombEntry", 1, ("SL", "POINT"), _entry_sig, _entry_body, conformance="exact-model-tested",
    doc="One entry (side, row, column high bits) of a hidden unit circuit of slot log SL: its share of the lincheck fold at "
        "each rep's coins POINT, (alpha side_0 + side_1) lw[i mod 64] eq(rho, i >> 6) eq(r, u)")


def _chunk_sig(S):
    ps = _check(S, "E")
    return (("acc", Array(len(ps), F128)), ("row", Array(row_words(S.SL, S.E), Array(WORD, BIT)))), Array(len(ps), F128)


def _chunk_body(B, S, acc, row):
    ps = points(S.POINT)
    W, M = entry_bits(S.SL), len(ps) * 128
    es = Coll(Array(S.E, Array(W, BIT)), row.refs.slice(0, S.E * W))
    vals = B.batch(bind(CombEntry, SL=S.SL, POINT=S.POINT), es, axes=(0,))
    both = Coll(Array(M * (S.E + 1), BIT), concat([vals.refs, acc.refs]))
    cols = Coll(Array(M, Array(S.E + 1, BIT)), strided_view(both, 0, S.E + 1, 1, M, M * (S.E + 1)))
    return Coll(Array(len(ps), F128), B.batch(BL.xor_gate(S.E + 1), cols, axes=(0,)).refs)


CombChunk = CompositeDefinition(
    "CombChunk", 1, ("SL", "E", "POINT"), _chunk_sig, _chunk_body, conformance="exact-model-tested",
    doc="A running sum acc plus the E entries (W = 2 SL - 4 bits each, LSB first) of one com_P row of a hidden unit circuit "
        "of slot log SL: their CombEntry values at each rep's coins POINT")


def _column_sig(S):
    ps = _check(S)
    if isinstance(S.COL, bool) or not isinstance(S.COL, int) or not 0 <= S.COL < 1 << S.SL:
        raise ValueError(f"COL is a column of the 2^{S.SL} block, not {S.COL!r}")
    return (("a", Array(1 << S.SL, BIT)), ("b", Array(1 << S.SL, BIT))), Array(len(ps), F128)


@lru_cache(maxsize=None)
def _traced_column(SL: int, COL: int, key: tuple) -> TR.Traced:
    ps = points(_unkey(key))
    t = Array(1 << SL, BIT)
    return TR.trace(lambda C, a, b: [w for X in build_column(C, SL, ps, COL, a, b) for w in _words16(X)], [t, t],
                    Array(len(ps), F128))


def _column_body(B, S, a, b):
    params, ret = _column_sig(S)
    return TR.emit(_traced_column(S.SL, S.COL, _key(S.POINT)), [t for _, t in params], ret, B, a, b)


CombColumn = CompositeDefinition(
    "CombColumn", 1, ("SL", "COL", "POINT"), _column_sig, _column_body, conformance="exact-model-tested",
    doc="A hidden unit circuit's dense column COL (slot log SL): its A bits a and B bits b per row, its share of the lincheck "
        "fold at each rep's coins POINT, eq(r, COL >> 6) sum_i (alpha a_i + b_i) lw[i mod 64] eq(rho, i >> 6)")


def f128_words(v: int) -> list[int]:
    """An element as `F128`'s eight u16 words, low first."""
    return [(v >> (WORD * w)) & 0xFFFF for w in range(8)]


def f128_of(words: Sequence[int]) -> int:
    return sum(int(w) << (WORD * j) for j, w in enumerate(words))


def entry_definition(SL: int, ps: Sequence[Point]):
    return bind(CombEntry, SL=SL, POINT=point_static(ps))


def chunk_definition(SL: int, E: int, ps: Sequence[Point]):
    return bind(CombChunk, SL=SL, E=E, POINT=point_static(ps))


def column_definition(SL: int, COL: int, ps: Sequence[Point]):
    return bind(CombColumn, SL=SL, COL=COL, POINT=point_static(ps))
