"""Dither's no-fold proof block for PoUW's filler (CMP-172, option (C)): the Definition a sampled cell-segment's proof
passes, the per-call pad and the coins it reads, and the CPU reference a units kernel's words and hiding leaves are checked
against on sampled cells. The protocol is Dither v3.2 (`verity/core/protocols/pouw/accel/PROTOCOL.md`); the guarantee it
instantiates is `Pouw.Guarantees.Dither.WorkSoundNoFold`, whose words are a no-fold round's prefix sums.

**A filler call** is Dither's call O = (A + d_A)(B + d_B)ᵀ with A the call's pad: A is m × K, B is the n × K stack of the
period's registered weights that share K (stacked along N, so one pad a call serves all of them). Its noised operands are
X = wrap8(A + d_A + V) and W′ = wrap8(B + d_B + U π_τ), τ = k div L′, and its committed words are, per cell (i, j),

    T_s = sum over k < min(s L, K) of X_ik W′_jk  mod 2^16,   s = 1 .. S = ceil(K / L),

Dither's 16-bit prefix sums, one a segment of L inner indices (L = t L′; L = 4,096 or 8,192 for the filler). The filler
has no served output, so it commits no O and no binding identity: its words are exactly `NoFold`'s, at 16 bits.

**The coins** are one live coin a call, read after the window's salt (`call_seeds`): seed_x = SHA-512(`SEEDS_TAG` ‖ salt ‖
coin ‖ x) for each stream x of `STREAMS`, and each stream is SHA-512 in counter mode, ``prg(seed, x, idx) = SHA-512(seed ‖
x ‖ be64(idx))`` (x NUL-padded to 8 bytes; one compression a 64-byte block), read as V31-GPU's prover reads its seeds
(`benchmarks/pouw/accel_v31/v31.py` on CMP-159):

* ``pad``: A[i, k] is byte k of blocks i ceil(K / 64) .. of the pad stream, as an int8 (`pad_rows`). A pad is derived per
  call: a fixed public pad lets a prover table A's subset sums once a period (four Russians), and A Bᵀ is then a period
  constant (FOLD-SHAPES, CMP-172 8adac603).
* ``dA``, ``dB``: d[i, k] is bit k mod 512 (least significant first) of block i ceil(K / 512) + k div 512.
* ``V``, ``U``: row r's entry c is ((w 161) >> 32) - 80 for w the little-endian u32 word r L′ + c of the stream, uniform on
  ±80 within 161 / 2^32.
* ``P``: π_τ is Fisher–Yates over L′ with words τ L′ + c: c from L′ - 1 down to 1, x = (w (c + 1)) >> 32, swap c and x.

The noise a proof block reads is, per inner index k, ``na = d_A + V[k mod L′]`` and ``nb = d_B + U[π_τ(k mod L′)]``, each
in [-80, 81] and so one int8 (`noise_a`, `noise_b`): the verifier computes them from the coins, and the block takes them as
public inputs, so the circuit holds no permutation.

**The proof block** (`PouwDitherBlock_v1{L, FIRST}`) is one cell (i, j) and one segment s of length L (the last segment's
own length where L doesn't divide K):

    (a: Array<L, i8>, na: Array<L, i8>, b: Array<L, i8>, nb: Array<L, i8>, [t0: u16,] t1: u16) -> ok: bit

with a the pad's row segment A[i, seg] (public: it is a function of the coin), na and nb the segment's noise (public), b
the weights' row segment B[j, seg] and t0 = T_{s-1}, t1 = T_s (private, opened from their hiding leaves). It forms
X = a + na and W′ = b + nb mod 2^8 (`segment.PouwI8Add_v2`), sums X W′ᵀ over the segment 32 entries at a time
(`segment.PouwSegDot0_v2`, then `PouwSegDot_v2`), and returns 1 exactly when t1 ≡ t0 + that sum (mod 2^16)
(`PouwDitherStep_v2`). At FIRST (s = 1) there is no t0: T_0 = 0 (`PouwDitherFirst_v2`). Its only output is the pass bit.

**The leaves** (`Shape`) are V31-GPU's, without its O rows: output tiles of 128 × 128 cells (tile tm (n / 128) + tn), a
tile's cells row-major (c = 128 r + col); per tile and segment the tile's 16,384 T_s as little-endian u16 (32 KiB), cut
into rows of `row` bytes (1 KiB by default, 512 words), so leaf t S (32768 / row) + (s - 1)(32768 / row) + r is bytes
[r row, (r + 1) row) of tile t's segment-s words. Each row is an ``hm96-sha512/row/v2`` leaf (`leaf_h`) in one
frame-v3-sha512 tree a call, under `Shape.domain`; its salt is the prover's (`prf_salt`, commit_gpu's PRF keyed by the
prover's private Ks, is what V31-GPU's committer uses).

The reference: `words` is a call's T (S × m × n); `cell_words` one cell's from its pad row and weight row alone;
`words_buffer`, `row_bytes` and `leaf_row` the leaf rows, the last from only the cells it holds; `commit` the tree; and
`block_inputs` a proof block's inputs to the Definition, which `check_block` evaluates in the clear.
"""
from __future__ import annotations

import hashlib
import struct
from array import array
from dataclasses import dataclass, field
from operator import mul
from typing import Any, Callable, Iterable, Mapping, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, add, and_reduce
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, primitive
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest

from . import segment as SG

#: the words tree's identity tag (`Shape.domain`)
TAG = "verity/pouw/dither/fill/v1"
SEEDS_TAG = b"verity/pouw/dither/fill-seeds/v1\0"
#: a call's coin streams, each from its own seed (`call_seeds`)
STREAMS = ("pad", "dA", "dB", "U", "V", "P")
TILE = 128
#: HYB noise's bound: U and V are uniform on ±NOISE
NOISE = 80
#: the bytes of a word tile-segment: 128 × 128 u16
TILE_WORD_BYTES = 2 * TILE * TILE
M16, M32 = (1 << 16) - 1, (1 << 32) - 1
BIT = BL.BIT
I8, U16, U32 = Array(8, BIT), Array(16, BIT), Array(32, BIT)
INST = hm96.SHA512
KEY = INST.default_key


# -- the shape -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """A filler call: m pad rows against n stacked weight rows at contraction k, HYB noise of period lp (L′), segments of L
    inner indices, and leaf rows of `row` bytes."""

    m: int
    n: int
    k: int
    lp: int
    L: int
    row: int = 1024

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.m % TILE or self.n % TILE:
            raise ValueError(f"m and n are whole tiles of {TILE}, not {self.m} × {self.n}")
        if self.lp <= 0 or self.lp % 32 or self.k % self.lp:
            raise ValueError(f"L′ is a multiple of 32 dividing K, not L′ {self.lp} at K {self.k}")
        if self.L <= 0 or self.L % self.lp:
            raise ValueError(f"L is t L′, a multiple of L′ {self.lp}, not {self.L}")
        if self.row <= 0 or self.row % 16 or TILE_WORD_BYTES % self.row:
            raise ValueError(f"a row is a multiple of 16 bytes dividing {TILE_WORD_BYTES}, not {self.row}")

    @property
    def t(self) -> int:
        return self.L // self.lp

    @property
    def S(self) -> int:
        return -(-self.k // self.L)

    @property
    def tiles(self) -> int:
        return (self.m // TILE) * (self.n // TILE)

    @property
    def row_w(self) -> int:
        """Leaves a tile-segment's words take."""
        return TILE_WORD_BYTES // self.row

    @property
    def per(self) -> int:
        """Words a leaf row holds."""
        return self.row // 2

    @property
    def lpt(self) -> int:
        """Leaves a tile takes."""
        return self.S * self.row_w

    @property
    def leaves(self) -> int:
        return self.tiles * self.lpt

    def seg(self, s: int) -> tuple[int, int]:
        """Segment s's inner range [a, b) (s = 1 .. S)."""
        if not 1 <= s <= self.S:
            raise ValueError(f"segment {s} is not one of 1 .. {self.S}")
        return (s - 1) * self.L, min(s * self.L, self.k)

    def label(self) -> dict:
        return {"m": self.m, "n": self.n, "k": self.k, "lp": self.lp, "L": self.L, "row": self.row, "leaves": self.leaves}

    def domain(self) -> CommitmentDomain:
        return CommitmentDomain(identity_digest_sha512(TAG, self.label()), -1, RangeIndexedDomain(0, self.leaves),
                                hash="sha512")

    def tile_of(self, i: int, j: int) -> tuple[int, int]:
        """(tile, cell index in it) of cell (i, j)."""
        return (i // TILE) * (self.n // TILE) + j // TILE, (i % TILE) * TILE + j % TILE

    def word_leaf(self, i: int, j: int, s: int) -> tuple[int, int]:
        """(leaf position, byte offset in its row) of T_s at cell (i, j)."""
        self.seg(s)
        t, c = self.tile_of(i, j)
        return t * self.lpt + (s - 1) * self.row_w + c // self.per, 2 * (c % self.per)

    def leaf_cells(self, pos: int) -> tuple[int, list[tuple[int, int]]]:
        """(s, the cells (i, j) in word order) of leaf `pos`."""
        if not 0 <= pos < self.leaves:
            raise ValueError(f"leaf {pos} is not one of the call's {self.leaves}")
        t, rem = divmod(pos, self.lpt)
        s, r = divmod(rem, self.row_w)
        tm, tn = divmod(t, self.n // TILE)
        cells = range(r * self.per, (r + 1) * self.per)
        return s + 1, [(tm * TILE + c // TILE, tn * TILE + c % TILE) for c in cells]


# -- the coins and the pad -------------------------------------------------------------------------------------------------------

Rows = list[list[int]]


def _digest(what: str, v: bytes) -> bytes:
    if len(v) != 64:
        raise ValueError(f"{what} is a 64-byte digest, not {len(v)} bytes")
    return bytes(v)


def call_seeds(salt: bytes, coin: bytes) -> dict[str, bytes]:
    """The seeds of the call whose live coin is `coin`, in the window whose salt is `salt`: SHA-512(`SEEDS_TAG` ‖ salt ‖
    coin ‖ x) for each stream x."""
    head = SEEDS_TAG + _digest("the window's salt", salt) + _digest("the call's coin", coin)
    return {x: hashlib.sha512(head + x.encode().ljust(8, b"\0")).digest() for x in STREAMS}


def prg(seed: bytes, tag: str, idx: int) -> bytes:
    """Block `idx` of stream `tag` of `seed`."""
    return hashlib.sha512(_digest("a seed", seed) + tag.encode().ljust(8, b"\0") + idx.to_bytes(8, "big")).digest()


def _blocks(seed: bytes, tag: str, idx: Iterable[int]) -> bytes:
    return b"".join(prg(seed, tag, b) for b in idx)


def pad_rows(seeds: Mapping[str, bytes], k: int, rows: Sequence[int]) -> Rows:
    """The pad's rows `rows` at contraction k (len(rows) × k int8 values): row i is the first k bytes of blocks
    i ceil(k / 64) .. of the pad stream."""
    kb = -(-k // 64)
    return [array("b", _blocks(seeds["pad"], "pad", range(i * kb, (i + 1) * kb))[:k]).tolist() for i in rows]


def _bits(seed: bytes, tag: str, rows: Sequence[int], k: int) -> Rows:
    kb = -(-k // 512)
    out = []
    for i in rows:
        raw = _blocks(seed, tag, range(i * kb, (i + 1) * kb))
        out.append([(raw[c >> 3] >> (c & 7)) & 1 for c in range(k)])
    return out


def _words32(seed: bytes, tag: str, first: int, count: int) -> tuple[int, ...]:
    """Little-endian u32 words [16 first, 16 first + count) of a stream (first in blocks)."""
    raw = _blocks(seed, tag, range(first, first + -(-count // 16)))
    return struct.unpack(f"<{count}I", raw[:4 * count])


def _noise(seed: bytes, tag: str, rows: Sequence[int], lp: int) -> Rows:
    return [[((w * (2 * NOISE + 1)) >> 32) - NOISE for w in _words32(seed, tag, r * (lp // 16), lp)] for r in rows]


def perms(seeds: Mapping[str, bytes], k: int, lp: int) -> Rows:
    """π_τ for each of the K / L′ noise tiles, (K / L′) × L′."""
    tiles = k // lp
    w = _words32(seeds["P"], "P", 0, tiles * lp)
    out = []
    for tau in range(tiles):
        p = list(range(lp))
        for c in range(lp - 1, 0, -1):
            x = (w[tau * lp + c] * (c + 1)) >> 32
            p[c], p[x] = p[x], p[c]
        out.append(p)
    return out


def noise_a(sh: Shape, seeds: Mapping[str, bytes], rows: Sequence[int]) -> Rows:
    """na = d_A + V[k mod L′] for A's rows `rows` (len(rows) × K, each in [-80, 81])."""
    V = _noise(seeds["V"], "V", rows, sh.lp)
    return [[d + v[c % sh.lp] for c, d in enumerate(ds)] for ds, v in zip(_bits(seeds["dA"], "dA", rows, sh.k), V)]


def noise_b(sh: Shape, seeds: Mapping[str, bytes], rows: Sequence[int], pi: Rows | None = None) -> Rows:
    """nb = d_B + U[π_τ(k mod L′)] for B's rows `rows` (len(rows) × K, each in [-80, 81])."""
    pi = perms(seeds, sh.k, sh.lp) if pi is None else pi
    U = _noise(seeds["U"], "U", rows, sh.lp)
    lp = sh.lp
    return [[d + u[pi[c // lp][c % lp]] for c, d in enumerate(ds)]
            for ds, u in zip(_bits(seeds["dB"], "dB", rows, sh.k), U)]


def wrap8(x):
    """x mod 2^8 as an int8 (an int, or anything with integer arithmetic)."""
    return ((x + 128) & 255) - 128


# -- the words -------------------------------------------------------------------------------------------------------------------

def _row(what: str, v: Sequence[int], k: int) -> list[int]:
    out = [int(x) for x in v]
    if len(out) != k:
        raise ValueError(f"{what} has K = {k} entries, not {len(out)}")
    return out


def _noised(a: Sequence[int], n: Sequence[int]) -> list[int]:
    return [wrap8(x + y) for x, y in zip(a, n, strict=True)]


def _operands(sh: Shape, seeds: Mapping[str, bytes], ii: Sequence[int], jj: Sequence[int], B: Mapping[int, Sequence[int]],
              A: Mapping[int, Sequence[int]] | None) -> tuple[Rows, Rows]:
    """X's rows ii and W′'s rows jj, B's (and A's, else the pad's) rows by index."""
    a = pad_rows(seeds, sh.k, ii) if A is None else [_row(f"A's row {i}", A[i], sh.k) for i in ii]
    b = [_row(f"B's row {j}", B[j], sh.k) for j in jj]
    return ([_noised(r, n) for r, n in zip(a, noise_a(sh, seeds, ii))],
            [_noised(r, n) for r, n in zip(b, noise_b(sh, seeds, jj))])


def _prefix(sh: Shape, x: Sequence[int], w: Sequence[int], upto: int) -> list[int]:
    """T_1 .. T_upto of one cell from its X row and W′ row."""
    out, acc = [], 0
    for s in range(1, upto + 1):
        a, b = sh.seg(s)
        acc += sum(map(mul, x[a:b], w[a:b]))
        out.append(acc & M16)
    return out


def words(sh: Shape, seeds: Mapping[str, bytes], B: Sequence[Sequence[int]],
          A: Sequence[Sequence[int]] | None = None) -> list[Rows]:
    """The call's words T (S × m × n, as T[s - 1][i][j]) on weights B (n × K int8), A the pad (or `A`, a served call's
    operand)."""
    if len(B) != sh.n:
        raise ValueError(f"B has n = {sh.n} rows, not {len(B)}")
    if A is not None and len(A) != sh.m:
        raise ValueError(f"A has m = {sh.m} rows, not {len(A)}")
    X, W = _operands(sh, seeds, range(sh.m), range(sh.n), B, A)
    cells = [[_prefix(sh, x, w, sh.S) for w in W] for x in X]
    return [[[c[s] for c in row] for row in cells] for s in range(sh.S)]


def cell_words(sh: Shape, seeds: Mapping[str, bytes], b_j: Sequence[int], i: int, j: int,
               a_i: Sequence[int] | None = None) -> list[int]:
    """T_1 .. T_S of cell (i, j) from B's row j (`b_j`) and the coins alone, A's row i the pad's (or `a_i`)."""
    X, W = _operands(sh, seeds, [i], [j], {j: b_j}, None if a_i is None else {i: a_i})
    return _prefix(sh, X[0], W[0], sh.S)


def words_buffer(sh: Shape, T: Sequence[Sequence[Sequence[int]]]) -> bytes:
    """Per tile, per segment, the tile's T_s in row-major cell order as little-endian u16."""
    pack = struct.Struct(f"<{TILE}H").pack
    out = bytearray()
    for tm in range(sh.m // TILE):
        for tn in range(sh.n // TILE):
            for s in range(sh.S):
                for r in range(tm * TILE, (tm + 1) * TILE):
                    out += pack(*(int(v) & M16 for v in T[s][r][tn * TILE:(tn + 1) * TILE]))
    return bytes(out)


def row_bytes(sh: Shape, buf: bytes, pos: int) -> bytes:
    """Leaf `pos`'s row from the call's `words_buffer`."""
    sh.leaf_cells(pos)
    return buf[pos * sh.row:(pos + 1) * sh.row]


def leaf_row(sh: Shape, seeds: Mapping[str, bytes], B: Sequence[Sequence[int]], pos: int,
             A: Sequence[Sequence[int]] | None = None) -> bytes:
    """Leaf `pos`'s row from only the cells it holds: their pad rows (or `A`'s), their rows of B (n × K) and the coins."""
    s, cells = sh.leaf_cells(pos)
    ii, jj = sorted({i for i, _ in cells}), sorted({j for _, j in cells})
    X, W = _operands(sh, seeds, ii, jj, {j: B[j] for j in jj}, None if A is None else {i: A[i] for i in ii})
    ri, rj = {i: x for x, i in enumerate(ii)}, {j: x for x, j in enumerate(jj)}
    end = sh.seg(s)[1]
    return struct.pack(f"<{len(cells)}H", *(sum(map(mul, X[ri[i]][:end], W[rj[j]][:end])) & M16 for i, j in cells))


# -- the leaves and the tree -----------------------------------------------------------------------------------------------------

def prf_salt(ks: bytes, pos: int) -> bytes:
    """commit_gpu's salt PRF (`benchmarks/pouw/commit_gpu/reference.py`): leaf `pos`'s hm96 salt under the prover's private
    key `ks`, three SHA-512 blocks."""
    return b"".join(hashlib.sha512(ks + pos.to_bytes(8, "big") + r.to_bytes(8, "big")).digest() for r in range(3))


def leaf_h(row: bytes, salt: bytes) -> bytes:
    """A row's hiding leaf digest: hm96-sha512 over its ``sha512/row/v2`` digest."""
    return INST.tree_leaf(KEY, INST.commit_string(KEY, sha512_row_v2_digest(row, 8 * len(row)), salt))


@dataclass
class Committed:
    """A call's words under its tree: each leaf's row, salt and digest, and the tree's levels."""

    sh: Shape
    buf: bytes
    salt: Callable[[int], bytes]
    hs: list[bytes] = field(default_factory=list)
    levels: list[list[bytes]] = field(default_factory=list)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, pos: int) -> tuple[bytes, ...]:
        return tuple(self.levels[d][(pos >> d) ^ 1] for d in range(len(self.levels) - 1))


def commit(sh: Shape, T: Sequence[Sequence[Sequence[int]]], salt: Callable[[int], bytes]) -> Committed:
    """The call's words T under one frame-v3-sha512 tree, leaf p's salt ``salt(p)``."""
    buf = words_buffer(sh, T)
    dom = sh.domain()
    cm = Committed(sh, buf, salt)
    cm.hs = [leaf_h(row_bytes(sh, buf, p), salt(p)) for p in range(sh.leaves)]
    cm.levels = [list(lv) for lv in tree_levels(dom, [row_leaf(dom, p, h, V2) for p, h in enumerate(cm.hs)])]
    return cm


def opens(sh: Shape, root: bytes, pos: int, row: bytes, salt: bytes, path: Sequence[bytes]) -> bool:
    """Whether leaf `pos`'s row and salt open `root` on `path`."""
    op = Opening(pos, leaf_h(row, salt), tuple(path))
    return verify_opening(sh.domain(), Commitment(root, sh.leaves), op, V2, VerificationLimits())


def word_at(sh: Shape, rows: Mapping[int, bytes], i: int, j: int, s: int) -> int:
    """T_s of cell (i, j) from its opened leaf rows (`rows` by position); T_0 = 0."""
    if s == 0:
        return 0
    p, o = sh.word_leaf(i, j, s)
    return int.from_bytes(rows[p][o:o + 2], "little")


# -- the Definition --------------------------------------------------------------------------------------------------------------

def step_ok(t0: int, acc: int, t1: int) -> int:
    """1 when t1 ≡ t0 + acc (mod 2^16)."""
    return int((t0 + acc - t1) & M16 == 0)


@primitive("PouwDitherStep", 1, [("t0", Value(16)), ("acc", Value(32)), ("t1", Value(16))], Value(1),
           conformance="protocol-defined")
def PouwDitherStepWord(t0: int, acc: int, t1: int) -> int:
    """1 when t1 ≡ t0 + acc (mod 2^16): T_s is T_{s-1} plus the segment's sum."""
    return step_ok(t0, acc, t1)


@primitive("PouwDitherFirst", 1, [("acc", Value(32)), ("t1", Value(16))], Value(1), conformance="protocol-defined")
def PouwDitherFirstWord(acc: int, t1: int) -> int:
    """1 when t1 ≡ acc (mod 2^16): T_1 is the first segment's sum."""
    return step_ok(0, acc, t1)


def _eq(C, x, y):
    return [and_reduce(C, [NOT(a ^ b) for a, b in zip(x, y, strict=True)])]


def _step_bits(C, t0, acc, t1):
    return _eq(C, add(C, list(t0), list(acc[:16]), width=16), t1)


def _first_bits(C, acc, t1):
    return _eq(C, acc[:16], t1)


PouwDitherStep = definition("PouwDitherStep", 2, _step_bits, (("t0", U16), ("acc", U32), ("t1", U16)), BIT,
                            word=PouwDitherStepWord, conformance="protocol-defined",
                            doc="t1 = t0 + acc mod 2^16 on bits: PouwDitherStep_v1")
PouwDitherFirst = definition("PouwDitherFirst", 2, _first_bits, (("acc", U32), ("t1", U16)), BIT,
                             word=PouwDitherFirstWord, conformance="protocol-defined",
                             doc="t1 = acc mod 2^16 on bits: PouwDitherFirst_v1")


def _block_signature(S: Any) -> tuple:
    if not isinstance(S.L, int) or S.L < 32 or S.L % 32:
        raise ValueError(f"PouwDitherBlock: L is a positive multiple of 32, not {S.L!r}")
    if not isinstance(S.FIRST, bool):
        raise ValueError(f"PouwDitherBlock: FIRST is a bool, not {S.FIRST!r}")
    ops = tuple((name, Array(S.L, I8)) for name in ("a", "na", "b", "nb"))
    return (*ops, *(() if S.FIRST else (("t0", U16),)), ("t1", U16)), BIT


def _block_body(B: Any, S: Any, a, na, b, nb, *t) -> Any:
    X = B.batch(SG.PouwI8Add, a, na, axes=(0, 0))
    W = B.batch(SG.PouwI8Add, b, nb, axes=(0, 0))
    acc = B.call(SG.PouwSegDot0, X[0:32], W[0:32])
    for c in range(32, S.L, 32):
        acc = B.call(SG.PouwSegDot, acc, X[c:c + 32], W[c:c + 32])
    return B.call(PouwDitherFirst, acc, t[0]) if S.FIRST else B.call(PouwDitherStep, t[0], acc, t[1])


PouwDitherBlock = CompositeDefinition(
    "PouwDitherBlock", 1, ("L", "FIRST"), _block_signature, _block_body, conformance="protocol-defined",
    doc="Dither's no-fold proof block (`circuit.dither`): one cell and one segment of L inner indices, the pad's row "
        "segment a, B's b and their noise na and nb (int8 each), T_{s-1} (t0; none at FIRST, where T_0 = 0) and T_s "
        "(t1), to 1 exactly when t1 = t0 + sum_k (a_k + na_k)(b_k + nb_k) mod 2^16, each operand wrapped to int8.")


# -- a proof block in the reference ----------------------------------------------------------------------------------------------

def _int8s(v: Sequence[int]) -> list[int]:
    return [int(x) & 0xFF for x in v]


def block_inputs(sh: Shape, seeds: Mapping[str, bytes], b_j: Sequence[int], i: int, j: int, s: int,
                 t0: int, t1: int, a_i: Sequence[int] | None = None) -> list[list[int]]:
    """Proof block (i, j, s)'s inputs to `PouwDitherBlock` at L = its segment's length and FIRST = (s = 1), as word
    lists: a (the pad's row segment, or `a_i`'s), na, b (`b_j`'s segment), nb, then t0 (but at s = 1) and t1."""
    lo, hi = sh.seg(s)
    a = pad_rows(seeds, sh.k, [i])[0] if a_i is None else _row(f"A's row {i}", a_i, sh.k)
    b = _row(f"B's row {j}", b_j, sh.k)
    ins = [_int8s(a[lo:hi]), _int8s(noise_a(sh, seeds, [i])[0][lo:hi]), _int8s(b[lo:hi]),
           _int8s(noise_b(sh, seeds, [j])[0][lo:hi])]
    return ins + ([] if s == 1 else [[t0 & M16]]) + [[t1 & M16]]


def block_bits(ins: Sequence[Sequence[int]]) -> list[list[int]]:
    """`block_inputs` on bits, as `PouwDitherBlock` reads them (each word least significant bit first)."""
    out = []
    for words_ in ins[:4]:
        out.append([(w >> e) & 1 for w in words_ for e in range(8)])
    out += [[(w >> e) & 1 for w in words_ for e in range(16)] for words_ in ins[4:]]
    return out


def check_block(ins: Sequence[Sequence[int]]) -> int:
    """`PouwDitherBlock`'s pass bit on `block_inputs`, in the clear."""
    a, na, b, nb = ins[:4]
    acc = sum(_sx8(x + y) * _sx8(u + v) for x, y, u, v in zip(a, na, b, nb, strict=True))
    return step_ok(ins[4][0] if len(ins) == 6 else 0, acc, ins[-1][0])


def _sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


__all__ = ["NOISE", "STREAMS", "Shape", "TAG", "TILE", "PouwDitherBlock", "PouwDitherFirst", "PouwDitherFirstWord",
           "PouwDitherStep", "PouwDitherStepWord", "block_bits", "block_inputs", "call_seeds", "cell_words",
           "check_block", "commit", "leaf_h", "leaf_row", "noise_a", "noise_b", "opens", "pad_rows", "perms", "prf_salt",
           "prg", "row_bytes", "step_ok", "word_at", "words", "words_buffer", "wrap8"]
