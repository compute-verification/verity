"""Dither's served unit: the Definitions a sampled proof block's proof passes, the layout of a call's
committed values, and the CPU reference a units kernel's values and hiding leaves are checked against on sampled cells.
The protocol is Dither v3.2 (`verity/core/protocols/pouw/dither/PROTOCOL.md`), as the v31 prover implements it
(`benchmarks/pouw/accel_v31/v31.py`); the guarantee it instantiates is `Pouw.Guarantees.Dither.WorkSoundNoFold`.

**A call** is O = (A + d_A)(B + d_B)ᵀ over the integers: A (m × K int8) is the call's activations, committed in the call's
operand root (`operand_tree`), and B (n × K int8) a published checkpoint's s8 weights, which registration pins by the
checkpoint's digest (`checkpoint_digest`: B is pinned). Its noised operands are
X = wrap8(A + d_A + V) and W′ = wrap8(B + d_B + U π_τ), τ = k div L′, and per cell (i, j) its values are

    T_s = sum over k < min(s L, K) of X_ik W′_jk  mod 2^16   (s = 1 .. S = ceil(K / L)),   O_ij mod 2^32,

with L = t L′ (by default L = 1,024 and L′ = 128, t = 8: `Shape`). It commits T_1 .. T_{S−1} and O:
T_S is implied by O, and at L ≥ K a cell commits O alone. Only the noised segments s < S are credited, K − L MACs a
cell: segment S is checked through O, the un-noised product of operands the prover chose, so its block checks O for
O's consumers and earns nothing, and neither does a call at L ≥ K. A filler call
may take its A from the per-call pad instead (`fill_seeds`, `pad_rows`); it commits no O, so it commits T_S (`Shape.o`).

**The coins** are each stream's 512-bit seed, expanded as v31 expands them:
``prg(seed, x, idx) = SHA-512(seed ‖ x ‖ be64(idx))``, x NUL-padded to 8 bytes. A served call's seeds come from the
challenger's raw coin opened after its operand roots (`seeds`): seed_x = SHA-512(`SEEDS_TAG` ‖ coin ‖ operand-root
receipt ‖ be64(call) ‖ x), the receipt being the digest of the certifier's signed receipt of the block's operand roots.

* ``dA``, ``dB``: d[i, k] is bit k mod 512 (least significant first) of block i ceil(K / 512) + k div 512.
* ``V``, ``U``: row r's entry c is ((w 161) >> 32) - 80 for w the little-endian u32 word r L′ + c of the stream.
* ``P``: π_τ is Fisher–Yates over L′ with words τ L′ + c: c from L′ - 1 down to 1, x = (w (c + 1)) >> 32, swap c and x.
* ``pad`` (a filler's only): seed_x = SHA-512(`FILL_SEEDS_TAG` ‖ window salt ‖ call coin ‖ x) for every stream, and
  A[i, k] is byte k of the pad stream's blocks i ceil(K / 64) .., as an int8. A fixed public pad lets a prover table A's
  subset sums once a period.

A proof block reads the noise per inner index k: ``na = d_A + V[k mod L′]`` and ``nb = d_B + U[π_τ(k mod L′)]``, each in
[-80, 81] (`noise_a`, `noise_b`), public inputs the verifier computes from the coins, so no circuit holds a permutation.

**The Definitions.** A proof block is one cell (i, j) and one segment s:

* s < S, `PouwDitherBlock_v1{L, FIRST}`: ``(a, na, b, nb: Array<L, i8>, [t0: u16,] t1: u16) -> ok``, 1 exactly when
  t1 ≡ t0 + sum_k (a_k + na_k)(b_k + nb_k) mod 2^16, each operand wrapped to int8 (FIRST: s = 1, T_0 = 0).
* s = S, `PouwDitherLast_v1{K, L}`: ``(a, da over the whole rows, [na over k < (S−1)L,] b, db, [nb,] [t0: u16,]
  o: u32) -> ok`` (na, nb and t0 only when S > 1). v3.2's Program takes T_S from O by the binding identity
  O = T_S + 256(w_X W′ᵀ + X w_Wᵀ + 256 w_X w_Wᵀ) − Z_X Uᵀ − V Y_Wᵀ, then checks T_S − T_{S−1} ≡ Σ_seg X W′ᵀ (mod 2^16)
  and O ≡ Σ (A + d_A)(B + d_B)ᵀ (mod 2^32). For any inputs, with V = na − da and U π = nb − db, the identity's
  correction term 256(…) − Z_X Uᵀ − V Y_Wᵀ is exactly Σ_k (a + da)_k (b + db)_k − Σ_k X_k W′_k, so the T_S it gives is
  O − Σ (a + da)(b + db) + Σ_{k<K} X W′ᵀ, and the segment check holds, given O's, exactly when t0 ≡ Σ_{k<(S−1)L} X W′ᵀ
  (mod 2^16). The block checks that and o ≡ Σ (a + da)(b + db) (mod 2^32): the same pass bit, without the last
  segment's noise (`check_last` evaluates v3.2's form in the clear).

Every row is private but the coins' (na, nb, da, db), and the only output is the pass bit. The X W′ᵀ sums run 32 entries
a step through `segment.PouwSegDot0_v2` and `PouwSegDot_v2`, the s8 tensor-core step on each card (`DEVICES`, `dot_steps`).

**The leaves** (`Shape`) are the value root's layout the kernels write:
v31's at 16 KiB rows, without segment S's rows. Output tiles of 128 × 128 cells (tile tm (n / 128) + tn), a tile's
cells row-major (c = 128 r + col); per tile, its tile-segments' 16,384 T_s for s = 1 .. S − 1 as little-endian u16
(32 KiB each) and then its O as little-endian u32 (64 KiB), cut into rows of `row` bytes, so a tile is `Shape.lpt`
consecutive leaves. Each row is an ``hm96-sha512/row/v2`` leaf (`leaf_h`) in one binary frame-v3-sha512 tree a call,
salted by the prover with ``prf_salt(Ks, off + pos)`` (`salts`, commit_gpu's PRF of its private Ks at the call's salt
offset); no coin feeds a salt. A block (i, j, s) opens T_s's row (s < S), T_{s−1}'s (s > 1), and at s = S O's.

The reference: `values` gives a call's T (S × m × n) and O; `cell_values` one cell's from its two rows alone;
`values_buffer`, `row_bytes` and `leaf_row` the leaf rows, the last from only the cells it holds; `tree` and `opens` the
commitments; and `block_inputs` a proof block's inputs to its Definition (`block_definition`), which `check_block` and
`check_last` evaluate in the clear.
"""
from __future__ import annotations

import hashlib
import struct
from array import array
from dataclasses import dataclass, field
from operator import mul
from typing import Any, Callable, Iterable, Mapping, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, ZERO, add, and_reduce, compress_columns
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind, primitive
from verity.ml.circuits.silicon.integer import (
    BLACKWELL_SM120_S8_M16N8K32_WRAP,
    HOPPER_SM90_S8_K32,
    IntegerMma,
    exact_k_limit,
)
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest

from . import segment as SG

#: a served call's values tree, and a filler's (no O)
TAG = "verity/pouw/dither/call/v1"
FILL_TAG = "verity/pouw/dither/fill/v1"
OPERAND_TAG = "verity/pouw/dither/operand/v1"
CHECKPOINT_TAG = b"verity/pouw/dither/checkpoint/v1\0"
SEEDS_TAG = b"verity/pouw/dither/seeds/v1\0"
FILL_SEEDS_TAG = b"verity/pouw/dither/fill-seeds/v1\0"
#: a call's coin streams; a filler's seeds add the pad's
STREAMS = ("dA", "dB", "U", "V", "P")
PAD = "pad"
TILE = 128
#: HYB noise's bound: U and V are uniform on ±NOISE
NOISE = 80
CELLS = TILE * TILE
#: the bytes of a tile-segment's words (u16) and of a tile's O (u32)
TILE_WORD_BYTES, TILE_O_BYTES = 2 * CELLS, 4 * CELLS
M16, M32 = (1 << 16) - 1, (1 << 32) - 1
#: the least contraction at which an s8 × s8 sum can leave s32: below it the wrapping and clamping steps agree
K_LIMIT = exact_k_limit("s8", "s8")
BIT = BL.BIT
I8, U16, U32 = Array(8, BIT), Array(16, BIT), Array(32, BIT)
INST = hm96.SHA512
KEY = INST.default_key


# -- the shape -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """A call: m rows of A against n rows of B at contraction k, HYB noise in noise tiles of lp (L′) inner indices,
    words every L, leaf rows of `row` bytes, and O committed (`o`; a filler's calls commit none)."""

    m: int
    n: int
    k: int
    lp: int = 128
    L: int = 1024
    row: int = 16384
    o: bool = True

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.m % TILE or self.n % TILE:
            raise ValueError(f"m and n are whole tiles of {TILE}, not {self.m} × {self.n}")
        if self.lp <= 0 or self.lp % 32 or self.k % self.lp:
            raise ValueError(f"L′ is a multiple of 32 dividing K, not L′ {self.lp} at K {self.k}")
        if self.k >= K_LIMIT:
            raise ValueError(f"K is below {K_LIMIT} (`K_LIMIT`), not {self.k}")
        if self.L <= 0 or self.L % self.lp:
            raise ValueError(f"L is t L′, a multiple of L′ {self.lp}, not {self.L}")
        if min(self.L, self.k) > 8 * self.lp:
            raise ValueError(f"t = min(L, K) / L′ is at most 8 (verity/core/protocols/pouw/dither/PROTOCOL.md, The segment "
                             f"and the noise tile), not {min(self.L, self.k)} / {self.lp} at L {self.L}, K {self.k}")
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
    def row_o(self) -> int:
        """Leaves a tile's O takes."""
        return TILE_O_BYTES // self.row if self.o else 0

    @property
    def W(self) -> int:
        """Segments whose words are committed: S − 1 when O is (T_S is implied by it), else S."""
        return self.S - 1 if self.o else self.S

    @property
    def lpt(self) -> int:
        """Leaves a tile takes."""
        return self.W * self.row_w + self.row_o

    @property
    def leaves(self) -> int:
        return self.tiles * self.lpt

    def seg(self, s: int) -> tuple[int, int]:
        """Segment s's inner range [a, b) (s = 1 .. S)."""
        if not 1 <= s <= self.S:
            raise ValueError(f"segment {s} is not one of 1 .. {self.S}")
        return (s - 1) * self.L, min(s * self.L, self.k)

    def label(self) -> dict:
        return {"m": self.m, "n": self.n, "k": self.k, "lp": self.lp, "t": self.t, "row": self.row, "leaves": self.leaves}

    def domain(self) -> CommitmentDomain:
        return CommitmentDomain(identity_digest_sha512(TAG if self.o else FILL_TAG, self.label()), -1,
                                RangeIndexedDomain(0, self.leaves), hash="sha512")

    def tile_of(self, i: int, j: int) -> tuple[int, int]:
        """(tile, cell index in it) of cell (i, j)."""
        return (i // TILE) * (self.n // TILE) + j // TILE, (i % TILE) * TILE + j % TILE

    def word_leaf(self, i: int, j: int, s: int) -> tuple[int, int]:
        """(leaf position, byte offset in its row) of T_s at cell (i, j), for a committed T_s (s ≤ `W`)."""
        self.seg(s)
        if s > self.W:
            raise ValueError(f"T_{s} is implied by O, not committed")
        t, c = self.tile_of(i, j)
        per = self.row // 2
        return t * self.lpt + (s - 1) * self.row_w + c // per, 2 * (c % per)

    def o_leaf(self, i: int, j: int) -> tuple[int, int]:
        """(leaf position, byte offset in its row) of O at cell (i, j)."""
        if not self.o:
            raise ValueError("this call commits no O")
        t, c = self.tile_of(i, j)
        per = self.row // 4
        return t * self.lpt + self.W * self.row_w + c // per, 4 * (c % per)

    def leaf_of(self, pos: int) -> tuple[int | None, list[tuple[int, int]]]:
        """(s, or None for an O row; the cells (i, j) in value order) of leaf `pos`."""
        if not 0 <= pos < self.leaves:
            raise ValueError(f"leaf {pos} is not one of the call's {self.leaves}")
        t, r = divmod(pos, self.lpt)
        tm, tn = divmod(t, self.n // TILE)
        if r < self.W * self.row_w:
            s, h = divmod(r, self.row_w)
            s, per = s + 1, self.row // 2
        else:
            s, h, per = None, r - self.W * self.row_w, self.row // 4
        return s, [(tm * TILE + c // TILE, tn * TILE + c % TILE) for c in range(h * per, (h + 1) * per)]

    def opened(self, i: int, j: int, s: int) -> list[int]:
        """The leaves proof block (i, j, s) opens: T_s's (s ≤ `W`), T_{s−1}'s (s > 1), and at s = S O's."""
        self.seg(s)
        pos = [self.word_leaf(i, j, s)[0]] if s <= self.W else []
        pos += [self.word_leaf(i, j, s - 1)[0]] if s > 1 else []
        return list(dict.fromkeys(pos + ([self.o_leaf(i, j)[0]] if s == self.S and self.o else [])))


# -- the coins and the pad -------------------------------------------------------------------------------------------------------

Rows = list[list[int]]


def _digest(what: str, v: bytes) -> bytes:
    if len(v) != 64:
        raise ValueError(f"{what} is a 64-byte digest, not {len(v)} bytes")
    return bytes(v)


def _x(x: str) -> bytes:
    return x.encode().ljust(8, b"\0")


def seeds(coin: bytes, receipt: bytes, call: int) -> dict[str, bytes]:
    """A served call's seeds: SHA-512(`SEEDS_TAG` ‖ coin ‖ receipt ‖ be64(call) ‖ x) for each stream x, from the
    challenger's raw coin `coin` opened after the operand roots, the digest `receipt` of the certifier's signed receipt
    of them, and the call's index `call` in its sampling block."""
    if not 0 <= call < 1 << 64:
        raise ValueError(f"a call's index is a u64, not {call}")
    head = SEEDS_TAG + _digest("the coin", coin) + _digest("the operand-root receipt", receipt) + call.to_bytes(8, "big")
    return {x: hashlib.sha512(head + _x(x)).digest() for x in STREAMS}


def fill_seeds(salt: bytes, coin: bytes) -> dict[str, bytes]:
    """A filler call's seeds, from its live coin `coin` after its window's salt `salt`: SHA-512(`FILL_SEEDS_TAG` ‖ salt ‖
    coin ‖ x) for each stream x and the pad."""
    head = FILL_SEEDS_TAG + _digest("the window's salt", salt) + _digest("the call's coin", coin)
    return {x: hashlib.sha512(head + _x(x)).digest() for x in (PAD, *STREAMS)}


def prg(seed: bytes, tag: str, idx: int) -> bytes:
    """Block `idx` of stream `tag` of `seed`."""
    return hashlib.sha512(_digest("a seed", seed) + _x(tag) + idx.to_bytes(8, "big")).digest()


def _blocks(seed: bytes, tag: str, idx: Iterable[int]) -> bytes:
    return b"".join(prg(seed, tag, b) for b in idx)


def pad_rows(seeds: Mapping[str, bytes], k: int, rows: Sequence[int]) -> Rows:
    """A filler's A: the pad's rows `rows` at contraction k (len(rows) × k int8 values), row i the first k bytes of
    blocks i ceil(k / 64) .. of the pad stream."""
    kb = -(-k // 64)
    return [array("b", _blocks(seeds[PAD], PAD, range(i * kb, (i + 1) * kb))[:k]).tolist() for i in rows]


def _bits(seed: bytes, tag: str, rows: Sequence[int], k: int) -> Rows:
    kb = -(-k // 512)
    out = []
    for i in rows:
        raw = _blocks(seed, tag, range(i * kb, (i + 1) * kb))
        out.append([(raw[c >> 3] >> (c & 7)) & 1 for c in range(k)])
    return out


def _words32(seed: bytes, tag: str, first: int, count: int) -> tuple[int, ...]:
    """`count` little-endian u32 words of a stream from block `first`."""
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


def dithers(sh: Shape, seeds: Mapping[str, bytes], side: str, rows: Sequence[int]) -> Rows:
    """d_A's (side "A") or d_B's ("B") rows `rows` (len(rows) × K bits)."""
    return _bits(seeds["d" + side], "d" + side, rows, sh.k)


def noise_a(sh: Shape, seeds: Mapping[str, bytes], rows: Sequence[int]) -> Rows:
    """na = d_A + V[k mod L′] for A's rows `rows` (len(rows) × K, each in [-80, 81])."""
    V = _noise(seeds["V"], "V", rows, sh.lp)
    return [[d + v[c % sh.lp] for c, d in enumerate(ds)] for ds, v in zip(dithers(sh, seeds, "A", rows), V)]


def noise_b(sh: Shape, seeds: Mapping[str, bytes], rows: Sequence[int], pi: Rows | None = None) -> Rows:
    """nb = d_B + U[π_τ(k mod L′)] for B's rows `rows` (len(rows) × K, each in [-80, 81])."""
    pi = perms(seeds, sh.k, sh.lp) if pi is None else pi
    U, lp = _noise(seeds["U"], "U", rows, sh.lp), sh.lp
    return [[d + u[pi[c // lp][c % lp]] for c, d in enumerate(ds)] for ds, u in zip(dithers(sh, seeds, "B", rows), U)]


def wrap8(x):
    """x mod 2^8 as an int8 (an int, or anything with integer arithmetic)."""
    return ((x + 128) & 255) - 128


# -- the values ------------------------------------------------------------------------------------------------------------------

def _row(what: str, v: Sequence[int], k: int) -> list[int]:
    out = [int(x) for x in v]
    if len(out) != k:
        raise ValueError(f"{what} has K = {k} entries, not {len(out)}")
    if any(not -128 <= x < 128 for x in out):
        raise ValueError(f"{what} is not int8")
    return out


@dataclass
class _Cells:
    """Rows ii of A and jj of B with their coins: X, W′ and A + d_A, B + d_B."""

    X: Rows
    W: Rows
    Ad: Rows
    Bd: Rows


def _cells(sh: Shape, seeds: Mapping[str, bytes], A: Mapping[int, Sequence[int]], B: Mapping[int, Sequence[int]],
           ii: Sequence[int], jj: Sequence[int]) -> _Cells:
    a = [_row(f"A's row {i}", A[i], sh.k) for i in ii]
    b = [_row(f"B's row {j}", B[j], sh.k) for j in jj]
    na, nb = noise_a(sh, seeds, ii), noise_b(sh, seeds, jj)
    da, db = dithers(sh, seeds, "A", ii), dithers(sh, seeds, "B", jj)
    return _Cells([[wrap8(x + y) for x, y in zip(r, n)] for r, n in zip(a, na)],
                  [[wrap8(x + y) for x, y in zip(r, n)] for r, n in zip(b, nb)],
                  [[x + y for x, y in zip(r, d)] for r, d in zip(a, da)],
                  [[x + y for x, y in zip(r, d)] for r, d in zip(b, db)])


def _words(sh: Shape, x: Sequence[int], w: Sequence[int], upto: int) -> list[int]:
    """T_1 .. T_upto of one cell from its X row and W′ row."""
    out, acc = [], 0
    for s in range(1, upto + 1):
        a, b = sh.seg(s)
        acc += sum(map(mul, x[a:b], w[a:b]))
        out.append(acc & M16)
    return out


def _out(ad: Sequence[int], bd: Sequence[int]) -> int:
    return sum(map(mul, ad, bd)) & M32


def values(sh: Shape, seeds: Mapping[str, bytes], A: Sequence[Sequence[int]],
           B: Sequence[Sequence[int]]) -> tuple[list[Rows], Rows | None]:
    """The call's T (S × m × n, as T[s - 1][i][j], T_S included though O implies it) and O (m × n, None when the call
    commits none) on A (m × K int8: the call's activations, or a filler's `pad_rows`) and B (n × K int8)."""
    if len(A) != sh.m or len(B) != sh.n:
        raise ValueError(f"A × B is m × n = {sh.m} × {sh.n} rows, not {len(A)} × {len(B)}")
    c = _cells(sh, seeds, A, B, range(sh.m), range(sh.n))
    cells = [[_words(sh, x, w, sh.S) for w in c.W] for x in c.X]
    T = [[[v[s] for v in row] for row in cells] for s in range(sh.S)]
    return T, ([[_out(ad, bd) for bd in c.Bd] for ad in c.Ad] if sh.o else None)


def cell_values(sh: Shape, seeds: Mapping[str, bytes], a_i: Sequence[int], b_j: Sequence[int], i: int,
                j: int) -> tuple[list[int], int | None]:
    """T_1 .. T_S and O of cell (i, j) from A's row i, B's row j and the coins alone."""
    c = _cells(sh, seeds, {i: a_i}, {j: b_j}, [i], [j])
    return _words(sh, c.X[0], c.W[0], sh.S), (_out(c.Ad[0], c.Bd[0]) if sh.o else None)


def values_buffer(sh: Shape, T: Sequence[Sequence[Sequence[int]]], O: Sequence[Sequence[int]] | None = None) -> bytes:
    """The call's leaf rows end to end: per tile, its tile-segments' committed T_s (u16, s ≤ `Shape.W`) and then its O
    (u32), cells row-major, little-endian."""
    if sh.o and O is None:
        raise ValueError("this call commits O: pass it")
    w16, w32 = struct.Struct(f"<{TILE}H").pack, struct.Struct(f"<{TILE}I").pack
    out = bytearray()
    for tm in range(sh.m // TILE):
        for tn in range(sh.n // TILE):
            cols = slice(tn * TILE, (tn + 1) * TILE)
            for s in range(sh.W):
                for r in range(tm * TILE, (tm + 1) * TILE):
                    out += w16(*(int(v) & M16 for v in T[s][r][cols]))
            if sh.o:
                for r in range(tm * TILE, (tm + 1) * TILE):
                    out += w32(*(int(v) & M32 for v in O[r][cols]))
    return bytes(out)


def row_bytes(sh: Shape, buf: bytes, pos: int) -> bytes:
    """Leaf `pos`'s row from the call's `values_buffer`."""
    sh.leaf_of(pos)
    return buf[pos * sh.row:(pos + 1) * sh.row]


def leaf_row(sh: Shape, seeds: Mapping[str, bytes], A: Sequence[Sequence[int]], B: Sequence[Sequence[int]],
             pos: int) -> bytes:
    """Leaf `pos`'s row from only the cells it holds: their rows of A and of B and the coins."""
    s, cells = sh.leaf_of(pos)
    ii, jj = sorted({i for i, _ in cells}), sorted({j for _, j in cells})
    c = _cells(sh, seeds, {i: A[i] for i in ii}, {j: B[j] for j in jj}, ii, jj)
    ri, rj = {i: x for x, i in enumerate(ii)}, {j: x for x, j in enumerate(jj)}
    if s is None:
        return struct.pack(f"<{len(cells)}I", *(_out(c.Ad[ri[i]], c.Bd[rj[j]]) for i, j in cells))
    end = sh.seg(s)[1]
    return struct.pack(f"<{len(cells)}H", *(sum(map(mul, c.X[ri[i]][:end], c.W[rj[j]][:end])) & M16 for i, j in cells))


# -- the leaves and the trees ----------------------------------------------------------------------------------------------------

def prf_salt(ks: bytes, pos: int) -> bytes:
    """commit_gpu's salt PRF (`benchmarks/pouw/commit_gpu/reference.py`): leaf `pos`'s hm96 salt under the prover's private
    key `ks`, three SHA-512 blocks."""
    return b"".join(hashlib.sha512(ks + pos.to_bytes(8, "big") + r.to_bytes(8, "big")).digest() for r in range(3))


def salts(ks: bytes, off: int = 0) -> Callable[[int], bytes]:
    """A call's salts: row pos's is ``prf_salt(ks, off + pos)``, at the call's salt offset `off`."""
    return lambda pos: prf_salt(ks, off + pos)


def leaf_h(row: bytes, salt: bytes) -> bytes:
    """A row's hiding leaf digest: hm96-sha512 over its ``sha512/row/v2`` digest."""
    return INST.tree_leaf(KEY, INST.commit_string(KEY, sha512_row_v2_digest(row, 8 * len(row)), salt))


@dataclass
class Tree:
    """Rows under one frame-v3-sha512 tree: each row's salt and leaf digest, and the tree's levels."""

    dom: CommitmentDomain
    rows: list[bytes]
    salt: Callable[[int], bytes]
    hs: list[bytes] = field(default_factory=list)
    levels: list[list[bytes]] = field(default_factory=list)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, pos: int) -> tuple[bytes, ...]:
        return tuple(self.levels[d][(pos >> d) ^ 1] for d in range(len(self.levels) - 1))

    def open(self, pos: int) -> tuple[bytes, bytes, tuple[bytes, ...]]:
        """(row, salt, path) of leaf `pos`."""
        return self.rows[pos], self.salt(pos), self.path(pos)


def tree(dom: CommitmentDomain, rows: Sequence[bytes], salt: Callable[[int], bytes]) -> Tree:
    """`rows` under one tree of `dom`, row p's salt ``salt(p)``."""
    t = Tree(dom, list(rows), salt)
    t.hs = [leaf_h(r, salt(p)) for p, r in enumerate(t.rows)]
    t.levels = [list(lv) for lv in tree_levels(dom, [row_leaf(dom, p, h, V2) for p, h in enumerate(t.hs)])]
    return t


def commit(sh: Shape, buf: bytes, salt: Callable[[int], bytes]) -> Tree:
    """The call's values (`values_buffer`) under its value root."""
    if len(buf) != sh.leaves * sh.row:
        raise ValueError(f"the call's values are {sh.leaves * sh.row} bytes, not {len(buf)}")
    return tree(sh.domain(), [buf[p * sh.row:(p + 1) * sh.row] for p in range(sh.leaves)], salt)


def opens(dom: CommitmentDomain, root: bytes, leaves: int, pos: int, row: bytes, salt: bytes, path: Sequence[bytes]) -> bool:
    """Whether leaf `pos`'s row and salt open `root` (a tree of `leaves` leaves under `dom`) on `path`."""
    return verify_opening(dom, Commitment(root, leaves), Opening(pos, leaf_h(row, salt), tuple(path)), V2,
                          VerificationLimits())


def operand_domain(side: str, rows: int, k: int) -> CommitmentDomain:
    """The domain of a call's A rows (side "A", its operand root) or of the weights' B rows ("B", the weight root)."""
    if side not in ("A", "B"):
        raise ValueError(f"an operand is A or B, not {side!r}")
    return CommitmentDomain(identity_digest_sha512(OPERAND_TAG, {"side": side, "rows": rows, "k": k}), -1,
                            RangeIndexedDomain(0, rows), hash="sha512")


def operand_tree(side: str, M: Sequence[Sequence[int]], salt: Callable[[int], bytes]) -> Tree:
    """An operand's rows (each its K int8 codes, one hm96 row) under its root: a call's A, or the pinned weights' B."""
    k = len(M[0])
    rows = [array("b", _row(f"{side}'s row {r}", v, k)).tobytes() for r, v in enumerate(M)]
    return tree(operand_domain(side, len(rows), k), rows, salt)


def operand_row(row: bytes) -> list[int]:
    """An opened operand row's int8 codes."""
    return array("b", row).tolist()


def checkpoint_digest(B: Sequence[Sequence[int]]) -> bytes:
    """The digest registration pins B by: SHA-512(`CHECKPOINT_TAG` ‖ be64(n) ‖ be64(K) ‖ B's int8 codes row-major),
    recomputed by the verifier from the published checkpoint."""
    k = len(B[0])
    h = hashlib.sha512(CHECKPOINT_TAG + len(B).to_bytes(8, "big") + k.to_bytes(8, "big"))
    for r, v in enumerate(B):
        h.update(array("b", _row(f"B's row {r}", v, k)).tobytes())
    return h.digest()


def word_at(sh: Shape, rows: Mapping[int, bytes], i: int, j: int, s: int) -> int:
    """T_s of cell (i, j) from its opened value rows (`rows` by position); T_0 = 0."""
    if s == 0:
        return 0
    p, o = sh.word_leaf(i, j, s)
    return int.from_bytes(rows[p][o:o + 2], "little")


def out_at(sh: Shape, rows: Mapping[int, bytes], i: int, j: int) -> int:
    """O of cell (i, j) from its opened O row."""
    p, o = sh.o_leaf(i, j)
    return int.from_bytes(rows[p][o:o + 4], "little")


# -- the devices -----------------------------------------------------------------------------------------------------------------

#: each card's s8 tensor-core step (`silicon.integer`, PINNED in `silicon.instructions`): sm_120's IMMA m16n8k32 without
#: .satfinite, which `PouwSegDot_v2` is bound to, and the H200's (``wgmma`` m64nNk32 and ``mma.sync`` m16n8k32),
#: the same step with .satfinite
DEVICES: dict[str, IntegerMma] = {"sm_120": BLACKWELL_SM120_S8_M16N8K32_WRAP, "sm_90": HOPPER_SM90_S8_K32}


def dot_steps(arch: str) -> tuple[Any, Any]:
    """The X W′ᵀ steps (`PouwSegDot0_v2`, `PouwSegDot_v2`) of the card `arch` (`DEVICES`). They are the card's step when its
    step is s8 × s8 at k = 32 and wraps, or clamps once at D as the H200's does: every Definition here refuses a sum of
    `K_LIMIT` products or more, below which no partial sum leaves s32 (`integer.exact_k_limit`), so the two agree word for
    word and one Program serves both cards. Refused (ValueError) for any other step."""
    if arch not in DEVICES:
        raise ValueError(f"Dither has no device record for {arch!r}: its records are {', '.join(DEVICES)}")
    got, ref = DEVICES[arch], SG.PouwSegDot.silicon
    if (got.dtype, got.b_dtype, got.k) != (ref.dtype, ref.b_dtype, ref.k):
        raise ValueError(f"{got.name} ({got.arch}) is not an s8 × s8 step at k = {ref.k}: PouwSegDot_v2 is not its step")
    return SG.PouwSegDot0, SG.PouwSegDot


# -- the Definitions -------------------------------------------------------------------------------------------------------------

def step_ok(t0: int, acc: int, t1: int) -> int:
    """1 when t1 ≡ t0 + acc (mod 2^16)."""
    return int((t0 + acc - t1) & M16 == 0)


def out_dot(acc: int, a: Sequence[int], da: Sequence[int], b: Sequence[int], db: Sequence[int]) -> int:
    """acc + sum_k (a_k + da_k)(b_k + db_k) mod 2^32, a and b int8 encodings and da, db bits."""
    return (acc + sum((_sx8(x) + (d & 1)) * (_sx8(y) + (e & 1)) for x, d, y, e in zip(a, da, b, db, strict=True))) & M32


@primitive("PouwDitherStep", 1, [("t0", Value(16)), ("acc", Value(32)), ("t1", Value(16))], Value(1),
           conformance="protocol-defined")
def PouwDitherStepWord(t0: int, acc: int, t1: int) -> int:
    """1 when t1 ≡ t0 + acc (mod 2^16): T_s is T_{s-1} plus the segment's sum."""
    return step_ok(t0, acc, t1)


@primitive("PouwDitherFirst", 1, [("acc", Value(32)), ("t1", Value(16))], Value(1), conformance="protocol-defined")
def PouwDitherFirstWord(acc: int, t1: int) -> int:
    """1 when t1 ≡ acc (mod 2^16): T_1 is the first segment's sum."""
    return step_ok(0, acc, t1)


def _dithers(d: int) -> list[int]:
    return [d >> k & 1 for k in range(32)]


@primitive("PouwDitherOutDot", 1, [("acc", Value(32)), ("a", Array(32, Value(8))), ("da", Value(32)),
                                   ("b", Array(32, Value(8))), ("db", Value(32))], Value(32),
           conformance="protocol-defined")
def PouwDitherOutDotWord(acc: int, *v: int) -> int:
    """O's sum 32 entries on: acc + sum_k (a_k + da_k)(b_k + db_k) mod 2^32, da_k and db_k bit k of da and db (the word view
    of `PouwDitherOutDot_v2`'s `Array<32, Value<1>>`, `boolean.word_type`)."""
    return out_dot(acc, v[0:32], _dithers(v[32]), v[33:65], _dithers(v[65]))


@primitive("PouwDitherOutDot0", 1, [("a", Array(32, Value(8))), ("da", Value(32)), ("b", Array(32, Value(8))),
                                    ("db", Value(32))], Value(32), conformance="protocol-defined")
def PouwDitherOutDot0Word(*v: int) -> int:
    """O's sum over its first 32 entries: sum_k (a_k + da_k)(b_k + db_k) mod 2^32, da_k and db_k bit k of da and db."""
    return out_dot(0, v[0:32], _dithers(v[32]), v[33:65], _dithers(v[65]))


@primitive("PouwDitherLastOk", 1, [("pre", Value(32)), ("t0", Value(16)), ("oacc", Value(32)), ("o", Value(32))],
           Value(1), conformance="protocol-defined")
def PouwDitherLastOkWord(pre: int, t0: int, oacc: int, o: int) -> int:
    """1 when t0 ≡ pre (mod 2^16) and o ≡ oacc (mod 2^32)."""
    return int(step_ok(0, pre, t0) and (o - oacc) & M32 == 0)


@primitive("PouwDitherOnlyOk", 1, [("oacc", Value(32)), ("o", Value(32))], Value(1), conformance="protocol-defined")
def PouwDitherOnlyOkWord(oacc: int, o: int) -> int:
    """1 when o ≡ oacc (mod 2^32): a call of one segment, whose T_1 O implies."""
    return int((o - oacc) & M32 == 0)


def _eqs(x, y):
    return [NOT(a ^ b) for a, b in zip(x, y, strict=True)]


def _step_bits(C, t0, acc, t1):
    return [and_reduce(C, _eqs(add(C, list(t0), list(acc[:16]), width=16), t1))]


def _first_bits(C, acc, t1):
    return [and_reduce(C, _eqs(acc[:16], t1))]


def _last_ok_bits(C, pre, t0, oacc, o):
    return [and_reduce(C, _eqs(pre[:16], t0) + _eqs(oacc, o))]


def _only_ok_bits(C, oacc, o):
    return [and_reduce(C, _eqs(oacc, o))]


def _out_dot0_bits(C, *v):
    return _out_dot_bits(C, [ZERO] * 32, *v)


def _out_dot_bits(C, acc, *v):
    """acc + sum_k (a_k + da_k)(b_k + db_k): each 9-bit two's-complement operand a + da (a sign-extended, da added), each
    product's Baugh-Wooley partial products (an AND each, the sign row's and column's inverted), acc's bits and the
    products' constant 32 (2^9 - 2^17) in one carry-save tree, its low 32 bits committed."""
    a, da, b, db = v[0:32], v[32], v[33:65], v[65]
    cols: list[list] = [[] for _ in range(32)]
    for x8, dx, y8, dy in zip(a, da, b, db, strict=True):
        x = add(C, list(x8) + [x8[7]], [dx] + [ZERO] * 8, width=9)
        y = add(C, list(y8) + [y8[7]], [dy] + [ZERO] * 8, width=9)
        for i in range(9):
            for j in range(9):
                p = C.AND(x[i], y[j])
                cols[i + j].append(NOT(p) if (i == 8) != (j == 8) else p)
    for i, bit in enumerate(acc):
        cols[i].append(bit)
    const = (32 * ((1 << 9) - (1 << 17))) & M32
    for i in range(32):
        if const >> i & 1:
            cols[i].append(ONE)
    return C.commit(compress_columns(C, cols)[:32])


PouwDitherStep = definition("PouwDitherStep", 2, _step_bits, (("t0", U16), ("acc", U32), ("t1", U16)), BIT,
                            word=PouwDitherStepWord, conformance="protocol-defined",
                            doc="t1 = t0 + acc mod 2^16 on bits: PouwDitherStep_v1")
PouwDitherFirst = definition("PouwDitherFirst", 2, _first_bits, (("acc", U32), ("t1", U16)), BIT,
                             word=PouwDitherFirstWord, conformance="protocol-defined",
                             doc="t1 = acc mod 2^16 on bits: PouwDitherFirst_v1")
PouwDitherOutDot = definition("PouwDitherOutDot", 2, _out_dot_bits,
                              (("acc", U32), ("a", Array(32, I8)), ("da", Array(32, BIT)), ("b", Array(32, I8)),
                               ("db", Array(32, BIT))), U32, word=PouwDitherOutDotWord, conformance="protocol-defined",
                              doc="acc + sum_k (a_k + da_k)(b_k + db_k) mod 2^32 on bits: PouwDitherOutDot_v1")
PouwDitherOutDot0 = definition("PouwDitherOutDot0", 2, _out_dot0_bits,
                               (("a", Array(32, I8)), ("da", Array(32, BIT)), ("b", Array(32, I8)), ("db", Array(32, BIT))),
                               U32, word=PouwDitherOutDot0Word, conformance="protocol-defined",
                               doc="sum_k (a_k + da_k)(b_k + db_k) mod 2^32 on bits: PouwDitherOutDot0_v1")
PouwDitherLastOk = definition("PouwDitherLastOk", 2, _last_ok_bits, (("pre", U32), ("t0", U16), ("oacc", U32), ("o", U32)),
                              BIT, word=PouwDitherLastOkWord, conformance="protocol-defined",
                              doc="t0 = pre mod 2^16 and o = oacc mod 2^32 on bits: PouwDitherLastOk_v1")
PouwDitherOnlyOk = definition("PouwDitherOnlyOk", 2, _only_ok_bits, (("oacc", U32), ("o", U32)), BIT,
                              word=PouwDitherOnlyOkWord, conformance="protocol-defined",
                              doc="o = oacc mod 2^32 on bits: PouwDitherOnlyOk_v1")


def _length(name: str, what: str, v: Any) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 32 or v % 32 or v >= K_LIMIT:
        raise ValueError(f"{name}: {what} is a positive multiple of 32 below {K_LIMIT}, not {v!r}")


def _xw(B: Any, a, na, b, nb, upto: int) -> Any:
    """Σ_{k<upto} X W′ᵀ, 32 entries a step."""
    dot0, dot = SG.PouwSegDot0, SG.PouwSegDot
    X = B.batch(SG.PouwI8Add, a[0:upto], na[0:upto], axes=(0, 0))
    W = B.batch(SG.PouwI8Add, b[0:upto], nb[0:upto], axes=(0, 0))
    acc = B.call(dot0, X[0:32], W[0:32])
    for c in range(32, upto, 32):
        acc = B.call(dot, acc, X[c:c + 32], W[c:c + 32])
    return acc


def _block_signature(S: Any) -> tuple:
    _length("PouwDitherBlock", "L", S.L)
    if not isinstance(S.FIRST, bool):
        raise ValueError(f"PouwDitherBlock: FIRST is a bool, not {S.FIRST!r}")
    ops = tuple((name, Array(S.L, I8)) for name in ("a", "na", "b", "nb"))
    return (*ops, *(() if S.FIRST else (("t0", U16),)), ("t1", U16)), BIT


def _block_body(B: Any, S: Any, a, na, b, nb, *t) -> Any:
    acc = _xw(B, a, na, b, nb, S.L)
    return B.call(PouwDitherFirst, acc, t[0]) if S.FIRST else B.call(PouwDitherStep, t[0], acc, t[1])


def _last_S(S: Any) -> int:
    return -(-S.K // S.L)


def _last_signature(S: Any) -> tuple:
    _length("PouwDitherLast", "K", S.K)
    _length("PouwDitherLast", "L", S.L)
    row, lo = Array(S.K, I8), (_last_S(S) - 1) * S.L
    if not lo:
        return (("a", row), ("da", Array(S.K, BIT)), ("b", row), ("db", Array(S.K, BIT)), ("o", U32)), BIT
    pre = Array(lo, I8)
    return (("a", row), ("da", Array(S.K, BIT)), ("na", pre), ("b", row), ("db", Array(S.K, BIT)), ("nb", pre),
            ("t0", U16), ("o", U32)), BIT


def _last_body(B: Any, S: Any, *v) -> Any:
    lo = (_last_S(S) - 1) * S.L
    a, da, na, b, db, nb, t0, o = v if lo else (v[0], v[1], None, v[2], v[3], None, None, v[4])
    oacc = B.call(PouwDitherOutDot0, a[0:32], da[0:32], b[0:32], db[0:32])
    for c in range(32, S.K, 32):
        oacc = B.call(PouwDitherOutDot, oacc, a[c:c + 32], da[c:c + 32], b[c:c + 32], db[c:c + 32])
    if not lo:
        return B.call(PouwDitherOnlyOk, oacc, o)
    pre = _xw(B, a, na, b, nb, lo)
    return B.call(PouwDitherLastOk, pre, t0, oacc, o)


PouwDitherBlock = CompositeDefinition(
    "PouwDitherBlock", 1, ("L", "FIRST"), _block_signature, _block_body, conformance="protocol-defined",
    doc="Dither's proof block of a segment s < S (`circuit.dither`): one cell and one segment of L inner indices, A's and "
        "B's row segments a and b and their noise na and nb (int8 each), T_{s-1} (t0; none at FIRST, where T_0 = 0) and "
        "T_s (t1), to 1 exactly when t1 = t0 + sum_k (a_k + na_k)(b_k + nb_k) mod 2^16, each operand wrapped to int8.")

PouwDitherLast = CompositeDefinition(
    "PouwDitherLast", 1, ("K", "L"), _last_signature, _last_body, conformance="protocol-defined",
    doc="Dither's proof block of a cell's last segment S = ceil(K / L) (`circuit.dither`): A's and B's whole rows a and b "
        "and their dithers da and db, their noise na and nb over k < (S-1)L, T_{S-1} (t0) and O (o) (no na, nb or t0 when "
        "S = 1), to 1 exactly when t0 = sum_{k<(S-1)L} X_k W'_k mod 2^16 (X = a + na, W' = b + nb wrapped to int8) and "
        "o = sum_k (a_k + da_k)(b_k + db_k) mod 2^32: v3.2's segment check on the T_S its binding identity takes from O, "
        "and O's check.")


def block_definition(sh: Shape, s: int) -> Any:
    """Proof block (·, ·, s)'s Definition: `PouwDitherBlock_v1` at its segment for s < S, `PouwDitherLast_v1` at s = S."""
    lo, hi = sh.seg(s)
    if s == sh.S and sh.o:
        return bind(PouwDitherLast, K=sh.k, L=sh.L)
    return bind(PouwDitherBlock, L=hi - lo, FIRST=s == 1)


# -- a proof block in the reference ----------------------------------------------------------------------------------------------

def _sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def _int8s(v: Sequence[int]) -> list[int]:
    return [int(x) & 0xFF for x in v]


def block_inputs(sh: Shape, seeds: Mapping[str, bytes], a_i: Sequence[int], b_j: Sequence[int], i: int, j: int, s: int,
                 t0: int = 0, t1: int | None = None, o: int | None = None) -> list[list[int]]:
    """Proof block (i, j, s)'s inputs to `block_definition`, as word lists. For s < S (and a call without O): a and b (A's
    row i and B's row j over the segment), na, nb, then t0 (but at s = 1) and t1. For s = S: a, da over the whole row,
    na over k < (S − 1)L, then b, db, nb likewise, t0, and o; when S = 1 only a, da, b, db and o."""
    a, b = _row(f"A's row {i}", a_i, sh.k), _row(f"B's row {j}", b_j, sh.k)
    na, nb = noise_a(sh, seeds, [i])[0], noise_b(sh, seeds, [j])[0]
    if s == sh.S and sh.o:
        if o is None:
            raise ValueError("the last segment's block reads O: pass it")
        da, db = dithers(sh, seeds, "A", [i])[0], dithers(sh, seeds, "B", [j])[0]
        lo = sh.seg(s)[0]
        if not lo:
            return [_int8s(a), da, _int8s(b), db, [o & M32]]
        return [_int8s(a), da, _int8s(na[:lo]), _int8s(b), db, _int8s(nb[:lo]), [t0 & M16], [o & M32]]
    if t1 is None:
        raise ValueError(f"segment {s}'s block reads T_{s}: pass it")
    lo, hi = sh.seg(s)
    ins = [_int8s(a[lo:hi]), _int8s(na[lo:hi]), _int8s(b[lo:hi]), _int8s(nb[lo:hi])]
    return ins + ([] if s == 1 else [[t0 & M16]]) + [[t1 & M16]]


def block_widths(ins: Sequence[Sequence[int]], last: bool) -> list[int]:
    """Each port's word width of `block_inputs` (`last`: a last segment's)."""
    if last:
        return [8, 1, 8, 1, 32] if len(ins) == 5 else [8, 1, 8, 8, 1, 8, 16, 32]
    return [8, 8, 8, 8] + [16] * (len(ins) - 4)


def block_bits(ins: Sequence[Sequence[int]], last: bool = False) -> list[list[int]]:
    """`block_inputs` on bits, as the Definition reads them (each word least significant bit first)."""
    return [[(w >> e) & 1 for w in words_ for e in range(n)] for words_, n in zip(ins, block_widths(ins, last), strict=True)]


def check_block(ins: Sequence[Sequence[int]]) -> int:
    """`PouwDitherBlock`'s pass bit on `block_inputs`, in the clear."""
    a, na, b, nb = ins[:4]
    acc = sum(_sx8(x + y) * _sx8(u + v) for x, y, u, v in zip(a, na, b, nb, strict=True))
    return step_ok(ins[4][0] if len(ins) == 6 else 0, acc, ins[-1][0])


def check_last(sh: Shape, seeds: Mapping[str, bytes], i: int, j: int, ins: Sequence[Sequence[int]]) -> int:
    """v3.2's checks of cell (i, j)'s last-segment block on `block_inputs`, in the clear, as its Program states them:
    T_S taken from O by the binding identity O = T_S + 256(w_X W′ᵀ + X w_Wᵀ + 256 w_X w_Wᵀ) − Z_X Uᵀ − V Y_Wᵀ over the
    whole rows, with V = na − da and U π = nb − db (so Z_X Uᵀ = Σ (A + d_A + V) U π and V Y_Wᵀ = Σ V (B + d_B)), then
    T_S − T_{S−1} against the segment's X W′ᵀ (mod 2^16) and O against (A + d_A)(B + d_B)ᵀ (mod 2^32). The noise past
    (S − 1)L, which the block doesn't read, comes from the coins."""
    lo = sh.seg(sh.S)[0]
    if lo:
        a8, da, na8, b8, db, nb8, (t0,), (o,) = ins
    else:
        (a8, da, b8, db, (o,)), na8, nb8, t0 = ins, [], [], 0
    a, b = [_sx8(x) for x in a8], [_sx8(x) for x in b8]
    na = [_sx8(x) for x in na8] + noise_a(sh, seeds, [i])[0][lo:]
    nb = [_sx8(x) for x in nb8] + noise_b(sh, seeds, [j])[0][lo:]
    x, y = [p + q for p, q in zip(a, na)], [p + q for p, q in zip(b, nb)]
    X, W = [wrap8(v) for v in x], [wrap8(v) for v in y]
    wX, wW = [(p - q) // 256 for p, q in zip(x, X)], [(p - q) // 256 for p, q in zip(y, W)]
    Ad, Bd = [p + q for p, q in zip(a, da)], [p + q for p, q in zip(b, db)]
    V, Up = [p - q for p, q in zip(na, da)], [p - q for p, q in zip(nb, db)]
    seg = sum(map(mul, X[lo:], W[lo:]))
    wrap = sum(map(mul, wX, W)) + sum(map(mul, X, wW)) + 256 * sum(map(mul, wX, wW))
    zu = sum((p + q) * u for p, q, u in zip(Ad, V, Up))
    vy = sum(map(mul, V, Bd))
    tS = o - 256 * wrap + zu + vy
    return int((tS - t0 - seg) & M16 == 0 and (o - sum(map(mul, Ad, Bd))) & M32 == 0)


__all__ = ["DEVICES", "K_LIMIT", "NOISE", "PAD", "STREAMS", "Shape", "TAG", "TILE", "Tree", "PouwDitherBlock",
           "PouwDitherFirst", "PouwDitherLast", "PouwDitherLastOk", "PouwDitherOnlyOk", "PouwDitherOutDot",
           "PouwDitherOutDot0", "PouwDitherStep", "block_bits", "block_definition", "block_inputs", "block_widths",
           "cell_values", "check_block", "check_last", "checkpoint_digest", "commit", "dithers", "dot_steps", "fill_seeds",
           "leaf_h", "leaf_row", "noise_a", "noise_b", "operand_domain", "operand_row", "operand_tree", "opens", "out_at",
           "out_dot", "pad_rows", "perms", "prf_salt", "prg", "row_bytes", "salts", "seeds", "step_ok", "tree", "values",
           "values_buffer", "word_at", "wrap8"]
