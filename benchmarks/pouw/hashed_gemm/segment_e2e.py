"""The compute protocol's checkpointed receipts and their spot check (`benchmarks/e2e_slice`), as word primitives and Boolean
Definitions.

The edited GEMM serves Y = A B'^T from P = A' B'^T (A' = A + U W^T, U in {±1..±64} with period 32 along k; B' = B + F, F ±1,
or under decision 29's dither B' = B + d, d in {0, 1}) on 128 x 128 tiles of 512 threads (fgemm.cu's mainloop at warps
32 x 32). Thread ``ctid`` (warp w = ctid / 32, lane l, wm = w / 4, wn = w % 4, g = l >> 2, q = l & 3) holds 32 accumulator
words: entry 4 t + e (t = 4 mi + ni, mi < 2, ni < 4, e < 4) at row wm 32 + g + 8 r and column wn 32 + 2 q + 8 ni + (e & 1),
with r = 2 mi + (e >> 1) its row slot and c = 2 ni + (e & 1) its column slot. After each 32-step segment j (k = 32 j ..
32 j + 31) its 32 prefix words are T_j (T_(-1) = 0).

Receipts are checkpointed every S segments. Window w covers segments w S .. w S + S - 1; its fold starts from keys the coin
gives the (tile, thread, window), absorbs the S + 1 prefix-word sets T_(w S - 1), T_(w S), ..., T_(w S + S - 1) (the
window's incoming words, then each of its segments'), and commits one word, R_w. A step-j check therefore recomputes j's
window from its keys (the last checkpoint) to its committed word, and both T_(j-1) and T_j are among the words it absorbs.
A call's receipts are KS / S words a thread (KS = k / 32), 4 KS / S bytes, laid out (tile, window, thread) and committed
RW = 512 words a leaf, so a leaf is one window of one tile.

The fold is a parameter (``FOLD``, `FOLDS`):

- ``duo`` (fgemm.cu's duo, the default): state y, z (8 chains each), y = mix32(seed ^ mix32(((tile 512 + ctid) NW + w) 8 + i)),
  z = mix32(y + 0x9E3779B9) (NW = KS / S windows). Chain 4 (t % 2) + e absorbs entry 4 t + e at position t:
  u = y ^ p ^ kx_t, y = u a_t + z, z = rotl(u, r_t) + y (mod 2^32, a_t odd, fgemm.cu's lane-0 keys); R = xor_i (y_i ^
  rotl(z_i, 16)).
- ``xin`` (P2-FF r9's XOR-in with rotations (13, 19), note:20261016T1117Z-draft-fullfloor-s-r9; provisional): state y (8
  chains, duo's y keys), chain t absorbing entries 4 t .. 4 t + 3 as d0 .. d3: y = (rotl(y, 13) ^ d0) a0_t + d1, then
  y = (rotl(y, 19) ^ d2) a1_t + d3, the multipliers odd and outside Lambda8 (`key_ok`); R = xor_t y_t.

Operands are committed as decision 46's tile leaves: R = 64 rows x 32 values of an operand (row block i / 64, segment j).
The thread's 4 rows of A at segment j are rows o + 8 r of A's leaf, o = 32 (wm % 2) + g (16 offsets); its 8 rows of B are
rows o' + 8 (c >> 1) + (c & 1) of B's leaf, o' = 32 (wn % 2) + 2 q (8 offsets).

    PouwI8Add_v1     (a, u: Value<8>) -> Value<8>                  a + u mod 2^8 (A' from A and U)
    PouwI8Pm1_v1     (b: Value<8>, f: Value<1>) -> Value<8>        b + 1 - 2 f mod 2^8 (B' from B and F's sign bit)
    PouwI8AddBit_v1  (b: Value<8>, d: Value<1>) -> Value<8>        b + d mod 2^8 (B' from B and decision 29's dither bit)
    PouwSegDot_v1    (acc: Value<32>, a, b: Array<32, Value<8>>) -> Value<32>      acc + sum_k a_k b_k mod 2^32, int8 a and b
    PouwSel8_v1      (s: Value<1>, w: Value<8>) -> Value<8>        w if s else 0
    PouwXor8_v1      (x, y: Value<8>) -> Value<8>                  x ^ y
    PouwSel32_v1     (s: Value<1>, w: Value<32>) -> Value<32>      w if s else 0
    PouwDuoAbsorb_v1 (st: Value<512>, p: Array<32, Value<32>>) -> Value<512>      duo's state (y_0..7, z_0..7) after p
    PouwDuoFinal_v1  (st: Value<512>) -> Value<32>                 duo's word of a state
    PouwXinAbsorb_v1 (st: Value<256>, p: Array<32, Value<32>>) -> Value<256>      XOR-in's state (y_0..7) after p
    PouwXinFinal_v1  (st: Value<256>) -> Value<32>                 XOR-in's word of a state
    PouwWinOk_v1     (r, w: Value<32>, tj, tw, p0: Array<32, Value<32>>, first: Value<1>) -> Value<1>
                     r = w, tj = tw, and (first ? p0 = 0 : 1)

Each has a Boolean twin (``_v2``, traced from a `forms` builder, word view its ``_v1``). The spot check of one thread at
segment j is

    PouwSegCheck_v2{S, FOLD}  (a: Array<64, Array<32, i8>>, asel: Array<16, bit>, u: Array<4, Array<32, i8>>,
                               b: Array<64, Array<32, i8>>, bsel: Array<8, bit>, f: Array<8, Array<32, bit>>,
                               p: Array<S + 1, Array<32, u32>>, st: Array<FW, u32>, rc: Array<512, u32>,
                               rsel: Array<16, u32>, jsel: Array<JW, bit>, first: bit) -> bit       (bits, LSB first)

over A's and B's tile leaves at segment j with the one-hot offsets of the thread's rows in them, U's rows and F's (or the
dither's) bits for those rows, the window's S + 1 absorbed prefix-word sets (witness), the window's fold keys (FW words:
16 for duo, 8 for xin), the receipt leaf holding the thread's window word and its one-hot position (512 bits as 16 words),
the one-hot position of j in its window (its first S bits; JW = `jsel_bits`(S), the rest zero), and whether the window is
the first. It computes T_j = T_(j-1) + A'_j B'_j^T entry by entry from p's set at j's position, folds the window, and returns 1 exactly when the fold is the leaf's word, T_j is p's next
set, and in the first window the incoming set is 0. A statement holds ``a``, ``b`` and ``rc`` to the trees committed before
and after the coin, recomputes ``u``, ``f``, ``st`` and the selections from the coin and the sample, and reads the result
through a row of known content: so a proof says the thread's committed window word is that of the committed operands'
noised product, segment j included.

    PouwSegCheckDither_v2{S, FOLD}   the same with decision 29's dither on the served side: B' = B + d (`PouwI8AddBit_v2`),
                                     d in {0, 1} the coin bit ``f`` carries (the served X = floor(x / s) + b)
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import (
    NOT,
    ONE,
    ZERO,
    add,
    and_reduce,
    compress_columns,
    const_bits,
    or_reduce,
)
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import composite, primitive
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Value

from .boolean import PouwXor32

__all__ = [
    "FOLDS",
    "KA",
    "KK",
    "MI",
    "NCH",
    "NCT",
    "NI",
    "RL",
    "ROT",
    "RW",
    "XA0",
    "XA1",
    "XROT",
    "PouwDuoAbsorb",
    "PouwDuoAbsorbWord",
    "PouwDuoFinal",
    "PouwDuoFinalWord",
    "PouwI8Add",
    "PouwI8AddBit",
    "PouwI8AddBitWord",
    "PouwI8AddWord",
    "PouwI8Pm1",
    "PouwI8Pm1Word",
    "PouwSegCheck",
    "PouwSegCheckDither",
    "PouwSegDot",
    "PouwSegDotWord",
    "PouwSel8",
    "PouwSel8Word",
    "PouwSel32",
    "PouwSel32Word",
    "PouwWinOk",
    "PouwWinOkWord",
    "PouwXinAbsorb",
    "PouwXinAbsorbWord",
    "PouwXinFinal",
    "PouwXinFinalWord",
    "PouwXor8",
    "PouwXor8Word",
    "a_offset",
    "b_offset",
    "col_slot",
    "fold_absorb",
    "fold_final",
    "fold_keys",
    "init",
    "jsel_bits",
    "key_ok",
    "mix32",
    "row_slot",
    "seg_dot",
    "win_check",
]

BIT = BL.BIT
U32 = Array(32, BIT)
I8 = Array(8, BIT)
V1, V8, V32 = Value(1), Value(8), Value(32)
M32 = (1 << 32) - 1
#: threads a tile, warp tiles, chains a thread; operand rows a tile leaf; receipt words a leaf
NCT, MI, NI, NCH = 512, 2, 4, 8
RL, RW = 64, 512
GOLDEN = 0x9E3779B9


def mix32(x: int) -> int:
    """fgemm.cu's mix32 on one word."""
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & M32


#: duo's multiplier, xor key and rotation at position t (fgemm.cu's key_mul(0, t), key_xor(0, t), rot_of(DUO, 0, t))
KA = tuple(mix32(0xA0000000 + 2 * t) | 1 for t in range(MI * NI))
KK = tuple(mix32(0xB0000000 + 2 * t + 1) for t in range(MI * NI))
ROT = tuple(((t * 7 + 5) % 31) + 1 for t in range(MI * NI))


def key_ok(a: int) -> bool:
    """XOR-in's multiplier condition (the Lambda8 half of P2-FF r9's key_ok_r9): a odd, and neither +a nor -a mod 2^30 in
    Lambda8 = {lambda : lambda b = b' mod 2^30 for some int8 b != 0 and int8 b'}."""
    if not a & 1:
        return False
    for lam in (a, -a):
        for b in range(-128, 128):
            if b and (((lam * b + (1 << 29)) % (1 << 30)) - (1 << 29)) in range(-128, 128):
                return False
    return True


def _xin_keys() -> tuple[tuple[int, ...], tuple[int, ...]]:
    out: list[int] = []
    i = 0
    while len(out) < 2 * MI * NI:
        a = mix32(0xC0000000 + i) | 1
        i += 1
        if key_ok(a):
            out.append(a)
    return tuple(out[0::2]), tuple(out[1::2])


#: XOR-in's multipliers per chain and its rotations
XA0, XA1 = _xin_keys()
XROT = (13, 19)


def row_slot(q: int) -> int:
    """Entry q's row slot r (row wm 32 + g + 8 r)."""
    t, e = divmod(q, 4)
    return 2 * (t // NI) + (e >> 1)


def col_slot(q: int) -> int:
    """Entry q's column slot c (column wn 32 + 2 q' + 8 (c >> 1) + (c & 1))."""
    t, e = divmod(q, 4)
    return 2 * (t % NI) + (e & 1)


def a_offset(i: int) -> int:
    """A's leaf row of row slot 0 at offset index i (the thread's 32 (wm % 2) + g)."""
    return 32 * (i >> 3) + (i & 7)


def b_offset(i: int) -> int:
    """B's leaf row of column slot 0 at offset index i (the thread's 32 (wn % 2) + 2 q)."""
    return 32 * (i >> 2) + 2 * (i & 3)


def init(seed: int, tile: int, ctid: int, w: int, nw: int) -> tuple[list[int], list[int]]:
    """duo's chain keys (y, z) of thread ``ctid`` of ``tile`` at window ``w`` of ``nw``."""
    y = [mix32((seed ^ mix32(((tile * NCT + ctid) * nw + w) * NCH + j)) & M32) for j in range(NCH)]
    return y, [mix32(v + GOLDEN) for v in y]


def fold_keys(fold: str, seed: int, tile: int, ctid: int, w: int, nw: int) -> list[int]:
    """The fold's state at the window's start, as words (duo: y then z; xin: y)."""
    y, z = init(seed, tile, ctid, w, nw)
    return y + z if fold == "duo" else y


def _sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def seg_dot(acc: int, a, b) -> int:
    return (acc + sum(_sx8(x) * _sx8(w) for x, w in zip(a, b, strict=True))) & M32


def _duo_absorb(st: list[int], p) -> list[int]:
    y, z = [v & M32 for v in st[:NCH]], [v & M32 for v in st[NCH:]]
    for t in range(MI * NI):
        h = 4 * (t % 2)
        for e in range(4):
            u = y[h + e] ^ (p[4 * t + e] & M32) ^ KK[t]
            y[h + e] = (u * KA[t] + z[h + e]) & M32
            z[h + e] = (rotl(u, ROT[t]) + y[h + e]) & M32
    return y + z


def _xin_absorb(st: list[int], p) -> list[int]:
    y = [v & M32 for v in st]
    for t in range(MI * NI):
        d = [v & M32 for v in p[4 * t:4 * t + 4]]
        y[t] = ((rotl(y[t], XROT[0]) ^ d[0]) * XA0[t] + d[1]) & M32
        y[t] = ((rotl(y[t], XROT[1]) ^ d[2]) * XA1[t] + d[3]) & M32
    return y


def fold_absorb(fold: str, st: list[int], p) -> list[int]:
    return _duo_absorb(st, p) if fold == "duo" else _xin_absorb(st, p)


def fold_final(fold: str, st: list[int]) -> int:
    out = 0
    if fold == "duo":
        for j in range(NCH):
            out ^= st[j] ^ rotl(st[NCH + j], 16)
    else:
        for v in st:
            out ^= v
    return out


def win_check(a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first, fold: str = "duo", dither: bool = False) -> int:
    """`PouwSegCheck_v2` (``dither``: `PouwSegCheckDither_v2`) on words: a and b (64 x 32) int8 encodings, asel (16) and bsel
    (8) one-hot, u (4 x 32), f (8 x 32) bits, p (S + 1 sets of 32), st (the fold's state words), rc and rsel (512 each),
    jsel (S), first."""
    oa = [a_offset(i) for i, s in enumerate(asel) if s & 1]
    ob = [b_offset(i) for i, s in enumerate(bsel) if s & 1]
    ar = [[0] * 32 for _ in range(4)]
    br = [[0] * 32 for _ in range(8)]
    for o in oa:
        ar = [[x ^ y for x, y in zip(ar[r], a[o + 8 * r])] for r in range(4)]
    for o in ob:
        br = [[x ^ y for x, y in zip(br[c], b[o + 8 * (c >> 1) + (c & 1)])] for c in range(8)]
    ap = [[(x + v) & 0xFF for x, v in zip(ar[r], u[r])] for r in range(4)]
    bp = [[(x + ((s & 1) if dither else 1 - 2 * (s & 1))) & 0xFF for x, s in zip(br[c], f[c])] for c in range(8)]
    prev, tw = [0] * 32, [0] * 32
    for i, s in enumerate(jsel):
        if s & 1:
            prev = [x ^ y for x, y in zip(prev, p[i])]
            tw = [x ^ y for x, y in zip(tw, p[i + 1])]
    tj = [seg_dot(prev[q], ap[row_slot(q)], bp[col_slot(q)]) for q in range(32)]
    state = list(st)
    for ps in p:
        state = fold_absorb(fold, state, ps)
    w = 0
    for i, s in enumerate(rsel):
        if s & 1:
            w ^= rc[i]
    return int(fold_final(fold, state) == w & M32 and tj == [v & M32 for v in tw]
               and (not first & 1 or all(v & M32 == 0 for v in p[0])))


def _words(v: int, n: int) -> list[int]:
    return [(v >> (32 * i)) & M32 for i in range(n)]


def _pack(ws: list[int]) -> int:
    return sum((w & M32) << (32 * i) for i, w in enumerate(ws))


# --- word primitives ----------------------------------------------------------------------------------------------------------

@primitive("PouwI8Add", 1, [("a", V8), ("u", V8)], V8, conformance="protocol-defined")
def PouwI8AddWord(a: int, u: int) -> int:
    """A'_ik = A_ik + U_i,k mod 2^8."""
    return (a + u) & 0xFF


@primitive("PouwI8Pm1", 1, [("b", V8), ("f", V1)], V8, conformance="protocol-defined")
def PouwI8Pm1Word(b: int, f: int) -> int:
    """B'_lk = B_lk + F_lk mod 2^8, F = +1 for a clear sign bit f, -1 for a set one."""
    return (b + 1 - 2 * (f & 1)) & 0xFF


@primitive("PouwI8AddBit", 1, [("b", V8), ("d", V1)], V8, conformance="protocol-defined")
def PouwI8AddBitWord(b: int, d: int) -> int:
    """B'_lk = B_lk + d_lk mod 2^8, d decision 29's dither bit."""
    return (b + (d & 1)) & 0xFF


@primitive("PouwSegDot", 1, [("acc", V32), ("a", Array(32, V8)), ("b", Array(32, V8))], V32, conformance="protocol-defined")
def PouwSegDotWord(acc: int, *ab: int) -> int:
    """A prefix word one segment on: acc + sum_k a_k b_k mod 2^32, a and b int8 encodings."""
    return seg_dot(acc, ab[:32], ab[32:])


@primitive("PouwSel8", 1, [("s", V1), ("w", V8)], V8, conformance="protocol-defined")
def PouwSel8Word(s: int, w: int) -> int:
    """w if s else 0."""
    return w & 0xFF if s & 1 else 0


@primitive("PouwXor8", 1, [("x", V8), ("y", V8)], V8, conformance="protocol-defined")
def PouwXor8Word(x: int, y: int) -> int:
    """x ^ y."""
    return (x ^ y) & 0xFF


@primitive("PouwSel32", 1, [("s", V1), ("w", V32)], V32, conformance="protocol-defined")
def PouwSel32Word(s: int, w: int) -> int:
    """w if s else 0."""
    return w & M32 if s & 1 else 0


@primitive("PouwDuoAbsorb", 1, [("st", Value(32 * 2 * NCH)), ("p", Array(32, V32))], Value(32 * 2 * NCH),
           conformance="protocol-defined")
def PouwDuoAbsorbWord(st: int, *p: int) -> int:
    """duo's state (y_0..7, z_0..7, word i at bits 32 i) after absorbing a thread's 32 prefix words p (fgemm.cu's duo)."""
    return _pack(_duo_absorb(_words(st, 2 * NCH), p))


@primitive("PouwDuoFinal", 1, [("st", Value(32 * 2 * NCH))], V32, conformance="protocol-defined")
def PouwDuoFinalWord(st: int) -> int:
    """duo's receipt word of a state: xor_i (y_i ^ rotl(z_i, 16))."""
    return fold_final("duo", _words(st, 2 * NCH))


@primitive("PouwXinAbsorb", 1, [("st", Value(32 * NCH)), ("p", Array(32, V32))], Value(32 * NCH),
           conformance="protocol-defined")
def PouwXinAbsorbWord(st: int, *p: int) -> int:
    """XOR-in's state (y_0..7) after absorbing a thread's 32 prefix words p, rotations (13, 19)."""
    return _pack(_xin_absorb(_words(st, NCH), p))


@primitive("PouwXinFinal", 1, [("st", Value(32 * NCH))], V32, conformance="protocol-defined")
def PouwXinFinalWord(st: int) -> int:
    """XOR-in's receipt word of a state: xor_t y_t."""
    return fold_final("xin", _words(st, NCH))


@primitive("PouwWinOk", 1, [("r", V32), ("w", V32), ("tj", Array(32, V32)), ("tw", Array(32, V32)), ("p0", Array(32, V32)),
                            ("first", V1)], V1, conformance="protocol-defined")
def PouwWinOkWord(r: int, w: int, *rest: int) -> int:
    """1 when r = w, tj = tw and, in the first window, every p0 word is 0."""
    tj, tw, p0, first = rest[:32], rest[32:64], rest[64:96], rest[96] & 1
    return int(r & M32 == w & M32 and all(x & M32 == y & M32 for x, y in zip(tj, tw))
               and (not first or all(v & M32 == 0 for v in p0)))


# --- Boolean twins ------------------------------------------------------------------------------------------------------------

def _i8_add(C, a, u):
    return add(C, a, u, width=8)


def _i8_pm1(C, b, f):
    return add(C, b, [ONE] + [f[0]] * 7, width=8)


def _i8_add_bit(C, b, d):
    return add(C, b, [d[0]] + [ZERO] * 7, width=8)


def _seg_dot(C, acc, *ab):
    """acc + sum_k a_k b_k: every product's Baugh-Wooley partial products (an AND each, the sign row's and column's
    inverted), acc's bits and the products' constant 32 (2^8 - 2^15) in one carry-save tree, its low 32 bits committed."""
    a, b = ab[:32], ab[32:]
    cols: list[list] = [[] for _ in range(32)]
    for x, w in zip(a, b):
        for i in range(8):
            for j in range(8):
                v = C.AND(x[i], w[j])
                cols[i + j].append(NOT(v) if (i == 7) != (j == 7) else v)
    for i, v in enumerate(acc):
        cols[i].append(v)
    for i, v in enumerate(const_bits((32 * ((1 << 8) - (1 << 15))) & M32, 32)):
        if v.c:
            cols[i].append(v)
    return C.commit(compress_columns(C, cols)[:32])


def _mul_add(C, u, k: int, z):
    """u k + z mod 2^32 for a constant k: u's shifts at k's set bits and z in one carry-save tree."""
    cols: list[list] = [[] for _ in range(32)]
    for s in range(32):
        if k >> s & 1:
            for i in range(32 - s):
                cols[i + s].append(u[i])
    for i, v in enumerate(z):
        cols[i].append(v)
    return compress_columns(C, cols)[:32]


def _rotl(x, r: int):
    return [x[(i - r) % 32] for i in range(32)]


def _split(st, n: int) -> list[list]:
    return [list(st[32 * i:32 * i + 32]) for i in range(n)]


def _duo_absorb_bits(C, st, *p):
    w = _split(st, 2 * NCH)
    y, z = w[:NCH], w[NCH:]
    for t in range(MI * NI):
        h = 4 * (t % 2)
        for e in range(4):
            j = h + e
            u = [yy ^ pp ^ c for yy, pp, c in zip(y[j], p[4 * t + e], const_bits(KK[t], 32))]
            y[j] = C.commit(_mul_add(C, u, KA[t], z[j]))
            z[j] = C.commit(add(C, _rotl(u, ROT[t]), y[j], width=32))
    return [v for word in y + z for v in word]


def _duo_final_bits(C, st):
    w = _split(st, 2 * NCH)
    out = []
    for i in range(32):
        v = w[0][i] ^ w[NCH][(i - 16) % 32]
        for j in range(1, NCH):
            v = v ^ w[j][i] ^ w[NCH + j][(i - 16) % 32]
        out.append(v)
    return out


def _xin_absorb_bits(C, st, *p):
    y = _split(st, NCH)
    for t in range(MI * NI):
        u = [a ^ b for a, b in zip(_rotl(y[t], XROT[0]), p[4 * t])]
        y[t] = C.commit(_mul_add(C, u, XA0[t], p[4 * t + 1]))
        u = [a ^ b for a, b in zip(_rotl(y[t], XROT[1]), p[4 * t + 2])]
        y[t] = C.commit(_mul_add(C, u, XA1[t], p[4 * t + 3]))
    return [v for word in y for v in word]


def _xin_final_bits(C, st):
    y = _split(st, NCH)
    out = []
    for i in range(32):
        v = y[0][i]
        for j in range(1, NCH):
            v = v ^ y[j][i]
        out.append(v)
    return out


def _sel(C, s, w):
    return [C.AND(s[0], x) for x in w]


def _win_ok(C, r, w, *rest):
    tj, tw, p0, first = rest[:32], rest[32:64], rest[64:96], rest[96][0]
    eq = and_reduce(C, [NOT(x ^ y) for x, y in zip(r, w)] + [NOT(x ^ y) for a, b in zip(tj, tw) for x, y in zip(a, b)])
    zero = NOT(or_reduce(C, [v for word in p0 for v in word]))
    return [C.AND(eq, NOT(C.AND(first, NOT(zero))))]


DS, XS = Array(32 * 2 * NCH, BIT), Array(32 * NCH, BIT)
PouwI8Add = definition("PouwI8Add", 2, _i8_add, (("a", I8), ("u", I8)), I8, word=PouwI8AddWord,
                       conformance="protocol-defined", doc="a + u mod 2^8 on bits: PouwI8Add_v1")
PouwI8Pm1 = definition("PouwI8Pm1", 2, _i8_pm1, (("b", I8), ("f", BIT)), I8, word=PouwI8Pm1Word,
                       conformance="protocol-defined", doc="b + 1 - 2 f mod 2^8 on bits: PouwI8Pm1_v1")
PouwI8AddBit = definition("PouwI8AddBit", 2, _i8_add_bit, (("b", I8), ("d", BIT)), I8, word=PouwI8AddBitWord,
                          conformance="protocol-defined", doc="b + d mod 2^8 on bits: PouwI8AddBit_v1")
PouwSegDot = definition("PouwSegDot", 2, _seg_dot, (("acc", U32), ("a", Array(32, I8)), ("b", Array(32, I8))), U32,
                        word=PouwSegDotWord, conformance="protocol-defined",
                        doc="acc + sum_k a_k b_k mod 2^32 (int8) on bits: PouwSegDot_v1")
PouwSel8 = definition("PouwSel8", 2, _sel, (("s", BIT), ("w", I8)), I8, word=PouwSel8Word,
                      conformance="protocol-defined", doc="w if s else 0 on bits: PouwSel8_v1")
PouwXor8 = definition("PouwXor8", 2, lambda C, x, y: [a ^ b for a, b in zip(x, y)], (("x", I8), ("y", I8)), I8,
                      word=PouwXor8Word, conformance="protocol-defined", doc="x ^ y on bits: PouwXor8_v1")
PouwSel32 = definition("PouwSel32", 2, _sel, (("s", BIT), ("w", U32)), U32, word=PouwSel32Word,
                       conformance="protocol-defined", doc="w if s else 0 on bits: PouwSel32_v1")
PouwDuoAbsorb = definition("PouwDuoAbsorb", 2, _duo_absorb_bits, (("st", DS), ("p", Array(32, U32))), DS,
                           word=PouwDuoAbsorbWord, conformance="protocol-defined",
                           doc="duo's state after a thread's 32 prefix words, on bits: PouwDuoAbsorb_v1")
PouwDuoFinal = definition("PouwDuoFinal", 2, _duo_final_bits, (("st", DS),), U32, word=PouwDuoFinalWord,
                          conformance="protocol-defined", doc="duo's receipt word of a state, on bits: PouwDuoFinal_v1")
PouwXinAbsorb = definition("PouwXinAbsorb", 2, _xin_absorb_bits, (("st", XS), ("p", Array(32, U32))), XS,
                           word=PouwXinAbsorbWord, conformance="protocol-defined",
                           doc="XOR-in's state after a thread's 32 prefix words, on bits: PouwXinAbsorb_v1")
PouwXinFinal = definition("PouwXinFinal", 2, _xin_final_bits, (("st", XS),), U32, word=PouwXinFinalWord,
                          conformance="protocol-defined", doc="XOR-in's receipt word of a state, on bits: PouwXinFinal_v1")
PouwWinOk = definition("PouwWinOk", 2, _win_ok, (("r", U32), ("w", U32), ("tj", Array(32, U32)), ("tw", Array(32, U32)),
                                                 ("p0", Array(32, U32)), ("first", BIT)), BIT, word=PouwWinOkWord,
                       conformance="protocol-defined", doc="r = w, tj = tw and (first ? p0 = 0 : 1) on bits: PouwWinOk_v1")

#: a fold's absorb and final Definitions and its state's bits
FOLDS = {"duo": (PouwDuoAbsorb, PouwDuoFinal, 32 * 2 * NCH), "xin": (PouwXinAbsorb, PouwXinFinal, 32 * NCH)}


def _xor_tree(B, words: list, xor):
    while len(words) > 1:
        words = [B.batch(xor, words[i], words[i + 1], axes=(0, 0)) if xor is PouwXor8 else B.call(xor, words[i], words[i + 1])
                 for i in range(0, len(words) - 1, 2)] + words[len(words) & ~1:]
    return words[0]


def _check_signature(S):
    if S.FOLD not in FOLDS:
        raise ValueError(f"PouwSegCheck: FOLD {S.FOLD!r} is none of {sorted(FOLDS)}")
    return ((("a", Array(RL, Array(32, I8))), ("asel", Array(16, BIT)), ("u", Array(4, Array(32, I8))),
             ("b", Array(RL, Array(32, I8))), ("bsel", Array(8, BIT)), ("f", Array(8, Array(32, BIT))),
             ("p", Array(S.S + 1, Array(32, U32))), ("st", Array(FOLDS[S.FOLD][2] // 32, U32)), ("rc", Array(RW, U32)),
             ("rsel", Array(RW // 32, U32)), ("jsel", Array(jsel_bits(S.S), BIT)), ("first", BIT)), BIT)


def jsel_bits(s: int) -> int:
    """jsel's width: S one-hot bits and zero bits up to the width that makes bsel, jsel and first whole u16 words (a
    statement's rows are)."""
    return 16 * -(-(s + 9) // 16) - 9


@composite("PouwSegCheck", 2, ["S", "FOLD"], _check_signature, conformance="protocol-defined")
def PouwSegCheck(B, S, a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first):
    """One thread's spot check at the segment ``jsel`` places in its window: the thread's rows out of A's and B's tile leaves
    (`PouwSel8_v2` and `PouwXor8_v2`), A' and B' (`PouwI8Add_v2`, `PouwI8Pm1_v2`), T_(j-1) and the set after it out of p
    (`PouwSel32_v2`, `PouwXor32_v2`), T_j (`PouwSegDot_v2` an entry), the window's fold (FOLD's absorb on each of p's S + 1
    sets, then its final), the leaf's word (`PouwSel32_v2`), and `PouwWinOk_v2`."""
    return _seg_check_body(B, S, PouwI8Pm1, a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first)


@composite("PouwSegCheckDither", 2, ["S", "FOLD"], _check_signature, conformance="protocol-defined")
def PouwSegCheckDither(B, S, a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first):
    """`PouwSegCheck_v2` with the served side under decision 29's dither: B' = B + f (`PouwI8AddBit_v2`), f the coin bit."""
    return _seg_check_body(B, S, PouwI8AddBit, a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first)


def _pick_row(B, leaf, sel, offs: list[int]):
    return _xor_tree(B, [B.batch(PouwSel8, sel[i], leaf[o], axes=(None, 0)) for i, o in enumerate(offs)], PouwXor8)


def _seg_check_body(B, S, served, a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first):
    if S.S < 1:
        raise ValueError("PouwSegCheck: a window of at least one segment")
    absorb, final, _ = FOLDS[S.FOLD]
    ar = [_pick_row(B, a, asel, [a_offset(i) + 8 * r for i in range(16)]) for r in range(4)]
    br = [_pick_row(B, b, bsel, [b_offset(i) + 8 * (c >> 1) + (c & 1) for i in range(8)]) for c in range(8)]
    ap = [B.batch(PouwI8Add, ar[r], u[r], axes=(0, 0)) for r in range(4)]
    bp = [B.batch(served, br[c], f[c], axes=(0, 0)) for c in range(8)]
    prev = [_xor_tree(B, [B.call(PouwSel32, jsel[i], p[i][q]) for i in range(S.S)], PouwXor32) for q in range(32)]
    tw = array_of([_xor_tree(B, [B.call(PouwSel32, jsel[i], p[i + 1][q]) for i in range(S.S)], PouwXor32) for q in range(32)])
    tj = array_of([B.call(PouwSegDot, prev[q], ap[row_slot(q)], bp[col_slot(q)]) for q in range(32)])
    state = st.reshape(Array(FOLDS[S.FOLD][2], BIT))
    for i in range(S.S + 1):
        state = B.call(absorb, state, p[i])
    r = B.call(final, state)
    rsel = rsel.reshape(Array(RW, BIT))
    w = _xor_tree(B, [B.call(PouwSel32, rsel[i], rc[i]) for i in range(RW)], PouwXor32)
    return B.call(PouwWinOk, r, w, tj, tw, p[0], first)
