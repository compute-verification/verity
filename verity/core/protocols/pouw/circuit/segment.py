"""The compute protocol's per-segment duo receipts and their spot check (`benchmarks/e2e_slice`), as word primitives and
Boolean Definitions.

The edited GEMM serves Y = A B'^T from P = A' B'^T (A' = A + U W^T, U in {±1..±64} with period 32 along k; B' = B + F, F ±1)
on 128 x 128 tiles of 512 threads (fgemm.cu's mainloop at warps 32 x 32). Thread ``ctid`` (warp w = ctid / 32, lane l,
wm = w / 4, wn = w % 4, g = l >> 2, q = l & 3) holds 32 accumulator words: entry 4 t + e (t = 4 mi + ni, mi < 2, ni < 4,
e < 4) at row wm 32 + g + 8 r and column wn 32 + 2 q + 8 ni + (e & 1), with r = 2 mi + (e >> 1) its row slot and
c = 2 ni + (e & 1) its column slot. After each 32-step segment s (k = 32 s .. 32 s + 31) the thread absorbs its 32 prefix
words T_s into 8 duo chains, chain 4 (t % 2) + e taking entry 4 t + e at position t, and commits one receipt word

    R_s = xor_j (y_j ^ rotl(z_j, 16)),     chains keyed y_j = mix32(seed ^ mix32(((tile 512 + ctid) KS + s) 8 + j)),
                                           z_j = mix32(y_j + 0x9E3779B9), KS = k / 32 segments,

so every prefix word is committed, a word a (thread, segment). Absorbing p at position t is duo's step (fgemm.cu's
``duo``): u = y ^ p ^ kx_t, y = u a_t + z, z = rotl(u, r_t) + y (mod 2^32, a_t odd), with fgemm.cu's lane-0 keys.

    PouwI8Add_v1     (a, u: Value<8>) -> Value<8>                  a + u mod 2^8 (A' from A and U)
    PouwI8Pm1_v1     (b: Value<8>, f: Value<1>) -> Value<8>        b + 1 - 2 f mod 2^8 (B' from B and F's sign bit)
    PouwSegDot_v1    (acc: Value<32>, a, b: Array<32, Value<8>>) -> Value<32>      acc + sum_k a_k b_k mod 2^32, int8 a and b
    PouwDuoSeg_v1    (y, z: Array<8, Value<32>>, p: Array<32, Value<32>>) -> Value<32>   the receipt word of p from y, z
    PouwSel32_v1     (s: Value<1>, w: Value<32>) -> Value<32>      w if s else 0
    PouwSegOk_v1     (rp, rj, wp, wj: Value<32>, prev: Array<32, Value<32>>, first: Value<1>) -> Value<1>
                     rj = wj, and (first ? prev = 0 : rp = wp)

Each has a Boolean twin (``_v2``, traced from a `forms` builder, word view its ``_v1``). The spot check of one thread at
segment j is

    PouwSegCheck_v1{KS}  (a: Array<4, Array<32, i8>>, u: Array<4, Array<32, i8>>, b: Array<8, Array<32, i8>>,
                          f: Array<8, Array<32, bit>>, prev: Array<32, u32>, st: Array<2, Array<16, u32>>,
                          rc: Array<KS, u32>, sel: Array<KS, bit>) -> bit          (bits, LSB first)

over the thread's 4 rows of A and U and 8 rows of B and F at segment j (row slot r, column slot c), its 32 prefix words
after segment j - 1 (``prev``, zero at j = 0), the chain keys of segments j - 1 and j (``st[i]``: y then z), its committed
receipt row (``rc``, the KS words) and the one-hot j (``sel``). It computes T_j = prev + A'_j B'_j^T entry by entry and both
receipt words, and returns 1 exactly when R(T_j) is ``rc[j]`` and, past j = 0, R(prev) is ``rc[j - 1]``. A statement
holds ``a``, ``b`` and ``rc`` to the trees committed before and after the coin, recomputes ``u``, ``f``, ``st`` and
``sel`` from the coin and the draw, and reads the result through a row of known content: so a proof says the thread's
committed receipt words j - 1 and j are those of the committed operands' noised product, and through the chain of
prefix words the last one, T_(KS-1), is the noised product itself (the served Y is it less the peel U S^T).
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, add, and_reduce, compress_columns, const_bits, or_reduce
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import composite, primitive
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Value

from .boolean import PouwXor32

__all__ = ["KA", "KK", "MI", "NCH", "NCT", "NI", "PouwDuoSeg", "PouwDuoSegWord", "PouwI8Add", "PouwI8AddWord", "PouwI8Pm1",
           "PouwI8Pm1Word", "PouwSegCheck", "PouwSegDot", "PouwSegDotWord", "PouwSegOk", "PouwSegOkWord", "PouwSel32",
           "PouwSel32Word", "ROT", "col_slot", "duo_seg", "init", "mix32", "row_slot", "seg_check", "seg_dot"]

BIT = BL.BIT
U32 = Array(32, BIT)
I8 = Array(8, BIT)
V1, V8, V32 = Value(1), Value(8), Value(32)
M32 = (1 << 32) - 1
#: threads a tile, warp tiles, chains a thread
NCT, MI, NI, NCH = 512, 2, 4, 8
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


def row_slot(q: int) -> int:
    """Entry q's row slot r (row wm 32 + g + 8 r)."""
    t, e = divmod(q, 4)
    return 2 * (t // NI) + (e >> 1)


def col_slot(q: int) -> int:
    """Entry q's column slot c (column wn 32 + 2 q' + 8 (c >> 1) + (c & 1))."""
    t, e = divmod(q, 4)
    return 2 * (t % NI) + (e & 1)


def init(seed: int, tile: int, ctid: int, s: int, ks: int) -> tuple[list[int], list[int]]:
    """The chain keys (y, z) of thread ``ctid`` of ``tile`` at segment ``s`` of ``ks``."""
    y = [mix32((seed ^ mix32(((tile * NCT + ctid) * ks + s) * NCH + j)) & M32) for j in range(NCH)]
    return y, [mix32(v + GOLDEN) for v in y]


def _sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def seg_dot(acc: int, a, b) -> int:
    return (acc + sum(_sx8(x) * _sx8(w) for x, w in zip(a, b, strict=True))) & M32


def duo_seg(y, z, p) -> int:
    y, z = [v & M32 for v in y], [v & M32 for v in z]
    for t in range(MI * NI):
        h = 4 * (t % 2)
        for e in range(4):
            u = y[h + e] ^ (p[4 * t + e] & M32) ^ KK[t]
            y[h + e] = (u * KA[t] + z[h + e]) & M32
            z[h + e] = (rotl(u, ROT[t]) + y[h + e]) & M32
    out = 0
    for j in range(NCH):
        out ^= y[j] ^ rotl(z[j], 16)
    return out


def seg_check(a, u, b, f, prev, st, rc, sel) -> int:
    """`PouwSegCheck_v1` on words: a, u (4 x 32), b (8 x 32) int8 encodings, f (8 x 32) bits, prev (32), st (2 x 16), rc and
    sel (KS each)."""
    ap = [[(x + v) & 0xFF for x, v in zip(a[r], u[r])] for r in range(4)]
    bp = [[(x + 1 - 2 * (s & 1)) & 0xFF for x, s in zip(b[c], f[c])] for c in range(8)]
    tj = [seg_dot(prev[q], ap[row_slot(q)], bp[col_slot(q)]) for q in range(32)]
    wj = wp = 0
    for i, s in enumerate(sel):
        if s & 1:
            wj ^= rc[i]
            if i:
                wp ^= rc[i - 1]
    first = sel[0] & 1
    return int(duo_seg(st[1][:8], st[1][8:], tj) == wj and (all(v & M32 == 0 for v in prev) if first
                                                             else duo_seg(st[0][:8], st[0][8:], prev) == wp))


# --- word primitives ----------------------------------------------------------------------------------------------------------

@primitive("PouwI8Add", 1, [("a", V8), ("u", V8)], V8, conformance="protocol-defined")
def PouwI8AddWord(a: int, u: int) -> int:
    """A'_ik = A_ik + U_i,k mod 2^8."""
    return (a + u) & 0xFF


@primitive("PouwI8Pm1", 1, [("b", V8), ("f", V1)], V8, conformance="protocol-defined")
def PouwI8Pm1Word(b: int, f: int) -> int:
    """B'_lk = B_lk + F_lk mod 2^8, F = +1 for a clear sign bit f, -1 for a set one."""
    return (b + 1 - 2 * (f & 1)) & 0xFF


@primitive("PouwSegDot", 1, [("acc", V32), ("a", Array(32, V8)), ("b", Array(32, V8))], V32, conformance="protocol-defined")
def PouwSegDotWord(acc: int, *ab: int) -> int:
    """A prefix word one segment on: acc + sum_k a_k b_k mod 2^32, a and b int8 encodings."""
    return seg_dot(acc, ab[:32], ab[32:])


@primitive("PouwDuoSeg", 1, [("y", Array(NCH, V32)), ("z", Array(NCH, V32)), ("p", Array(32, V32))], V32,
           conformance="protocol-defined")
def PouwDuoSegWord(*w: int) -> int:
    """A thread's receipt word of its 32 prefix words p from chain keys y, z (fgemm.cu's DUOS)."""
    return duo_seg(w[:NCH], w[NCH:2 * NCH], w[2 * NCH:])


@primitive("PouwSel32", 1, [("s", V1), ("w", V32)], V32, conformance="protocol-defined")
def PouwSel32Word(s: int, w: int) -> int:
    """w if s else 0."""
    return w & M32 if s & 1 else 0


@primitive("PouwSegOk", 1, [("rp", V32), ("rj", V32), ("wp", V32), ("wj", V32), ("prev", Array(32, V32)), ("first", V1)], V1,
           conformance="protocol-defined")
def PouwSegOkWord(rp: int, rj: int, wp: int, wj: int, *rest: int) -> int:
    """1 when rj = wj and, at the first segment, every prev word is 0, else rp = wp."""
    prev, first = rest[:32], rest[32] & 1
    return int(rj & M32 == wj & M32 and (all(v & M32 == 0 for v in prev) if first else rp & M32 == wp & M32))


# --- Boolean twins ------------------------------------------------------------------------------------------------------------

def _i8_add(C, a, u):
    return add(C, a, u, width=8)


def _i8_pm1(C, b, f):
    return add(C, b, [ONE] + [f[0]] * 7, width=8)


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


def _duo_seg(C, *w):
    y, z, p = [list(v) for v in w[:NCH]], [list(v) for v in w[NCH:2 * NCH]], w[2 * NCH:]
    for t in range(MI * NI):
        h = 4 * (t % 2)
        for e in range(4):
            j = h + e
            u = [yy ^ pp ^ c for yy, pp, c in zip(y[j], p[4 * t + e], const_bits(KK[t], 32))]
            y[j] = C.commit(_mul_add(C, u, KA[t], z[j]))
            z[j] = C.commit(add(C, _rotl(u, ROT[t]), y[j], width=32))
    out = []
    for i in range(32):
        v = y[0][i] ^ z[0][(i - 16) % 32]
        for j in range(1, NCH):
            v = v ^ y[j][i] ^ z[j][(i - 16) % 32]
        out.append(v)
    return out


def _sel32(C, s, w):
    return [C.AND(s[0], x) for x in w]


def _seg_ok(C, rp, rj, wp, wj, *rest):
    prev, first = rest[:32], rest[32][0]
    eqj = and_reduce(C, [NOT(x ^ y) for x, y in zip(rj, wj)])
    eqp = and_reduce(C, [NOT(x ^ y) for x, y in zip(rp, wp)])
    zero = NOT(or_reduce(C, [v for word in prev for v in word]))
    return [C.AND(eqj, eqp ^ C.AND(first, zero ^ eqp))]


PouwI8Add = definition("PouwI8Add", 2, _i8_add, (("a", I8), ("u", I8)), I8, word=PouwI8AddWord,
                       conformance="protocol-defined", doc="a + u mod 2^8 on bits: PouwI8Add_v1")
PouwI8Pm1 = definition("PouwI8Pm1", 2, _i8_pm1, (("b", I8), ("f", BIT)), I8, word=PouwI8Pm1Word,
                       conformance="protocol-defined", doc="b + 1 - 2 f mod 2^8 on bits: PouwI8Pm1_v1")
PouwSegDot = definition("PouwSegDot", 2, _seg_dot, (("acc", U32), ("a", Array(32, I8)), ("b", Array(32, I8))), U32,
                        word=PouwSegDotWord, conformance="protocol-defined",
                        doc="acc + sum_k a_k b_k mod 2^32 (int8) on bits: PouwSegDot_v1")
PouwDuoSeg = definition("PouwDuoSeg", 2, _duo_seg, (("y", Array(NCH, U32)), ("z", Array(NCH, U32)), ("p", Array(32, U32))), U32,
                        word=PouwDuoSegWord, conformance="protocol-defined",
                        doc="a thread's per-segment duo receipt word on bits: PouwDuoSeg_v1")
PouwSel32 = definition("PouwSel32", 2, _sel32, (("s", BIT), ("w", U32)), U32, word=PouwSel32Word,
                       conformance="protocol-defined", doc="w if s else 0 on bits: PouwSel32_v1")
PouwSegOk = definition("PouwSegOk", 2, _seg_ok, (("rp", U32), ("rj", U32), ("wp", U32), ("wj", U32), ("prev", Array(32, U32)),
                                                 ("first", BIT)), BIT, word=PouwSegOkWord, conformance="protocol-defined",
                       doc="rj = wj and (first ? prev = 0 : rp = wp) on bits: PouwSegOk_v1")


def _xor_tree(B, words: list):
    while len(words) > 1:
        words = [B.call(PouwXor32, words[i], words[i + 1]) for i in range(0, len(words) - 1, 2)] + words[len(words) & ~1:]
    return words[0]


@composite("PouwSegCheck", 1, ["KS"],
           lambda S: ((("a", Array(4, Array(32, I8))), ("u", Array(4, Array(32, I8))), ("b", Array(8, Array(32, I8))),
                       ("f", Array(8, Array(32, BIT))), ("prev", Array(32, U32)), ("st", Array(2, Array(2 * NCH, U32))),
                       ("rc", Array(S.KS, U32)), ("sel", Array(S.KS, BIT))), BIT),
           conformance="protocol-defined")
def PouwSegCheck(B, S, a, u, b, f, prev, st, rc, sel):
    """One thread's spot check at the segment ``sel`` names: T_j = prev + A'_j B'_j^T (`PouwSegDot_v2` an entry, on A' and B'
    formed once a row), R(prev) and R(T_j) (`PouwDuoSeg_v2`), the receipt row's words j - 1 and j (`PouwSel32_v2`, an xor
    tree of `PouwXor32_v2`), and `PouwSegOk_v2`."""
    if S.KS < 2:
        raise ValueError("PouwSegCheck: at least two segments")
    ap = [B.batch(PouwI8Add, a[r], u[r], axes=(0, 0)) for r in range(4)]
    bp = [B.batch(PouwI8Pm1, b[c], f[c], axes=(0, 0)) for c in range(8)]
    tj = array_of([B.call(PouwSegDot, prev[q], ap[row_slot(q)], bp[col_slot(q)]) for q in range(32)])
    rp = B.call(PouwDuoSeg, st[0][0:NCH], st[0][NCH:2 * NCH], prev)
    rj = B.call(PouwDuoSeg, st[1][0:NCH], st[1][NCH:2 * NCH], tj)
    wj = _xor_tree(B, [B.call(PouwSel32, sel[s], rc[s]) for s in range(S.KS)])
    wp = _xor_tree(B, [B.call(PouwSel32, sel[s + 1], rc[s]) for s in range(S.KS - 1)])
    return B.call(PouwSegOk, rp, rj, wp, wj, prev, sel[0])
