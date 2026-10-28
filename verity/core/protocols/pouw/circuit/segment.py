"""The compute protocol's receipts and their spot checks, as word primitives and Boolean Definitions.

The edited GEMM serves Y = A B'^T from P = A' B'^T (A' = A + U W^T, U in {±1..±64} with period 32 along k; B' = B + F, F ±1,
or under the dither B' = B + d, d in {0, 1}) on 128 x 128 tiles of 512 threads (fgemm.cu's mainloop at warps
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
- ``xin`` (XOR-in with rotations (13, 19); provisional): state y (8
  chains, duo's y keys), chain t absorbing entries 4 t .. 4 t + 3 as d0 .. d3: y = (rotl(y, 13) ^ d0) a0_t + d1, then
  y = (rotl(y, 19) ^ d2) a1_t + d3, the multipliers odd and outside Lambda8 (`key_ok`); R = xor_t y_t.

Operands are committed as tile leaves: R = 64 rows x 32 values of an operand (row block i / 64, segment j).
The thread's 4 rows of A at segment j are rows o + 8 r of A's leaf, o = 32 (wm % 2) + g (16 offsets); its 8 rows of B are
rows o' + 8 (c >> 1) + (c & 1) of B's leaf, o' = 32 (wn % 2) + 2 q (8 offsets).

    PouwI8Add_v1     (a, u: Value<8>) -> Value<8>                  a + u mod 2^8 (A' from A and U)
    PouwI8Pm1_v1     (b: Value<8>, f: Value<1>) -> Value<8>        b + 1 - 2 f mod 2^8 (B' from B and F's sign bit)
    PouwI8AddBit_v1  (b: Value<8>, d: Value<1>) -> Value<8>        b + d mod 2^8 (B' from B and the dither bit)
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

Each has a Boolean twin (``_v2``, traced from a `forms` builder, word view its ``_v1``). The segment dots' twins
(`PouwSegDot_v2`, and `PouwSegDot0_v2` from 0 below) are bound to the wrapping s8 IMMA coordinate they compute
(`census.silicon_step`), so census/v1 counts their gates as `tc:int8`, and serve none of their products (`census.serves_none`:
Lean's `Accounting.Gates.roleOf` puts an H1 tile's lanes at `Site.other`). The spot check of one thread at segment j is

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
and after the coin, recomputes ``u``, ``f``, ``st`` and the selections from the coin and the draw, and reads the result
through a row of known content.

    PouwSegCheckDither_v2{S, FOLD}   the same with the dither on the served side: B' = B + d (`PouwI8AddBit_v2`),
                                     d in {0, 1} the coin bit ``f`` carries (the served X = floor(x / s) + b)

v2 is unsound: it ties only T_j to T_(j-1), so every other set of ``p`` is free
witness, and a meet-in-the-middle over two of their words opens any committed R_w (`tests/test_circuit_segment.py`). The
checks below take no prefix word as witness: every word a fold absorbs is recomputed in the proof from committed operands
and the coin, from T_(-1) = 0, so a thread's (or warp's) whole call is one unit and the committed value is binding by
SHA-512 alone.

    PouwSegCheck_v3{KS, S, FOLD}  (a: Array<KS, Array<64, Array<32, i8>>>, asel: Array<16, bit>, u: Array<4, Array<32, i8>>,
                                   b: Array<KS, Array<64, Array<32, i8>>>, bsel: Array<16, bit>,
                                   f: Array<KS, Array<8, Array<32, bit>>>, st: Array<KS / S, Array<FW, u32>>,
                                   rc: Array<KS / S, Array<512, u32>>, rsel: Array<16, u32>) -> bit
    PouwSegCheckDither_v3{KS, S, FOLD}   the same under the dither (B' = B + f)

v2's receipts (windows of S segments, one word each, the same keys and leaves), checked whole: A's and B's leaves at every
segment, every window's keys and receipt leaf (the thread at the same position ``rsel`` in each), bsel's bits 8.. unread.
1 exactly when every window's fold of T_(wS-1), T_(wS), ..., T_(wS+S-1), the T recomputed, is its leaf's word.

H1 (cgemm.cu's c_l7qs, `benchmarks/pouw/edited_gemm/cgemm_ref.py`) serves O = (W + c) X^T with X = V + b (b the dither
bits) on the tile's rows and W' = W + c + U (c the coin bits, U period 32) on its columns, and absorbs one call-wide fold a
thread: T_0 .. T_(KS-1), then O' = T_last - U A_0 and D_d = U A_d (1 <= d < m) for the digits A_d of X's residue sums
(`call_words`, m = `n_digits`(KS)). fx1 multiplies word q by `fx1_key`(q) first. Its receipt is a thread's 8 chain words
y_i ^ rotl(z_i, 16), R[(tile 512 + ctid) 8 + i]; case (a) (`Accounting.TS.DuoUnits`, `Fx1Units`) needs every chain's full
state, R[(tile 512 + ctid) 16 + i], (y_0..7, z_0..7).

    PouwCallCheck_v1{KS, FOLD, OUT}  (x: Array<KS, Array<64, Array<32, i8>>>, xsel: Array<16, bit>,
                                      b: Array<KS, Array<4, Array<32, bit>>>, w: Array<KS, Array<64, Array<32, i8>>>,
                                      wsel: Array<16, bit>, c: Array<KS, Array<8, Array<32, bit>>>,
                                      u: Array<8, Array<32, i8>>, z: Array<m - 1, Array<4, Array<32, bit>>>,
                                      st: Array<16, u32>, rc: Array<512, u32>, rsel: Array<512 / N, bit>) -> bit

over V's and W's leaves at every segment (the thread's rows at the one-hot xsel and wsel, wsel's bits 8.. unread), the
thread's dither bits, coin bits, U and digit coins z_d(row slot, residue), its keys (duo's y then z), and the receipt
leaf with the thread's N words at slot rsel: FOLD ``duo`` or ``fx1``, OUT ``chains`` (H1's 8 words, N = 8) or ``state``
(the full state, N = 16). 1 exactly when the leaf's N words are the fold's over the recomputed words.

LI-CP-W: li's chains (8 a lane: chain c absorbs `li_entries`(c) as two
half steps a set, y <- (y ^ d0 ^ `LK`[c][h]) `LA`[c][h] + d1, from duo's y keys) and a checkpoint after every set, the XOR
of the warp's chain states (redux.sync.xor), one word a set, KS + m words a warp in leaves of their own.

    PouwWarpCheck_v1{KS, NL}  (x, xsel: Array<NL, ...>, b: Array<NL, ...>, w, wsel: Array<NL, ...>, c: Array<NL, ...>,
                               u: Array<NL, ...>, z: Array<NL, ...>, st: Array<NL, Array<8, u32>>, rc: Array<512 L, u32>) -> bit

`PouwCallCheck_v1`'s ports with a lane axis on the thread's own (NL = 32 a warp; fewer lanes check that many lanes'
XOR), 1 exactly when word s of rc is the checkpoint after set s for every s < KS + m. rc is the warp's receipt, L =
ceil(P (KS + m) / 512) leaves in order (P = 1 here; L = 2 for duo and fx1 at K = 8192: KS = 256, m = 3, 518 words), and
the words after the last checkpoint are unread.

DUO-CP-W and FX1-CP-W: duo's or fx1's chains with a full-state checkpoint after every set, the XOR over the warp
of every chain's y and, separately, of every chain's z (`warp_checkpoint`), two words a set.

    PouwWarpCheck_v2{KS, NL, FOLD}  (PouwWarpCheck_v1's ports, st: Array<NL, Array<P 8, u32>>) -> bit

FOLD li (P = 1, `PouwWarpCheck_v1`), duo or fx1 (P = 2, st each lane's y then z); 1 exactly when words P s .. P s + P - 1
of rc are the checkpoint after set s for every s < KS + m. The checkpoint is a set's, never spaced wider: the
binding proved without a whole-state assumption is a set's 4 words a chain.

A statement holds the operand leaves and ``rc`` to the trees committed before and after the coin, and recomputes the
noise, keys and selections from the coin and the draw.
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
from verity.ml.circuits.census import FAMILY_COUNTS, serves_none, silicon_step
from verity.ml.circuits.defs import composite, primitive
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32_WRAP
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
    "PouwCallCheck",
    "PouwSegCheck",
    "PouwSegCheckDither",
    "PouwSegCheckDitherV3",
    "PouwSegCheckV3",
    "PouwWarpCheck",
    "PouwWarpCheckV2",
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
    "CALL_FOLDS",
    "CALL_OUTS",
    "LA",
    "LK",
    "V3FOLDS",
    "WARP_FOLDS",
    "absorb",
    "call_check",
    "call_words",
    "chain_words",
    "fx1_key",
    "lane_words",
    "li_entries",
    "n_digits",
    "prefixes",
    "seg_check3",
    "thread_rows",
    "warp_check",
    "warp_checkpoint",
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
    """XOR-in's multiplier condition (the Lambda8 half of the filler's `key_ok_r9`): a odd, and neither +a nor -a mod 2^30 in
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
    """B'_lk = B_lk + d_lk mod 2^8, d the dither bit."""
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
    """`PouwSegCheck_v2` with the served side under the dither: B' = B + f (`PouwI8AddBit_v2`), f the coin bit."""
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


# --- every absorbed word recomputed: PouwSegCheck_v3, PouwCallCheck_v1, PouwWarpCheck_v1 and _v2 --------------------------------

def fx1_key(q: int) -> int:
    """fx1's multiplier of entry q (egemm.cu's fx_chain<FX1, t, e>: key_mul(1, 32 q), q = 4 t + e)."""
    return mix32(0xA0000000 + 2 * (16 + 32 * q)) | 1


#: li's multiplier and XOR key of chain c's half step h (key_mul(2, 2 c + h), key_xor(2, 2 c + h))
LA = tuple(tuple(mix32(0xA0000000 + 2 * (32 + 2 * c + h)) | 1 for h in range(2)) for c in range(NCH))
LK = tuple(tuple(mix32(0xB0000000 + 2 * (32 + 2 * c + h) + 1) for h in range(2)) for c in range(NCH))


def li_entries(ch: int) -> list[int]:
    """The entries chain ``ch`` absorbs from a set, in order: 4 t + e for e = ch % 4 and t = ch // 4, + 2, + 4, + 6 (duo's
    chain of an entry); half step h takes the pair (entries 2 h, 2 h + 1)."""
    par, e = divmod(ch, 4)
    return [4 * (par + 2 * i) + e for i in range(4)]


def _fx1_absorb(st: list[int], p) -> list[int]:
    return _duo_absorb(st, [(v * fx1_key(q)) & M32 for q, v in enumerate(p)])


def _li_absorb(st: list[int], p) -> list[int]:
    y = [v & M32 for v in st]
    for ch in range(NCH):
        w = [p[q] & M32 for q in li_entries(ch)]
        for h in range(2):
            y[ch] = ((y[ch] ^ w[2 * h] ^ LK[ch][h]) * LA[ch][h] + w[2 * h + 1]) & M32
    return y


def absorb(fold: str, st: list[int], p) -> list[int]:
    """One set of 32 words into a thread's state, any fold here (duo, xin, fx1, li)."""
    return {"duo": _duo_absorb, "xin": _xin_absorb, "fx1": _fx1_absorb, "li": _li_absorb}[fold](st, p)


def chain_words(st: list[int]) -> list[int]:
    """duo's (or fx1's) receipt of a state chain by chain, H1's 8 words: y_i ^ rotl(z_i, 16)."""
    return [st[i] ^ rotl(st[NCH + i], 16) for i in range(NCH)]


def n_digits(ks: int) -> int:
    """H1's digit count m at KS segments (`checked_strip.n_digits`(-128 KS, 127 KS)): the least m with -128^m <= -128 KS
    and 127 KS // 128^(m - 1) <= 127."""
    m = 1
    while not (-(128 ** m) <= -128 * ks and 127 * ks // 128 ** (m - 1) <= 127):
        m += 1
    return m


def _pick(leaf, sel, offs: list[int]) -> list[int]:
    out = [0] * 32
    for i, o in enumerate(offs):
        if sel[i] & 1:
            out = [x ^ y for x, y in zip(out, leaf[o])]
    return out


def thread_rows(a, asel, b, bsel) -> tuple[list[list[int]], list[list[int]]]:
    """The thread's 4 rows of A's leaf and 8 of B's at its one-hot offsets (``asel`` 16, ``bsel`` its first 8)."""
    return ([_pick(a, asel, [a_offset(i) + 8 * r for i in range(16)]) for r in range(4)],
            [_pick(b, bsel, [b_offset(i) + 8 * (c >> 1) + (c & 1) for i in range(8)]) for c in range(8)])


def prefixes(ap, bp) -> list[list[int]]:
    """T_0 .. T_(KS-1) of a thread from its rows of A' and B' at every segment (T_(-1) = 0)."""
    acc, out = [0] * 32, []
    for ar, br in zip(ap, bp, strict=True):
        acc = [seg_dot(acc[q], ar[row_slot(q)], br[col_slot(q)]) for q in range(32)]
        out.append(acc)
    return out


def seg_check3(a, asel, u, b, bsel, f, st, rc, rsel, fold: str = "duo", dither: bool = False) -> int:
    """`PouwSegCheck_v3` (``dither``: `PouwSegCheckDither_v3`) on words: a and b KS leaves (64 x 32 int8 encodings each),
    asel (16) and bsel (16, its first 8 read) one-hot, u (4 x 32), f (KS x 8 x 32 bits), st (NW windows' fold keys), rc (NW
    receipt leaves of 512 words) and rsel (16 words, the thread's one-hot position)."""
    ap, bp = [], []
    for aj, bj, fj in zip(a, b, f, strict=True):
        ar, br = thread_rows(aj, asel, bj, bsel)
        ap.append([[(x + v) & 0xFF for x, v in zip(ar[r], u[r])] for r in range(4)])
        bp.append([[(x + ((s & 1) if dither else 1 - 2 * (s & 1))) & 0xFF for x, s in zip(br[c], fj[c])] for c in range(8)])
    T = prefixes(ap, bp)
    nw = len(st)
    s = len(a) // nw
    pos = [i for i in range(RW) if rsel[i // 32] >> (i % 32) & 1]
    ok = True
    for w in range(nw):
        state = absorb(fold, list(st[w]), T[w * s - 1] if w else [0] * 32)
        for j in range(w * s, w * s + s):
            state = absorb(fold, state, T[j])
        got = 0
        for i in pos:
            got ^= rc[w][i]
        ok &= fold_final(fold, state) == got & M32
    return int(ok)


def digit_words(X, z, m: int) -> list[list[list[int]]]:
    """H1's digits A_0 .. A_(m-1) (int8 encodings, 4 rows x 32 residues each) of the residue sums Y_r[rho] = sum_s X_s[r][rho]
    (`checked_strip.digits` with z (m - 1 x 4 x 32 bits): A_d = (R_d mod 128) - 128 z_d, R_(d+1) = floor(R_d / 128) + z_d,
    z_(m-2) taken as 0 where floor(R / 128) = 127, A_(m-1) = R_(m-1))."""
    return sum_digits([[sum(_sx8(X[s][r][rho]) for s in range(len(X))) for rho in range(32)] for r in range(4)], z, m)


def sum_digits(Y, z, m: int) -> list[list[list[int]]]:
    """`digit_words` of the residue sums themselves, Y (4 rows x 32 integers)."""
    A = [[[0] * 32 for _ in range(4)] for _ in range(m)]
    for r in range(4):
        for rho in range(32):
            R = Y[r][rho]
            for d in range(m - 1):
                q = R >> 7
                zd = 0 if d == m - 2 and q == 127 else z[d][r][rho] & 1
                A[d][r][rho] = (R & 127) | zd << 7
                R = q + zd
            A[m - 1][r][rho] = R & 0xFF
    return A


def call_words(xr, b, wr, c, u, z) -> list[list[int]]:
    """H1's absorbed sets for one thread (cgemm.cu's c_l7qs, `cgemm_ref.check`): T_0 .. T_(KS-1), O' and D_1 .. D_(m-1) over
    X = V + b (the thread's 4 rows of V at each segment, b the dither bits) and W' = W + c + U (its 8 rows of W, c the coin
    bits, U its rows' 32 values); D_d = U A_d and O' = T_last - U A_0 entry by entry."""
    X = [[[(v + (bb & 1)) & 0xFF for v, bb in zip(xs[r], bs[r])] for r in range(4)] for xs, bs in zip(xr, b, strict=True)]
    Wp = [[[(v + (cc & 1) + uu) & 0xFF for v, cc, uu in zip(ws[k], cs[k], u[k])] for k in range(8)]
          for ws, cs in zip(wr, c, strict=True)]
    T = prefixes(X, Wp)
    m = n_digits(len(xr))
    A = digit_words(X, z, m)
    neg = [[(-v) & 0xFF for v in row] for row in u]
    op = [seg_dot(T[-1][q], neg[col_slot(q)], A[0][row_slot(q)]) for q in range(32)]
    return T + [op] + [[seg_dot(0, u[col_slot(q)], A[d][row_slot(q)]) for q in range(32)] for d in range(1, m)]


def lane_words(x, w, xsel, b, wsel, c, u, z) -> list[list[int]]:
    """`call_words` of the thread whose rows ``xsel`` and ``wsel`` pick out of the leaves ``x`` and ``w``."""
    xr, wr = zip(*[thread_rows(xj, xsel, wj, wsel) for xj, wj in zip(x, w, strict=True)])
    return call_words(xr, b, wr, c, u, z)


def call_check(x, xsel, b, w, wsel, c, u, z, st, rc, rsel, fold: str = "duo", out: str = "chains") -> int:
    """`PouwCallCheck_v1` on words: x and w KS leaves, xsel (16) and wsel (16, its first 8 read) one-hot, b (KS x 4 x 32
    bits), c (KS x 8 x 32 bits), u (8 x 32), z (m - 1 x 4 x 32 bits), st (16 words), rc (512 words), rsel (64 or 32
    bits)."""
    state = list(st)
    for p in lane_words(x, w, xsel, b, wsel, c, u, z):
        state = absorb(fold, state, p)
    want = chain_words(state) if out == "chains" else state
    n = len(want)
    got = [0] * n
    for s, bit in enumerate(rsel):
        if bit & 1:
            got = [g ^ rc[n * s + i] for i, g in enumerate(got)]
    return int([v & M32 for v in got] == [v & M32 for v in want])


def warp_checkpoint(states: list[list[int]]) -> list[int]:
    """A warp's checkpoint from its lanes' states: per part of a state (li: y; duo, fx1: y, then z), the XOR over lanes and
    chains."""
    out = []
    for j in range(len(states[0]) // NCH):
        cp = 0
        for st in states:
            for v in st[NCH * j:NCH * j + NCH]:
                cp ^= v
        out.append(cp & M32)
    return out


def warp_check(x, xsel, b, w, wsel, c, u, z, st, rc, fold: str = "li") -> int:
    """`PouwWarpCheck_v1` (``fold`` li) and `PouwWarpCheck_v2` on words: x and w KS leaves, then per lane xsel, b, wsel, c,
    u, z and st (li: 8 words; duo, fx1: 16) as `call_check`'s, and rc (512 L words, the warp's L receipt leaves: P words
    after each set s at P s .. P s + P - 1, P the state's parts)."""
    states = [list(s) for s in st]
    sets = [lane_words(x, w, xsel[ln], b[ln], wsel[ln], c[ln], u[ln], z[ln]) for ln in range(len(st))]
    ok = True
    for s in range(len(sets[0])):
        states = [absorb(fold, states[ln], sets[ln][s]) for ln in range(len(st))]
        cp = warp_checkpoint(states)
        ok &= cp == [v & M32 for v in rc[len(cp) * s:len(cp) * s + len(cp)]]
    return int(ok)


@primitive("PouwSegDot0", 1, [("a", Array(32, V8)), ("b", Array(32, V8))], V32, conformance="protocol-defined")
def PouwSegDot0Word(*ab: int) -> int:
    """A first prefix word: sum_k a_k b_k mod 2^32, a and b int8 encodings."""
    return seg_dot(0, ab[:32], ab[32:])


@primitive("PouwDuoAbsorb0", 1, [("st", Value(32 * 2 * NCH))], Value(32 * 2 * NCH), conformance="protocol-defined")
def PouwDuoAbsorb0Word(st: int) -> int:
    """duo's state after absorbing 32 zero words (a first window's incoming set, T_(-1) = 0)."""
    return _pack(_duo_absorb(_words(st, 2 * NCH), [0] * 32))


@primitive("PouwXinAbsorb0", 1, [("st", Value(32 * NCH))], Value(32 * NCH), conformance="protocol-defined")
def PouwXinAbsorb0Word(st: int) -> int:
    """XOR-in's state after absorbing 32 zero words."""
    return _pack(_xin_absorb(_words(st, NCH), [0] * 32))


@primitive("PouwFx1Absorb", 1, [("st", Value(32 * 2 * NCH)), ("p", Array(32, V32))], Value(32 * 2 * NCH),
           conformance="protocol-defined")
def PouwFx1AbsorbWord(st: int, *p: int) -> int:
    """fx1's state (duo's y_0..7, z_0..7) after a thread's 32 words p, word q multiplied by `fx1_key`(q) before duo's step."""
    return _pack(_fx1_absorb(_words(st, 2 * NCH), p))


@primitive("PouwLiAbsorb", 1, [("st", Value(32 * NCH)), ("p", Array(32, V32))], Value(32 * NCH), conformance="protocol-defined")
def PouwLiAbsorbWord(st: int, *p: int) -> int:
    """li's state (y_0..7) after a thread's 32 words p: chain c's two half steps y <- (y ^ d0 ^ LK[c][h]) LA[c][h] + d1 over
    the pairs of its entries (`li_entries`)."""
    return _pack(_li_absorb(_words(st, NCH), p))


@primitive("PouwDuoChains", 1, [("st", Value(32 * 2 * NCH))], Value(32 * NCH), conformance="protocol-defined")
def PouwDuoChainsWord(st: int) -> int:
    """duo's receipt chain by chain (H1's 8 words): y_i ^ rotl(z_i, 16) at bits 32 i."""
    return _pack(chain_words(_words(st, 2 * NCH)))


@primitive("PouwEq32", 1, [("x", V32), ("y", V32)], V1, conformance="protocol-defined")
def PouwEq32Word(x: int, y: int) -> int:
    """1 when x = y."""
    return int(x & M32 == y & M32)


@primitive("PouwEqAnd", 1, [("ok", V1), ("x", V32), ("y", V32)], V1, conformance="protocol-defined")
def PouwEqAndWord(ok: int, x: int, y: int) -> int:
    """ok and x = y."""
    return int(bool(ok & 1) and x & M32 == y & M32)


@primitive("PouwSx8", 1, [("x", V8)], V32, conformance="protocol-defined")
def PouwSx8Word(x: int) -> int:
    """x's int8 value as a 32-bit word (a first residue sum)."""
    return _sx8(x) & M32


@primitive("PouwResAdd", 1, [("acc", V32), ("x", V8)], V32, conformance="protocol-defined")
def PouwResAddWord(acc: int, x: int) -> int:
    """acc + x mod 2^32, x int8 (a residue sum one segment on)."""
    return (acc + _sx8(x)) & M32


@primitive("PouwDigitLo", 1, [("r", V32), ("z", V1)], V8, conformance="protocol-defined")
def PouwDigitLoWord(r: int, z: int) -> int:
    """A digit's int8 encoding: (r mod 128) - 128 z."""
    return (r & 127) | (z & 1) << 7


@primitive("PouwDigitHi", 1, [("r", V32), ("z", V1)], V32, conformance="protocol-defined")
def PouwDigitHiWord(r: int, z: int) -> int:
    """The rest after a digit: floor(r / 128) + z mod 2^32, r a 32-bit two's-complement word."""
    r &= M32
    return ((r >> 7 | (M32 << 25 & M32 if r >> 31 else 0)) + (z & 1)) & M32


@primitive("PouwDigitZ", 1, [("r", V32), ("z", V1)], V1, conformance="protocol-defined")
def PouwDigitZWord(r: int, z: int) -> int:
    """The last coin digit's z: z, but 0 where floor(r / 128) = 127 (so the top digit stays int8)."""
    r &= M32
    return 0 if (r >> 7 | (M32 << 25 & M32 if r >> 31 else 0)) == 127 else z & 1


@primitive("PouwLow8", 1, [("r", V32)], V8, conformance="protocol-defined")
def PouwLow8Word(r: int) -> int:
    """r mod 2^8 (the top digit's int8 encoding)."""
    return r & 0xFF


@primitive("PouwI8Neg", 1, [("x", V8)], V8, conformance="protocol-defined")
def PouwI8NegWord(x: int) -> int:
    """-x mod 2^8."""
    return -x & 0xFF


def _seg_dot0(C, *ab):
    return _seg_dot(C, [ZERO] * 32, *ab)


def _mul(C, u, k: int):
    """u k mod 2^32 for a constant k."""
    cols: list[list] = [[] for _ in range(32)]
    for s in range(32):
        if k >> s & 1:
            for i in range(32 - s):
                cols[i + s].append(u[i])
    return compress_columns(C, cols)[:32]


def _zero_set():
    return [const_bits(0, 32) for _ in range(32)]


def _fx1_absorb_bits(C, st, *p):
    return _duo_absorb_bits(C, st, *[C.commit(_mul(C, p[q], fx1_key(q))) for q in range(32)])


def _li_absorb_bits(C, st, *p):
    y = _split(st, NCH)
    for ch in range(NCH):
        w = [p[q] for q in li_entries(ch)]
        for h in range(2):
            u = [a ^ b ^ k for a, b, k in zip(y[ch], w[2 * h], const_bits(LK[ch][h], 32))]
            y[ch] = C.commit(_mul_add(C, u, LA[ch][h], w[2 * h + 1]))
    return [v for word in y for v in word]


def _duo_chains_bits(C, st):
    w = _split(st, 2 * NCH)
    return [w[j][i] ^ w[NCH + j][(i - 16) % 32] for j in range(NCH) for i in range(32)]


def _eq(C, x, y):
    return and_reduce(C, [NOT(a ^ b) for a, b in zip(x, y)])


def _sx8_bits(x):
    return list(x) + [x[7]] * 24


def _shr7(r):
    return list(r[7:]) + [r[31]] * 7


def _digit_z(C, r, z):
    q = _shr7(r)
    top = and_reduce(C, q[:7] + [NOT(v) for v in q[7:]])
    return [C.AND(z[0], NOT(top))]


V512, V256 = Array(32 * 2 * NCH, BIT), Array(32 * NCH, BIT)
PouwSegDot0 = definition("PouwSegDot0", 2, _seg_dot0, (("a", Array(32, I8)), ("b", Array(32, I8))), U32, word=PouwSegDot0Word,
                         conformance="protocol-defined", doc="sum_k a_k b_k mod 2^32 (int8) on bits: PouwSegDot0_v1")
for _dot in (PouwSegDot, PouwSegDot0):
    silicon_step(_dot, BLACKWELL_SM120_S8_M16N8K32_WRAP)
    _dot.counts = FAMILY_COUNTS[f"{_dot.name}_v{_dot.version}"] = serves_none  # type: ignore[attr-defined]
PouwDuoAbsorb0 = definition("PouwDuoAbsorb0", 2, lambda C, st: _duo_absorb_bits(C, st, *_zero_set()), (("st", V512),), V512,
                            word=PouwDuoAbsorb0Word, conformance="protocol-defined",
                            doc="duo's state after 32 zero words, on bits: PouwDuoAbsorb0_v1")
PouwXinAbsorb0 = definition("PouwXinAbsorb0", 2, lambda C, st: _xin_absorb_bits(C, st, *_zero_set()), (("st", V256),), V256,
                            word=PouwXinAbsorb0Word, conformance="protocol-defined",
                            doc="XOR-in's state after 32 zero words, on bits: PouwXinAbsorb0_v1")
PouwFx1Absorb = definition("PouwFx1Absorb", 2, _fx1_absorb_bits, (("st", V512), ("p", Array(32, U32))), V512,
                           word=PouwFx1AbsorbWord, conformance="protocol-defined",
                           doc="fx1's state after a thread's 32 words, on bits: PouwFx1Absorb_v1")
PouwLiAbsorb = definition("PouwLiAbsorb", 2, _li_absorb_bits, (("st", V256), ("p", Array(32, U32))), V256,
                          word=PouwLiAbsorbWord, conformance="protocol-defined",
                          doc="li's state after a thread's 32 words, on bits: PouwLiAbsorb_v1")
PouwDuoChains = definition("PouwDuoChains", 2, _duo_chains_bits, (("st", V512),), V256, word=PouwDuoChainsWord,
                           conformance="protocol-defined", doc="duo's receipt chain by chain, on bits: PouwDuoChains_v1")
PouwEq32 = definition("PouwEq32", 2, lambda C, x, y: [_eq(C, x, y)], (("x", U32), ("y", U32)), BIT, word=PouwEq32Word,
                      conformance="protocol-defined", doc="x = y on bits: PouwEq32_v1")
PouwEqAnd = definition("PouwEqAnd", 2, lambda C, ok, x, y: [C.AND(ok[0], _eq(C, x, y))], (("ok", BIT), ("x", U32), ("y", U32)),
                       BIT, word=PouwEqAndWord, conformance="protocol-defined", doc="ok and x = y on bits: PouwEqAnd_v1")
PouwSx8 = definition("PouwSx8", 2, lambda C, x: _sx8_bits(x), (("x", I8),), U32, word=PouwSx8Word,
                     conformance="protocol-defined", doc="int8 to 32 bits on bits: PouwSx8_v1")
PouwResAdd = definition("PouwResAdd", 2, lambda C, acc, x: add(C, acc, _sx8_bits(x), width=32), (("acc", U32), ("x", I8)), U32,
                        word=PouwResAddWord, conformance="protocol-defined", doc="acc + x (int8) mod 2^32 on bits: PouwResAdd_v1")
PouwDigitLo = definition("PouwDigitLo", 2, lambda C, r, z: list(r[:7]) + [z[0]], (("r", U32), ("z", BIT)), I8,
                         word=PouwDigitLoWord, conformance="protocol-defined", doc="(r mod 128) - 128 z on bits: PouwDigitLo_v1")
PouwDigitHi = definition("PouwDigitHi", 2, lambda C, r, z: add(C, _shr7(r), [z[0]], width=32), (("r", U32), ("z", BIT)), U32,
                         word=PouwDigitHiWord, conformance="protocol-defined",
                         doc="floor(r / 128) + z mod 2^32 on bits: PouwDigitHi_v1")
PouwDigitZ = definition("PouwDigitZ", 2, _digit_z, (("r", U32), ("z", BIT)), BIT, word=PouwDigitZWord,
                        conformance="protocol-defined", doc="z unless floor(r / 128) = 127, on bits: PouwDigitZ_v1")
PouwLow8 = definition("PouwLow8", 2, lambda C, r: list(r[:8]), (("r", U32),), I8, word=PouwLow8Word,
                      conformance="protocol-defined", doc="r mod 2^8 on bits: PouwLow8_v1")
PouwI8Neg = definition("PouwI8Neg", 2, lambda C, x: add(C, [NOT(v) for v in x], [ONE], width=8), (("x", I8),), I8,
                       word=PouwI8NegWord, conformance="protocol-defined", doc="-x mod 2^8 on bits: PouwI8Neg_v1")

#: PouwSegCheck_v3's folds: absorb, absorb of the zero set, final, state bits
V3FOLDS = {"duo": (PouwDuoAbsorb, PouwDuoAbsorb0, PouwDuoFinal, 32 * 2 * NCH),
           "xin": (PouwXinAbsorb, PouwXinAbsorb0, PouwXinFinal, 32 * NCH)}
#: PouwCallCheck_v1's folds, and its receipts: words a thread, threads a leaf
CALL_FOLDS = {"duo": PouwDuoAbsorb, "fx1": PouwFx1Absorb}
CALL_OUTS = {"chains": (NCH, RW // NCH), "state": (2 * NCH, RW // (2 * NCH))}
LEAF = Array(RL, Array(32, I8))


def _rows_of(B, leaf, sel, n: int):
    """The thread's ``n`` rows of an operand's leaf (4 of A's or X's, 8 of B's or W's) at its one-hot offsets."""
    if n == 4:
        return [_pick_row(B, leaf, sel, [a_offset(i) + 8 * r for i in range(16)]) for r in range(4)]
    return [_pick_row(B, leaf, sel, [b_offset(i) + 8 * (c >> 1) + (c & 1) for i in range(8)]) for c in range(8)]


def _prefix(B, prev, ap, bp):
    """The next 32 prefix words (`PouwSegDot0_v2` from T_(-1) = 0, else `PouwSegDot_v2`)."""
    if prev is None:
        return [B.call(PouwSegDot0, ap[row_slot(q)], bp[col_slot(q)]) for q in range(32)]
    return [B.call(PouwSegDot, prev[q], ap[row_slot(q)], bp[col_slot(q)]) for q in range(32)]


def _all_equal(B, pairs):
    ok = B.call(PouwEq32, *pairs[0])
    for x, y in pairs[1:]:
        ok = B.call(PouwEqAnd, ok, x, y)
    return ok


def _check3_signature(S):
    if S.FOLD not in V3FOLDS:
        raise ValueError(f"PouwSegCheck_v3: FOLD {S.FOLD!r} is none of {sorted(V3FOLDS)}")
    if S.S < 1 or S.KS < S.S or S.KS % S.S:
        raise ValueError(f"PouwSegCheck_v3: KS {S.KS} is no positive multiple of S {S.S}")
    nw = S.KS // S.S
    return ((("a", Array(S.KS, LEAF)), ("asel", Array(16, BIT)), ("u", Array(4, Array(32, I8))), ("b", Array(S.KS, LEAF)),
             ("bsel", Array(16, BIT)), ("f", Array(S.KS, Array(8, Array(32, BIT)))),
             ("st", Array(nw, Array(V3FOLDS[S.FOLD][3] // 32, U32))), ("rc", Array(nw, Array(RW, U32))),
             ("rsel", Array(RW // 32, U32))), BIT)


@composite("PouwSegCheck", 3, ["KS", "S", "FOLD"], _check3_signature, conformance="protocol-defined")
def PouwSegCheckV3(B, S, a, asel, u, b, bsel, f, st, rc, rsel):
    """One thread's whole call against its windowed receipts, every absorbed word recomputed: at each segment j the thread's
    rows out of A's and B's leaves (`PouwSel8_v2`, `PouwXor8_v2`), A' and B' (`PouwI8Add_v2`, `PouwI8Pm1_v2`) and T_j
    (`PouwSegDot0_v2` at j = 0, `PouwSegDot_v2` after); each window's fold from its keys over T_(wS-1) (the zero set in
    window 0, `PouwDuoAbsorb0_v2` or `PouwXinAbsorb0_v2`) and its S sets, its final, the leaf's word at rsel
    (`PouwSel32_v2`, `PouwXor32_v2`), and every window's equality (`PouwEq32_v2`, `PouwEqAnd_v2`)."""
    return _seg_check3_body(B, S, PouwI8Pm1, a, asel, u, b, bsel, f, st, rc, rsel)


@composite("PouwSegCheckDither", 3, ["KS", "S", "FOLD"], _check3_signature, conformance="protocol-defined")
def PouwSegCheckDitherV3(B, S, a, asel, u, b, bsel, f, st, rc, rsel):
    """`PouwSegCheck_v3` with the served side under the dither: B' = B + f (`PouwI8AddBit_v2`), f the coin bit."""
    return _seg_check3_body(B, S, PouwI8AddBit, a, asel, u, b, bsel, f, st, rc, rsel)


def _seg_check3_body(B, S, served, a, asel, u, b, bsel, f, st, rc, rsel):
    ab, ab0, final, fw = V3FOLDS[S.FOLD]
    T, prev = [], None
    for j in range(S.KS):
        ar, br = _rows_of(B, a[j], asel, 4), _rows_of(B, b[j], bsel, 8)
        ap = [B.batch(PouwI8Add, ar[r], u[r], axes=(0, 0)) for r in range(4)]
        bp = [B.batch(served, br[c], f[j][c], axes=(0, 0)) for c in range(8)]
        prev = _prefix(B, prev, ap, bp)
        T.append(array_of(prev))
    rsel = rsel.reshape(Array(RW, BIT))
    pairs = []
    for w in range(S.KS // S.S):
        state = st[w].reshape(Array(fw, BIT))
        state = B.call(ab, state, T[w * S.S - 1]) if w else B.call(ab0, state)
        for j in range(w * S.S, w * S.S + S.S):
            state = B.call(ab, state, T[j])
        pairs.append((B.call(final, state), _xor_tree(B, [B.call(PouwSel32, rsel[i], rc[w][i]) for i in range(RW)], PouwXor32)))
    return _all_equal(B, pairs)


def _call_ports(ks: int):
    """One thread's operand and coin ports of H1's call (`PouwCallCheck_v1`'s first eight)."""
    m = n_digits(ks)
    if m < 2:
        raise ValueError(f"H1's call checks: KS {ks} has one digit; KS >= 2")
    return (("x", Array(ks, LEAF)), ("xsel", Array(16, BIT)), ("b", Array(ks, Array(4, Array(32, BIT)))), ("w", Array(ks, LEAF)),
            ("wsel", Array(16, BIT)), ("c", Array(ks, Array(8, Array(32, BIT)))), ("u", Array(8, Array(32, I8))),
            ("z", Array(m - 1, Array(4, Array(32, BIT)))))


def _call_sets(B, ks, x, xsel, b, w, wsel, c, u, z):
    """H1's absorbed sets of one thread (`call_words`): T_0 .. T_(KS-1), O' and D_1 .. D_(m-1)."""
    m = n_digits(ks)
    T, Y, prev = [], [], None
    for s in range(ks):
        xr, wr = _rows_of(B, x[s], xsel, 4), _rows_of(B, w[s], wsel, 8)
        X = [B.batch(PouwI8AddBit, xr[r], b[s][r], axes=(0, 0)) for r in range(4)]
        Wp = [B.batch(PouwI8Add, B.batch(PouwI8AddBit, wr[k], c[s][k], axes=(0, 0)), u[k], axes=(0, 0)) for k in range(8)]
        prev = _prefix(B, prev, X, Wp)
        T.append(prev)
        Y = [B.batch(PouwResAdd, Y[r], X[r], axes=(0, 0)) if s else B.batch(PouwSx8, X[r], axes=(0,)) for r in range(4)]
    A = [[None] * 4 for _ in range(m)]
    for r in range(4):
        R = Y[r]
        for d in range(m - 1):
            zd = B.batch(PouwDigitZ, R, z[d][r], axes=(0, 0)) if d == m - 2 else z[d][r]
            A[d][r] = B.batch(PouwDigitLo, R, zd, axes=(0, 0))
            R = B.batch(PouwDigitHi, R, zd, axes=(0, 0))
        A[m - 1][r] = B.batch(PouwLow8, R, axes=(0,))
    neg = [B.batch(PouwI8Neg, u[k], axes=(0,)) for k in range(8)]
    op = [B.call(PouwSegDot, T[-1][q], neg[col_slot(q)], A[0][row_slot(q)]) for q in range(32)]
    D = [[B.call(PouwSegDot0, u[col_slot(q)], A[d][row_slot(q)]) for q in range(32)] for d in range(1, m)]
    return [array_of(t) for t in [*T, op, *D]]


def _call_signature(S):
    if S.FOLD not in CALL_FOLDS:
        raise ValueError(f"PouwCallCheck_v1: FOLD {S.FOLD!r} is none of {sorted(CALL_FOLDS)}")
    if S.OUT not in CALL_OUTS:
        raise ValueError(f"PouwCallCheck_v1: OUT {S.OUT!r} is none of {sorted(CALL_OUTS)}")
    return ((*_call_ports(S.KS), ("st", Array(2 * NCH, U32)), ("rc", Array(RW, U32)),
             ("rsel", Array(CALL_OUTS[S.OUT][1], BIT))), BIT)


@composite("PouwCallCheck", 1, ["KS", "FOLD", "OUT"], _call_signature, conformance="protocol-defined")
def PouwCallCheck(B, S, x, xsel, b, w, wsel, c, u, z, st, rc, rsel):
    """One thread's whole H1 call against its receipt, every absorbed word recomputed: H1's sets (T_0 .. T_(KS-1) over X =
    V + b and W' = W + c + U; the residue sums, digits, O' and D_d, `call_words`) folded from the thread's keys (FOLD's
    absorb), its receipt (OUT ``chains``: `PouwDuoChains_v2`; ``state``: the state's 16 words) and the leaf's words at rsel,
    equal word by word."""
    state = st.reshape(Array(32 * 2 * NCH, BIT))
    for p in _call_sets(B, S.KS, x, xsel, b, w, wsel, c, u, z):
        state = B.call(CALL_FOLDS[S.FOLD], state, p)
    n, slots = CALL_OUTS[S.OUT]
    want = (B.call(PouwDuoChains, state) if S.OUT == "chains" else state).reshape(Array(n, U32))
    got = [_xor_tree(B, [B.call(PouwSel32, rsel[s], rc[n * s + i]) for s in range(slots)], PouwXor32) for i in range(n)]
    return _all_equal(B, [(want[i], got[i]) for i in range(n)])


#: the warp checks' folds: absorb, and a lane's state words (li: y; duo, fx1: y, then z)
WARP_FOLDS = {"li": (PouwLiAbsorb, NCH), "duo": (PouwDuoAbsorb, 2 * NCH), "fx1": (PouwFx1Absorb, 2 * NCH)}


def _warp_ports(name: str, ks: int, nl: int, fold: str):
    if fold not in WARP_FOLDS:
        raise ValueError(f"{name}: FOLD {fold!r} is none of {sorted(WARP_FOLDS)}")
    if not 1 <= nl <= 32:
        raise ValueError(f"{name}: NL {nl} lanes is not 1 .. 32")
    words = WARP_FOLDS[fold][1]
    ports = dict(_call_ports(ks))
    n = words // NCH * (ks + n_digits(ks))
    lane = ("xsel", "b", "wsel", "c", "u", "z")
    return ((("x", ports["x"]), *[(k, Array(nl, ports[k])) for k in lane[:2]], ("w", ports["w"]),
             *[(k, Array(nl, ports[k])) for k in lane[2:]], ("st", Array(nl, Array(words, U32))),
             ("rc", Array(RW * -(-n // RW), U32))), BIT)


def _warp_signature(S):
    return _warp_ports("PouwWarpCheck_v1", S.KS, S.NL, "li")


def _warp2_signature(S):
    return _warp_ports("PouwWarpCheck_v2", S.KS, S.NL, S.FOLD)


def _warp_checkpoint(B, states, parts: int):
    """A warp's checkpoint words from its lanes' states (`warp_checkpoint`): per part, each lane's XOR over its chains
    (`PouwXinFinal_v2`), then the XOR across lanes (`PouwXor32_v2`, redux.sync.xor)."""
    return [_xor_tree(B, [B.call(PouwXinFinal, s.reshape(Array(parts, XS))[j]) for s in states], PouwXor32)
            for j in range(parts)]


def _warp_body(B, ks, nl, fold, x, xsel, b, w, wsel, c, u, z, st, rc):
    ab, words = WARP_FOLDS[fold]
    parts = words // NCH
    sets = [_call_sets(B, ks, x, xsel[ln], b[ln], w, wsel[ln], c[ln], u[ln], z[ln]) for ln in range(nl)]
    states = [st[ln].reshape(Array(32 * words, BIT)) for ln in range(nl)]
    pairs = []
    for s in range(len(sets[0])):
        states = [B.call(ab, states[ln], sets[ln][s]) for ln in range(nl)]
        pairs += [(cp, rc[parts * s + j]) for j, cp in enumerate(_warp_checkpoint(B, states, parts))]
    return _all_equal(B, pairs)


@composite("PouwWarpCheck", 1, ["KS", "NL"], _warp_signature, conformance="protocol-defined")
def PouwWarpCheck(B, S, x, xsel, b, w, wsel, c, u, z, st, rc):
    """A warp's whole H1 call under li with a checkpoint a set (LI-CP-W): each of NL lanes' sets (`_call_sets`) absorbed by
    `PouwLiAbsorb_v2` from the lane's keys, each set's checkpoint the XOR of every lane's 8 chain states
    (`PouwXinFinal_v2`, the XOR of a state's words, then `PouwXor32_v2` across lanes, redux.sync.xor), equal to the leaf's
    word at the set's index."""
    return _warp_body(B, S.KS, S.NL, "li", x, xsel, b, w, wsel, c, u, z, st, rc)


@composite("PouwWarpCheck", 2, ["KS", "NL", "FOLD"], _warp2_signature, conformance="protocol-defined")
def PouwWarpCheckV2(B, S, x, xsel, b, w, wsel, c, u, z, st, rc):
    """`PouwWarpCheck_v1` for FOLD li, duo or fx1, with a full-state checkpoint a set (DUO-CP-W): each lane's sets
    absorbed by FOLD (`PouwLiAbsorb_v2`, `PouwDuoAbsorb_v2`, `PouwFx1Absorb_v2`) from its keys (li: y; duo, fx1: y, then
    z), and after set s, per part of the state, the XOR over lanes and chains equal to the leaf's word: li's y at s; duo's
    and fx1's y at 2 s and z at 2 s + 1."""
    return _warp_body(B, S.KS, S.NL, S.FOLD, x, xsel, b, w, wsel, c, u, z, st, rc)
