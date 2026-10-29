"""AddXor, checked: no single instruction of W1 or of sm_120's SASS computes (x + y) ^ c or (x ^ c) + y in one write.

    python3 hash_addxor.py [--configs N] [--mutations] [--verdict PATH]

The folds R80X and RDX (hash_rounds.py) XOR each 32-bit word with a public per-position constant c before the round's
add. That keeps two writes a word that a zero-data prover can't skip: the word itself, w = R + C (a row walk plus a
column walk), and the XOR. Two targets, each over 32-bit words:

- F1(x, y) = (x + y) ^ c: the word's XOR without forming the word (`decomp`), x a row-side and y a column-side value;
- F2(x, y) = (x ^ c) + y: the XOR fused into the round's add, x the word and y the state term it is added to.

**The model.** One write is one instruction writing one 32-bit register. Its operands are functions of x alone, of y
alone, or constants, each computed as often as wanted and free (shared over a row's or a column's cells, or already in
a register). Predicate writes are free, so the instruction may also read any number of predicates of (x, y), as carry
and select inputs, and may run predicated with a free fallback. A function of x alone and (x, y) both is a write. It
must be right on most cells, not all: each family's certificate is an invariant of small configurations (a 2x2
rectangle, a 3x3 grid, a 6x6 grid) that the family satisfies on every configuration whose cells it gets right. If a
fraction q of random configurations of k cells violates it, every member of the family is wrong on at least q/k of the
cells (union bound), less 1/|X| + 1/|Y| for a free predicated fallback, since F1 and F2 are injective in each argument.

**The families** (`FAMILIES`), each with its invariant, covering W1's instruction set (`Security/Definitions/Pouw/PearlC/
W1.lean`) and every sm_120 SASS opcode that writes a 32-bit register (`OPCODES`):
- add: s1 f(x) + s2 g(y) + k + e, e in [0, 2] from two carry-in predicates. The mixed second difference is in [-4, 4].
- mul: f(x) g(y) + h + e. The 2x2 matrix of mixed differences over a 3x3 grid is rank 1 mod 2^32 up to [-2, 2].
- hi: hi32(f(x) g(y) + h) + h' + e. The same over the reals, to within the floors' error.
- bit: a 3-input LUT of bits at one position. At some bit, some 3x3 grid has 3 distinct rows and 3 distinct columns.
- eq: a word of equality bits (MATCH). Two distinct rows of a bit share no 1, nor do their complements.
- fp: FADD, FMUL, FFMA, FMNMX and the half and bf16 pairs. Monotone in one side's value at each value of the other.
- dot: IDP4A, IDP.2A, VABSDIFF4's sum. The mixed differences are bounded by 2^26 and of rank at most 4 over Q.
- pick: one side's value or the other's, by a free predicate (SEL, FSEL, IMNMX, a predicated move). Right on at most
  one cell a row and one a column, since F is injective in each argument.
- shift: a y-selected shift or field of x (SHF with a register shift, SHL, SHR, SGXT, BMSK, PRMT with a register
  selector), and its additive y part. Two columns with the same selector have mixed differences 0.

**Beyond one write** (part T): the amortized routes. The add, mul, hi and dot invariants read only mixed differences, so
they hold for F minus any functions of x alone and of y alone: they also refute Y + Z with Y one-sided and Z one
instruction (the round's IADD3 adding Y and Z for free). For Z a LOP3, both LUT orientations are refuted exhaustively at
8 bits, which the low byte of any width inherits. A P2R'd correction of up to 8 predicate bits in one byte is refuted by
the mixed differences. The ballot route (VOTE.ANY of 32 lanes' bits) and a table lookup at a summed address are
priced, not refuted (part T's last checks).

Constants: c runs over the 160 halves of hash_rounds.xc(t). Positive controls (c = 0, 2^31, ~0, 2^31 - 1, which make
the XOR an add) must be found single-instruction by the add family, and are.

Stdlib, numpy and exact_kit.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_kit.checks import Checks
from hash_rounds import xc

M32 = (1 << 32) - 1
TWO32 = 1 << 32
CONSTS = sorted({xc(t) >> s & M32 for t in range(80) for s in (0, 32)})
BAD = (0, 1 << 31, M32, (1 << 31) - 1)
SPAN = 1 << 20

# Every SASS opcode of sm_120 that writes a general or uniform 32-bit register from data, by family. Opcodes that write
# no register (stores, branches, barriers, ISETP, PLOP3 and the other predicate writers, which are free), that write
# two or more registers (IMAD.WIDE, IADD.64, DADD and the other 64-bit forms, MMA and HMMA, LDSM), or that read one
# data operand (MOV, IABS, POPC, FLO, BREV, I2F, F2I, I2I, F2F, FRND, MUFU, S2R, CS2R, SHFL's moved value, R2UR) are
# listed with their reason. The uniform datapath's forms (U*) are the same functions on a warp's one value.
OPCODES = {
    "add": "IADD3, IADD3.X (two carry-ins), UIADD3(.X), VIADD, IADD32I, LEA, LEA.X, LEA.HI, LEA.HI.X, ULEA(.HI), "
           "IMAD and IMAD.X with a one-sided product (IMAD.IADD, IMAD.SHL, IMAD.MOV), SHF and USHF with an immediate "
           "shift (funnels of the two sides are disjoint fields), PRMT and UPRMT with a constant selector, I2IP, "
           "F2FP and F2IP packs, P2R's merge (a field of predicates into one side's register), REDUX.SUM, "
           "VIADDMNMX's add (its min side is pick); W1: add, sub, iadd3, prmt (constant selector), cvt",
    "mul": "IMAD, IMAD.X, UIMAD, IMUL, SHF, SHL and USHF with an x-side register shift of y-side data (a power of two "
           "times y); W1: mul, shl",
    "hi": "IMAD.HI, IMAD.HI.X, UIMAD.HI, IMUL.HI, IMAD.WIDE's high half (and IMAD.WIDE writes two); W1: mulhi",
    "bit": "LOP3.LUT and ULOP3.LUT (every LUT; the predicate operand reaches only the predicate result), LOP, LOP32I, "
           "REDUX.AND/OR/XOR; W1: and, or, xor, lop3",
    "eq": "MATCH.ANY, MATCH.ALL (lane masks of equalities)",
    "fp": "FADD, FMUL, FFMA, FMNMX, FADD32I, FMUL32I, FFMA32I, FSWZADD, HADD2, HMUL2, HFMA2, HMNMX2 and their "
          "BF16 forms (on the low half), FMNMX3; W1: fop add, mul, fma, min, max",
    "dot": "IDP.4A, IDP.2A, VABSDIFF4 (.ADD) and VABSDIFF",
    "pick": "SEL, FSEL, USEL, IMNMX, VIMNMX, VIMNMX3, UIMNMX, VIADDMNMX's min, REDUX.MIN/MAX, predicated moves, "
            "VOTE.ANY's fallback; W1: sel, min, max, lt, eq",
    "shift": "SHF, SHL, SHR and USHF with a y-side register shift, SGXT, USGXT, BMSK, UBMSK, PRMT and UPRMT with a "
             "register selector (the byte argument), VABSDIFF4 per byte; W1: shr, prmt (register selector)",
}
NOT_ONE_WRITE = {
    "no register written (free predicates, stores, control)": "ISETP, FSETP, HSETP2, DSETP, PLOP3, PSETP, R2P, "
        "UISETP, UPLOP3, ST*, RED*, ATOM* (no atomics in W1), BRA, BAR, MEMBAR, DEPBAR, EXIT, NOP, YIELD",
    "two or more registers written": "IMAD.WIDE, IMAD.WIDE.U32, IADD.64, DADD, DMUL, DFMA, HMMA, IMMA, QMMA, OMMA, "
        "UTC*MMA, LDSM, LDG/LDS .64/.128, SHF.*.W pairs; W1: mma, sha",
    "one data operand": "MOV, UMOV, MOV32I, IABS, POPC, UPOPC, FLO, UFLO, BREV, UBREV, I2F, F2I, I2I, F2F, FRND, "
        "MUFU, S2R, S2UR, CS2R, R2UR, SHFL (moves a lane's value), LDC; W1: mov, cvt of one side",
    "loads at one register's address": "LD, LDG, LDS, LDL, LDC, TEX, TLD; W1: ld (address M r + off)",
    "cross-lane (part T)": "VOTE.ANY/ALL, VOTEU, REDUX: priced in part T",
}


def f1(x, y, c):
    return ((x + y) & M32) ^ c


def f2(x, y, c):
    return ((x ^ c) + y) & M32


def sample(rng, dom, shape):
    if dom == "full":
        return rng.integers(0, TWO32, shape, dtype=np.int64)
    if dom == "span":
        return (rng.integers(-SPAN // 2, SPAN // 2, shape, dtype=np.int64)) & M32
    base = int(rng.integers(0, TWO32))
    return (base + rng.integers(0, SPAN, shape, dtype=np.int64)) & M32


def grid(rng, dom, n, a, b, fn, c):
    """n configurations of a rows x b columns: F at (x_i, y_j), rows and columns distinct."""
    xs = sample(rng, dom, (n, a))
    ys = sample(rng, dom, (n, b))
    ok = np.ones(n, bool)
    for i, j in itertools.combinations(range(a), 2):
        ok &= xs[:, i] != xs[:, j]
    for i, j in itertools.combinations(range(b), 2):
        ok &= ys[:, i] != ys[:, j]
    v = fn(xs[:, :, None].astype(np.uint64), ys[:, None, :].astype(np.uint64), np.uint64(c)).astype(np.int64)
    return v[ok]


def signed(d):
    d = d & M32
    return np.where(d >= 1 << 31, d - TWO32, d)


def mixed(v, i, j, i0=0, j0=0):
    return v[:, i, j] - v[:, i0, j] - v[:, i, j0] + v[:, i0, j0]


# ---------------------------------------------------------------- the invariants; each returns violations per config

def v_add(v):
    return np.abs(signed(mixed(v, 1, 1))) > 4


def v_add_p2r(v):
    """IADD3 of a one-sided value, the other side merged with up to 8 predicate bits in byte b (P2R), and two
    carry-ins: d = 2^(8b) m + e with |m| <= 510, |e| <= 4."""
    d = mixed(v, 1, 1)
    fits = np.zeros(len(v), bool)
    for e in range(-4, 5):
        t = signed(d - e)
        for b in range(4):
            fits |= (t % (1 << 8 * b) == 0) & (np.abs(t >> 8 * b) <= 510)
    return ~fits


def v_mul(v):
    e = np.arange(-2, 3)

    def prods(p, q):
        a = ((mixed(v, *p)[:, None] - e[None, :]) & M32).astype(np.uint64)
        b = ((mixed(v, *q)[:, None] - e[None, :]) & M32).astype(np.uint64)
        return ((a[:, :, None] * b[:, None, :]) & np.uint64(M32)).reshape(len(v), -1)

    lhs, rhs = prods((1, 1), (2, 2)), prods((1, 2), (2, 1))
    return ~np.any(lhs[:, :, None] == rhs[:, None, :], axis=(1, 2))


def v_hi(v):
    """D_ij = lam_i mu_j + err, lam in (-1, 1), |err| < 6 (four floors, an addend's carry, a carry-in): for every lift of
    the four entries to (-2^32, 2^32), |det D| exceeds what the errors allow."""
    d = [[(mixed(v, i, j) & M32).astype(np.float64) for j in (1, 2)] for i in (1, 2)]
    fits = np.zeros(len(v), bool)
    for lift in itertools.product((0.0, -float(TWO32)), repeat=4):
        a, b, c, g = d[0][0] + lift[0], d[1][1] + lift[1], d[0][1] + lift[2], d[1][0] + lift[3]
        det = a * b - c * g
        tol = 6 * (np.abs(a) + np.abs(b) + np.abs(c) + np.abs(g) + 24) + 72
        fits |= np.abs(det) <= tol
    return ~fits


def _distinct3(m):  # m: (n, 3) patterns
    return (m[:, 0] != m[:, 1]) & (m[:, 0] != m[:, 2]) & (m[:, 1] != m[:, 2])


def v_bit(v):
    out = np.zeros(len(v), bool)
    for i in range(32):
        bits = (v >> i) & 1
        rows = bits[:, :, 0] * 4 + bits[:, :, 1] * 2 + bits[:, :, 2]
        cols = bits[:, 0, :] * 4 + bits[:, 1, :] * 2 + bits[:, 2, :]
        out |= _distinct3(rows) & _distinct3(cols)
    return out


def v_eq(v):
    """Rows are bits of a 2x3 grid; an equality matrix's distinct rows have disjoint supports, as do its complement's."""
    out = np.zeros(len(v), bool)
    for i in range(32):
        bits = (v >> i) & 1
        r0, r1 = bits[:, 0, :], bits[:, 1, :]
        differ = np.any(r0 != r1, axis=1)
        out |= differ & np.any(r0 & r1, axis=1) & np.any((1 - r0) & (1 - r1), axis=1)
    return out


FP_VIEWS = {"fp32": (32, 8), "fp16": (16, 5), "bf16": (16, 8)}  # (bits read from the low end, exponent bits)


def fp_key(w, view):
    """The real order of an IEEE word as an integer key (infinities at the ends), and its NaN cells, which are wildcards
    because a NaN result is producible."""
    bits, ebits = FP_VIEWS[view]
    w = w & ((1 << bits) - 1)
    sign = w >> (bits - 1)
    mag = w & ((1 << (bits - 1)) - 1)
    nan = mag > (((1 << ebits) - 1) << (bits - 1 - ebits))
    return np.where(sign == 1, -mag, mag), nan


def _orders_clash(k, nan, axis):
    """Two rows (axis 1) or columns (axis 2) of a 3x3 grid whose strict orders agree on one pair and disagree on another."""
    out = np.zeros(len(k), bool)
    pairs = list(itertools.combinations(range(3), 2))
    for a, b in pairs:
        same = np.zeros(len(k), bool)
        opp = np.zeros(len(k), bool)
        for p, q in pairs:
            if axis == 1:
                s1 = np.sign(k[:, a, q] - k[:, a, p])
                s2 = np.sign(k[:, b, q] - k[:, b, p])
                bad = nan[:, a, q] | nan[:, a, p] | nan[:, b, q] | nan[:, b, p]
            else:
                s1 = np.sign(k[:, q, a] - k[:, p, a])
                s2 = np.sign(k[:, q, b] - k[:, p, b])
                bad = nan[:, q, a] | nan[:, p, a] | nan[:, q, b] | nan[:, p, b]
            same |= (s1 * s2 == 1) & ~bad
            opp |= (s1 * s2 == -1) & ~bad
        out |= same & opp
    return out


def v_fp(v):
    out = np.ones(len(v), bool)
    for view in FP_VIEWS:  # FP32 on the word; the pairs' FP16 and BF16 on the low half
        k, nan = fp_key(v, view)
        out &= _orders_clash(k, nan, 1) & _orders_clash(k, nan, 2)
    return out


def _rank_p(m, p=2_147_483_647):
    m = [[int(x) % p for x in row] for row in m]
    r = 0
    for col in range(len(m[0])):
        piv = next((i for i in range(r, len(m)) if m[i][col]), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        inv = pow(m[r][col], p - 2, p)
        for i in range(len(m)):
            if i != r and m[i][col]:
                f = m[i][col] * inv % p
                m[i] = [(a - f * b) % p for a, b in zip(m[i], m[r])]
        r += 1
    return r


def v_dot(v):
    d = signed(np.stack([np.stack([mixed(v, i, j) for j in range(1, 6)], 1) for i in range(1, 6)], 1))
    big = np.any(np.abs(d) > 1 << 26, axis=(1, 2))
    return np.array([b or _rank_p(m) == 5 for b, m in zip(big, d)])


def v_pick(v):
    """A 3x3 grid of distinct rows and columns: F equals a row's value at most once a row and a column's once a column."""
    return np.ones(len(v), bool)


FAMILIES = {  # name: (rows, cols, invariant)
    "add": (2, 2, v_add), "mul": (3, 3, v_mul), "hi": (3, 3, v_hi), "bit": (3, 3, v_bit), "eq": (2, 3, v_eq),
    "fp": (3, 3, v_fp), "dot": (6, 6, v_dot), "pick": (3, 3, v_pick),
}
N_CFG = {"dot": 300}


def family_bounds(rng, fn, c, dom, n):
    """Per family: the fraction q of violating configurations and the bound q/k on its wrong cells."""
    out = {}
    for name, (a, b, inv) in FAMILIES.items():
        v = grid(rng, dom, N_CFG.get(name, n), a, b, fn, c)
        q = float(np.mean(inv(v)))
        out[name] = (q, q / (a * b))
    return out


def shift_bound(rng, fn, c, dom, n):
    """The shift family: two columns sharing one of at most 64 selectors have every mixed difference 0. On 8 rows and
    2 columns with p0 the fraction whose 7 mixed differences are all 0, a member is wrong on at least (1/64 - p0)/16 of
    the cells."""
    v = grid(rng, dom, n, 8, 2, fn, c)
    p0 = float(np.mean(np.all(np.stack([mixed(v, i, 1) & M32 for i in range(1, 8)]) == 0, axis=0)))
    return p0, max(0.0, (1 / 64 - p0) / 16)


# ---------------------------------------------------------------- F2: operands are the word a, the term b, constants

def transforms(a):
    """One operand's forms inside one instruction: itself, negated, complemented, shifted or rotated by an immediate
    (LEA, LEA.HI, SHF, IMAD.SHL), or absent."""
    out = {"0": 0 * a, "a": a, "-a": (-a) & M32, "~a": a ^ M32}
    for s in range(1, 32):
        out[f"a<<{s}"] = (a << s) & M32
        out[f"-(a<<{s})"] = (-(a << s)) & M32
        out[f"a>>{s}"] = a >> s
        out[f"a>>>{s}"] = ((a ^ 0x80000000) >> s) - (0x80000000 >> s) & M32
        out[f"rot{s}"] = ((a << s) | (a >> (32 - s))) & M32
    return out


def best_window(d, width=3):
    """The largest fraction of d (mod 2^32) in `width` consecutive values: an instruction g(a) + b + k + e, e in
    [0, 2], agrees with (a ^ c) + b at most this often."""
    u, cnt = np.unique(d & M32, return_counts=True)
    tot = cnt.astype(np.int64)
    for k in range(1, width):
        t = (u + k) & M32
        idx = np.minimum(np.searchsorted(u, t), len(u) - 1)
        tot = tot + np.where(u[idx] == t, cnt[idx], 0)
    return float(tot.max()) / len(d)


def f2_bounds(rng, c, dom, n):
    """F2's one-write instructions over (a, b, constants), each with the largest fraction of cells it can get right."""
    a = sample(rng, dom, 20000).astype(np.int64)
    t = (a ^ c).astype(np.int64)
    out = {}
    out["add (IADD3, LEA, LEA.HI, SHF of a with b)"] = max(best_window(t - g) for g in transforms(a).values())
    ks = {((((u + 1) & M32) ^ c) - (u ^ c) - e) & M32 for e in range(-2, 3)
          for u in sample(rng, dom, 50).tolist()}
    out["IMAD(a, k, b), any k"] = max(best_window(t - (k * a & M32)) for k in ks)
    x = sample(rng, dom, (n, 2)).astype(np.uint64)
    y = rng.integers(0, TWO32, (n, 2)).astype(np.uint64)

    def two_sided(g):
        v = g(x[:, :, None], y[:, None, :]).astype(np.int64)
        return float(np.mean(np.abs(signed(mixed(v, 1, 1))) > 6)) / 4

    def idp(p, q, signed_bytes):
        s = np.zeros(np.broadcast_shapes(p.shape, q.shape), np.int64)
        for k in range(4):
            u = ((p >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.int64)
            w = ((q >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.int64)
            if signed_bytes:
                u, w = np.where(u > 127, u - 256, u), np.where(w > 127, w - 256, w)
            s += u * w
        return s & M32

    def absdiff(p, q, packed):
        s = np.zeros(np.broadcast_shapes(p.shape, q.shape), np.int64)
        for k in range(4):
            u = ((p >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.int64)
            w = ((q >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.int64)
            s += np.abs(u - w) << (8 * k if packed else 0)
        return s

    out["IMAD(a, b, k)"] = 1 - two_sided(lambda p, q: (p * q) & np.uint64(M32))
    out["IMAD.HI(a, b, k)"] = 1 - two_sided(lambda p, q: (p.astype(object) * q.astype(object) >> 32).astype(np.int64))
    out["IDP.4A(a, b, k)"] = 1 - min(two_sided(lambda p, q: idp(p, q, s)) for s in (False, True))
    out["VABSDIFF4(a, b)"] = 1 - min(two_sided(lambda p, q: absdiff(p, q, s)) for s in (False, True))
    # IDP(a, k, b): a step of 256^j in a moves (a ^ c) by 256^j up to sign; one byte's product moves at most 255.
    j = rng.integers(1, 4, 20000)
    a2 = sample(rng, dom, 20000).astype(np.int64)
    step = signed((((a2 + (1 << (8 * j))) & M32) ^ c) - (a2 ^ c))
    out["IDP.4A(a, k, b)"] = 1 - float(np.mean(np.abs(step) > 300)) / 2
    # FP ops reading a: at a fixed b, monotone in a's value or its magnitude.
    v_in = sample(rng, dom, (n, 3)).astype(np.int64)
    b = rng.integers(0, TWO32, (n, 1)).astype(np.int64)
    v_out = ((v_in ^ c) + b) & M32
    clash = np.ones(n, bool)
    for view in FP_VIEWS:
        ki, ni = fp_key(v_in, view)
        ko, no = fp_key(v_out, view)
        for key in (ki, np.abs(ki)):
            same = np.zeros(n, bool)
            opp = np.zeros(n, bool)
            for p, q in itertools.combinations(range(3), 2):
                s = np.sign(key[:, q] - key[:, p]) * np.sign(ko[:, q] - ko[:, p])
                ok = ~(ni[:, p] | ni[:, q] | no[:, p] | no[:, q])
                same |= (s == 1) & ok
                opp |= (s == -1) & ok
            clash &= same & opp
    out["FP ops reading a (FADD, FMUL, FFMA, FMNMX, the pairs)"] = 1 - float(np.mean(clash)) / 3
    return out


def prmt_byte_bound():
    """PRMT, any selector source: at a fixed y, byte 0 is one of at most 16 functions of x (a byte of an x-side
    register, or its sign fill) or constant in x. F's byte 0 at the 256 values of y's low byte are 256 functions of x
    that never agree at any x (a Latin square), so the average agreement over y is at most 17/256."""
    return 17 / 256


def latin(fn, c):
    lo = np.arange(256, dtype=np.uint64)
    t = fn(lo[:, None], lo[None, :], np.uint64(c & 0xFF)) & np.uint64(0xFF)
    return all(len(set(r.tolist())) == 256 for r in t) and all(len(set(r.tolist())) == 256 for r in t.T)


# ---------------------------------------------------------------- part T: Y + LOP3 exhaustively at 8 bits

def lop3_plus_one_sided(fn, c):
    """Is F = Y(x) + LOP3(...) on 8-bit words, for some x-side Y (F1 only: for F2 a function of x is a write)? With
    v_x = Y(x), Z(x, .) = F(x, .) - v_x is a bijection, so every bit of Z(x, .) is balanced, never constant.
    - LUT reading two x-side operands and one y-side: a bit's columns take at most 2 patterns, so bit j of Z(x, y) is
      beta_j(y) ^ s_j(x), and Z = B(y) ^ S(x). Exhaustive over v_0 (S_0 = 0 absorbed into B) and each row's v_x.
    - LUT reading one x-side and two y-side: a bit's rows take at most 2 patterns. At some bit j no two patterns hit
      every row's 256 choices {bit j of F(x, .) - v : v} (a hitting set of size 2, exhaustive).
    Returns which orientations admit a solution."""
    n = 256
    lo = np.arange(n, dtype=np.int64)
    F = fn(lo[:, None].astype(np.uint64), lo[None, :].astype(np.uint64), np.uint64(c & 0xFF)).astype(np.int64) & 0xFF
    Z = (F[:, None, :] - lo[None, :, None]) & 0xFF  # Z[x, v, y] = F(x, y) - v
    alive = np.arange(n)
    for x in range(1, n):
        t = Z[x][None, :, :] ^ Z[0, alive][:, None, :]
        alive = alive[np.any(np.all(t == t[:, :, :1], axis=2), axis=1)]
        if not len(alive):
            break
    xxy = bool(len(alive))
    xyy = all(_hit2(((Z >> j) & 1).astype(np.uint8)) for j in range(8))
    return xxy, xyy


def _hit2(bits):
    """bits[x, v, y]: do two row patterns hit every row's choices?"""
    n = bits.shape[0]
    keys = np.packbits(bits, axis=2).reshape(n * n, -1)
    uniq, ids = np.unique(keys, axis=0, return_inverse=True)
    ids = ids.reshape(n, n)
    hit = np.zeros((len(uniq), n), bool)  # hit[p, x]: pattern p is one of row x's choices
    hit[ids, np.arange(n)[:, None]] = True
    words = np.packbits(hit, axis=1).view(np.uint64)  # (patterns, 4)
    for a in np.flatnonzero(hit[:, 0]):
        need = ~words[a]
        if np.any(np.all((words & need[None, :]) == need[None, :], axis=1)):
            return True
    return False


# ---------------------------------------------------------------- main

def main() -> int:
    checks = Checks()
    n = 4000
    if "--configs" in checks.args:
        n = int(checks.args[checks.args.index("--configs") + 1])
    rng = np.random.default_rng(20261029)

    checks.check("constants: the 160 halves of xc(t) avoid the four values that make the XOR an add or a negation "
                 "(0, 2^31, ~0, 2^31 - 1)", len(CONSTS) == 160 and not set(CONSTS) & set(BAD), f"{len(CONSTS)} halves")

    doms = ("full", "span", "window")
    worst1 = {}
    for dom in doms:
        for c in CONSTS:
            fb = family_bounds(rng, f1, c, dom, n)
            fb["shift"] = shift_bound(rng, f1, c, dom, n)
            for fam, (q, eps) in fb.items():
                worst1[(dom, fam)] = min(worst1.get((dom, fam), 1.0), eps)
    fams1 = [f for f in list(FAMILIES) + ["shift"] if f != "pick"]
    every1 = all(worst1[(d, f)] > 0 for d in doms for f in fams1)
    checks.check("one write, F1 = (x + y) ^ c with x, y any row-side and column-side values: for every constant and "
                 "each domain (full 32-bit; the zero-data span of 2^20 around 0; a 2^20 window at a random base), every "
                 "family's invariant fails on a positive fraction of configurations, so each family is wrong on at "
                 "least the least fraction shown of cells", every1,
                 "; ".join(f"{d}: " + ", ".join(f"{f} {worst1[(d, f)]:.3g}" for f in fams1) for d in doms))
    worst2 = {}
    for dom in doms:
        for c in CONSTS:
            fb = {f: e for f, (q, e) in family_bounds(rng, f2, c, dom, n).items() if f in ("bit", "eq")}
            for name, right in f2_bounds(rng, c, dom, n).items():
                fb[name] = 1 - right
            for fam, eps in fb.items():
                worst2[(dom, fam)] = min(worst2.get((dom, fam), 1.0), eps)
    fams2 = sorted({f for _, f in worst2})
    every2 = all(worst2[(d, f)] > 0 for d in doms for f in fams2)
    checks.check("one write, F2 = (a ^ c) + b with a the word and b the term it is added to, each operand a, b or a "
                 "constant: for every constant and domain of a (b full 32-bit), every instruction is wrong on at least "
                 "the least fraction shown of cells", every2,
                 "; ".join(f"{d}: " + ", ".join(f"{f} {worst2[(d, f)]:.3g}" for f in fams2) for d in doms))
    lat = all(latin(f1, c) and latin(f2, c) for c in CONSTS)
    worst_pick = 2 / SPAN
    checks.check("one write, pick and PRMT: F1 and F2 are Latin squares on their low bytes for every constant (so a "
                 "pick is right on at most 1/|X| + 1/|Y| of cells and PRMT, any selector source, on at most 17/256)",
                 lat, f"pick <= {worst_pick:.2g} at a 2^20 span; PRMT <= {prmt_byte_bound():.4f}")
    fams_named = set(OPCODES)
    checks.check("coverage: every family in the opcode table has an invariant (pick, shift and PRMT by count)",
                 fams_named <= set(FAMILIES) | {"shift"}, sorted(fams_named))

    # positive controls: the four bad constants are single additions, and the add invariant must not refute them.
    ctl = []
    for c in BAD:
        for fn in (f1, f2):
            v = grid(rng, "full", 2000, 2, 2, fn, c)
            ctl.append(float(np.mean(v_add(v))))
    checks.check("controls: with c in {0, 2^31, ~0, 2^31 - 1} the XOR is an add or a negation, and the add invariant "
                 "never fires (it finds single instructions when they exist)", max(ctl) == 0.0,
                 f"violation fractions {ctl}")
    # control: the honest two-instruction program (IADD3 then LOP3) passes every family invariant applied to its first
    # write's output, w = x + y, which is an add.
    v = grid(rng, "span", 2000, 2, 2, lambda x, y, c: (x + y) & M32, 0)
    checks.check("controls: the honest first write, w = x + y, satisfies the add invariant everywhere",
                 float(np.mean(v_add(v))) == 0.0)

    # part T: amortized routes.
    p2r = []
    for dom in ("full", "span"):
        for c in CONSTS:
            p2r.append(float(np.mean(v_add_p2r(grid(rng, dom, 2000, 2, 2, f1, c)))) / 4)
    checks.check("T: F1 as Y + Z with Z an IADD3 reading one side's register merged with a P2R'd byte of up to 8 "
                 "predicate bits and two carry-ins: refuted for every constant, wrong on at least the fraction shown",
                 min(p2r) > 0, f"min {min(p2r):.3g}")
    lows = sorted({c & 0xFF for c in CONSTS})
    lops = [lop3_plus_one_sided(f1, c) for c in lows]
    checks.check(f"T: F1 as Y + Z with Z a LOP3 (any LUT, either orientation) and Y one-sided: no solution at 8 bits "
                 f"for any of the {len(lows)} distinct low bytes of the constants, so none at any width",
                 not any(a or b for a, b in lops), f"{sum(a or b for a, b in lops)} of {len(lops)} admit one")
    ctl_l = [lop3_plus_one_sided(f1, c) for c in (0, 0x80, 0xFF)]
    checks.check("T controls: at c in {0, 0x80, 0xFF} (an add) the search finds Y + LOP3 (Y = x, Z = y or ~y)",
                 all(a or b for a, b in ctl_l), str(ctl_l))
    J, T = 64, 16
    ballot = 1 + 32 / J + 32 / T
    checks.check("T: the ballot route (lane k makes bit k of F as free predicates, VOTEU.ANY packs them) needs each "
                 "lane's own shifted copy of the row and column values, written per lane: 1 + 32/J + 32/T writes a word "
                 "at the served layout (J 64 columns a row, T 16 rows a class), above honest's 2 (argued, not checked)",
                 ballot > 2, f"{ballot:.2f}")
    if checks.mutations:
        with checks.mutant("the constants include 2^31 (XOR as an add)"):
            checks.check("constants avoid the bad values", not (set(CONSTS) | {1 << 31}) & set(BAD))
        with checks.mutant("the add invariant with carry-ins of up to 2^20"):
            v = grid(rng, "span", 2000, 2, 2, f1, CONSTS[0])
            checks.check("add refutes", float(np.mean(np.abs(signed(mixed(v, 1, 1))) > 1 << 22)) > 0)
        with checks.mutant("the target is the honest add, w = x + y"):
            v = grid(rng, "full", 2000, 2, 2, lambda x, y, c: (x + y) & M32, 0)
            checks.check("add refutes", float(np.mean(v_add(v))) > 0)
        with checks.mutant("the target is a product, x * y (mul must not refute it)"):
            v = grid(rng, "full", 2000, 3, 3, lambda x, y, c: (x * y) & M32, 0)
            checks.check("mul refutes", float(np.mean(v_mul(v))) > 0)
        with checks.mutant("the target is bitwise, x ^ y ^ c (bit must not refute it)"):
            v = grid(rng, "full", 2000, 3, 3, lambda x, y, c: x ^ y ^ c, np.uint64(CONSTS[0]))
            checks.check("bit refutes", float(np.mean(v_bit(v))) > 0)
        with checks.mutant("the target is a LOP3 plus a one-sided Y, (x ^ y ^ c) + 3x"):
            xxy, xyy = lop3_plus_one_sided(lambda x, y, c: ((x ^ y ^ c) + 3 * x) & M32, CONSTS[0])
            checks.check("no Y + LOP3", not (xxy or xyy))
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
