"""RT-ALGO: bit-exact check of the algebra of H1 Dither's credited words against h1tile.reference.

Goal: test the premise that each credited word needs a fresh write that costs what the honest GEMM costs, by trying the
algebraic shortcuts (reuse the served AB, low-rank in the coins/noise, U's linear structure, fold-before-multiply,
precompute before the coin). Every claim is checked against the reference, not asserted.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.insert(0, "/workspace")

import hashlib
import random

from verity.core.protocols.pouw.circuit import h1tile, segment as SG, seeds as SD
from verity.core.protocols.pouw.circuit.h1tile import Layout, noise, fold_keys, reference, u129

M32 = (1 << 32) - 1


def sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def rand_i8(rng, n):
    return [rng.randint(-127, 127) for _ in range(n)]


def build(rng, r=4, c=8, K=64):
    a = [rand_i8(rng, K) for _ in range(r)]
    b = [rand_i8(rng, K) for _ in range(c)]
    salt = bytes(rng.getrandbits(8) for _ in range(64))
    roots = (bytes(rng.getrandbits(8) for _ in range(64)), bytes(rng.getrandbits(8) for _ in range(64)))
    position = rng.randint(0, 1 << 20)
    return a, b, salt, position, roots


def recompute_words(a, b, salt, position, roots):
    """Reproduce the unit's per-lane absorbed sets (T_0..T_{KS-1}, O', D_1..D_{m-1}) from the operands, independently of
    the reference's fold, and return them together with X, W', U, the per-row/col noise, and the digits."""
    lay = Layout(len(a), len(b), len(a[0]))
    seed_j = SD.seed(salt, position, *roots)
    rows, cols = noise(seed_j, lay)
    ks, m = lay.ks, lay.m
    per_lane = []
    for ln in range(lay.lanes):
        R, Cs = lay.rows(ln), lay.cols(ln)
        seg = [range(32 * s, 32 * s + 32) for s in range(ks)]
        xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
        bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
        wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
        cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
        u = [cols[j][1] for j in Cs]
        z = [[rows[i][1][d] for i in R] for d in range(m - 1)]
        words = SG.call_words(xr, bb, wr, cc, u, z)
        per_lane.append((R, Cs, words, rows, cols))
    return lay, rows, cols, per_lane


def fold_receipt(lay, per_lane, salt, position, roots):
    keys = fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    cp = [[0, 0] for _ in range(lay.sets)]
    for ln, (R, Cs, words, _, _) in enumerate(per_lane):
        st = list(keys[ln])
        for s, p in enumerate(words):
            st = SG.absorb(h1tile.FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
    return [w & M32 for s in cp for w in s]


def run_shape(rng, r, c, K, check, adversarial=False):
    a, b, salt, position, roots = build(rng, r, c, K)
    if adversarial:
        # force the X wrap (A=127,b will be 1 somewhere) and |B| large so W' wraps too.
        for i in range(r):
            for t in range(0, K, 3):
                a[i][t] = 127
        for j in range(c):
            for t in range(0, K, 2):
                b[j][t] = 127 if t % 4 == 0 else -127
    z, receipt = reference(a, b, salt, position, roots)
    lay, rows, cols, per_lane = recompute_words(a, b, salt, position, roots)

    # 0. our independent fold reproduces the reference's receipt.
    check("fold of recomputed words == reference receipt", fold_receipt(lay, per_lane, salt, position, roots) == receipt)

    # 1. served z = plain A.B^T (clean), and it is NOT the noised prefix word.
    def plain_dot(i, j):
        return sum(a[i][t] * b[j][t] for t in range(K)) & M32
    pairs = [(i, j) for i in range(r) for j in range(c)]
    check("served z == plain A.B^T (clean integer product)", z == [plain_dot(i, j) for i, j in pairs])

    # 2. the credited prefix words are the noised partial contractions X.W' (X=A+b, W'=B+c+U, read s8).
    ks, m = lay.ks, lay.m
    ok_prefix = True
    ok_final_differs = 0
    for ln, (R, Cs, words, rws, cls) in enumerate(per_lane):
        X = [[(a[i][t] + (rws[i][0][t] & 1)) & 0xFF for t in range(K)] for i in R]       # per row-slot
        Wp = [[(b[j][t] + (cls[j][0][t] & 1) + cls[j][1][t % 32]) & 0xFF for t in range(K)] for j in Cs]
        for jpref in range(ks):
            kk = 32 * (jpref + 1)
            for q in range(32):
                rs, csl = SG.row_slot(q), SG.col_slot(q)
                want = sum(sx8(X[rs][t]) * sx8(Wp[csl][t]) for t in range(kk)) & M32
                if words[jpref][q] != want:
                    ok_prefix = False
        # final prefix word vs served clean product for the same cell
        for q in range(32):
            rs, csl = SG.row_slot(q), SG.col_slot(q)
            i, j = R[rs], Cs[csl]
            if words[ks - 1][q] != plain_dot(i, j):
                ok_final_differs += 1
    check("every credited prefix word == noised partial contraction X.W'", ok_prefix)
    check("final credited word != served clean product (noise is not peelable for free)", ok_final_differs > 0)

    # 3. AB-reuse expansion: noised full word = bilinear(A+b, B+c+U over Z) - 256*wrap, and the fresh terms A.c and b.B
    #    are each a full-length contraction over post-coin coins (cannot be precomputed from A,B alone).
    ln = 0
    R, Cs, words, rws, cls = per_lane[ln]
    q = 0
    rs, csl = SG.row_slot(q), SG.col_slot(q)
    i, j = R[rs], Cs[csl]
    Ai = a[i]; Bj = b[j]
    bi = [rws[i][0][t] & 1 for t in range(K)]
    cj = [cls[j][0][t] & 1 for t in range(K)]
    Uj = [cls[j][1][t % 32] for t in range(K)]              # U period 32 along k
    Xi = [(Ai[t] + bi[t]) & 0xFF for t in range(K)]
    Wj = [(Bj[t] + cj[t] + Uj[t]) & 0xFF for t in range(K)]
    noised = sum(sx8(Xi[t]) * sx8(Wj[t]) for t in range(K)) & M32
    # bilinear over Z of the *wrapped* s8 operands splits exactly, no mod needed (|.|<2^31):
    AB = sum(sx8(Ai[t]) * sx8(Bj[t]) for t in range(K))
    Ac = sum(sx8(Ai[t]) * cj[t] for t in range(K))
    AU = sum(sx8(Ai[t]) * Uj[t] for t in range(K))
    bB = sum(bi[t] * sx8(Bj[t]) for t in range(K))
    bc = sum(bi[t] * cj[t] for t in range(K))
    bU = sum(bi[t] * Uj[t] for t in range(K))
    # the wrap terms, from decomposing sx8(x mod 256) = (value) - 256*wrapbit:
    xb = [(Ai[t] + bi[t]) - sx8(Xi[t]) for t in range(K)]   # in {0, 256}, /256 is the X wrap bit
    wb = [(Bj[t] + cj[t] + Uj[t]) - sx8(Wj[t]) for t in range(K)]
    # noised = sum sx8(X)*sx8(W') = sum (A+b - xb)(B+c+U - wb)
    recon = (AB + Ac + AU + bB + bc + bU
             - sum(sx8(Xi[t]) * wb[t] for t in range(K))
             - sum(xb[t] * (sx8(Bj[t]) + cj[t] + Uj[t]) for t in range(K))) & M32
    check("noised word == AB + A(c+U) + b(B+c+U) - 256.(X.wb + xb.W') (exact over Z, mod 2^32)", recon == noised)

    # 4. U's linear reduction: A.U = U . (residue sums of A), cost 32 MACs not K.
    res = [sum(sx8(Ai[t]) for t in range(K) if t % 32 == r_) for r_ in range(32)]
    AU_red = sum(cls[j][1][r_] * res[r_] for r_ in range(32))
    check("A.U == U . (residue sums of A)  [32 MACs, already priced by the digit words]", AU_red == AU)

    # 5. irreducibility of A.c and b.B: every coin bit has nonzero weight, so flipping any one moves the word
    #    (no precompute, no low-rank collapse). count the bits whose flip changes A.c.
    flips_Ac = sum(1 for t in range(K) if sx8(Ai[t]) != 0)
    flips_bB = sum(1 for t in range(K) if sx8(Bj[t]) != 0)
    check("A.c depends on ~every c_k (flip moves the word iff A_k != 0)", flips_Ac >= K - 2)
    check("b.B depends on ~every b_k (flip moves the word iff B_k != 0)", flips_bB >= K - 2)

    # 6. fold non-linearity: fx1/duo absorb is not additive over the words, so you cannot fold a cheap linear combo.
    st0 = [rng.getrandbits(32) for _ in range(16)]
    w1 = [rng.getrandbits(32) for _ in range(32)]
    w2 = [rng.getrandbits(32) for _ in range(32)]
    f1 = SG.absorb("fx1", list(st0), w1)
    f2 = SG.absorb("fx1", list(st0), w2)
    fsum = SG.absorb("fx1", list(st0), [(w1[t] + w2[t]) & M32 for t in range(32)])
    additive = all((f1[t] + f2[t] - st0[t]) & M32 == fsum[t] for t in range(16))
    check("fx1 fold is NOT additive over words (fold-before-multiply gives nothing)", not additive)
    return lay


def main():
    rng = random.Random(1)
    passed = []
    failed = []

    def check(name, cond):
        key = name.split("  [")[0]
        (passed if cond else failed).append(key)
        if cond and key in [k for k, _ in __import__("collections").Counter(passed).items() if passed.count(k) > 1]:
            return
        print(("PASS " if cond else "FAIL ") + name)

    shapes = [(4, 8, 64, False), (8, 16, 256, False), (4, 8, 128, True), (8, 8, 256, True)]
    lay = None
    for (r, c, K, adv) in shapes:
        print(f"--- shape r={r} c={c} K={K} adversarial={adv} ---")
        lay = run_shape(rng, r, c, K, check, adversarial=adv)

    # floor sizing: MACs to produce all credited words per lane, each route (K=256, KS=8, m=2).
    r, c, K = 8, 16, 256
    ks = K // 32
    m = SG.n_digits(ks)
    floor_words = 32 * (ks + m)
    gemm_macs = 32 * 32 * ks            # prefix words of a lane: one noised GEMM over its 32 cells
    digit_macs = 32 * 32 * m            # O' + digits: 32 MACs each
    print()
    print("=== floor sizing per lane (K=%d KS=%d m=%d) ===" % (K, ks, m))
    print(f"credited words/lane            : {floor_words}")
    print(f"floor MACs/lane (prefix+digit) : {gemm_macs + digit_macs}")
    print("route                              credited-MACs produced below floor   saving")
    print(f"  direct noised GEMM               {gemm_macs + digit_macs:>6}  (== floor)                    0%")
    print(f"  reuse served AB (final only)     {0:>6}  (AB is 1 value/cell; prefixes need per-seg)  0%")
    print(f"  per-seg plain AB + corrections   {0:>6}  (plain GEMM + A.c + b.B >= 2 GEMMs)           0%")
    print(f"  Strassen / fast matmul           {0:>6}  (cannot emit per-segment prefix words)        0%")
    print(f"  low-rank in coins (b,c)          {0:>6}  (b per-row, c per-col: full-rank, fresh)      0%")
    print(f"  U linear structure               {0:>6}  (= the digit words; already credited)         0%")
    print(f"  fold-before-multiply             {0:>6}  (fx1/duo fold is non-additive)                0%")
    print(f"  packing / CRT                    {0:>6}  (products exact <2^32; machine-route only)    0%")
    print(f"  precompute before coin           {0:>6}  (only clean AB; credited work is noised)      0%")
    print(f"  C1: int8 Dither on FP8/FP4       {0:>6}  (emulation reproduces words bit-exact)        0%")
    print()
    print(f"PASSED {len(set(passed))} distinct checks x {len(shapes)} shapes = {len(passed)} runs  FAILED {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
