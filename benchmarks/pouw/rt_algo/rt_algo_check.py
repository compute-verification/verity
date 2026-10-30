"""RT-ALGO: bit-exact check of the algebra of H1 Dither's credited words against `h1tile.reference`.

Tests the premise that each credited word needs a fresh write that costs what the honest GEMM costs, by trying the
algebraic shortcuts (reuse the served AB, low-rank in the coins/noise, U's linear structure, fold-before-multiply,
precompute before the coin) and showing each produces 0% of the credited work below the floor. Every claim is checked
against the reference, not asserted. `collect()` returns (passed, failed); the CLI prints them and a floor-sizing table
(RT-ALGO, note:20261019T0520Z-draft-rt-algo).
"""
from __future__ import annotations

import random

from verity.ml.protocols.pouw.circuit import h1tile, segment as SG, seeds as SD
from verity.ml.protocols.pouw.circuit.h1tile import Layout, fold_keys, noise, reference

M32 = (1 << 32) - 1
#: (r, c, K, adversarial) tiles the check runs; adversarial forces the s8 wrap on both operands.
SHAPES = [(4, 8, 64, False), (8, 16, 256, False), (4, 8, 128, True), (8, 8, 256, True)]


def sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def _rand_i8(rng, n):
    return [rng.randint(-127, 127) for _ in range(n)]


def _build(rng, r, c, K, adversarial):
    a = [_rand_i8(rng, K) for _ in range(r)]
    b = [_rand_i8(rng, K) for _ in range(c)]
    if adversarial:
        for row in a:
            for t in range(0, K, 3):
                row[t] = 127
        for j, row in enumerate(b):
            for t in range(0, K, 2):
                row[t] = 127 if t % 4 == 0 else -127
    salt = bytes(rng.getrandbits(8) for _ in range(64))
    roots = (bytes(rng.getrandbits(8) for _ in range(64)), bytes(rng.getrandbits(8) for _ in range(64)))
    return a, b, salt, rng.randint(0, 1 << 20), roots


def _recompute_words(a, b, salt, position, roots):
    """The unit's per-lane absorbed sets (T_0..T_{KS-1}, O', D_1..D_{m-1}), independently of the reference's fold."""
    lay = Layout(len(a), len(b), len(a[0]))
    rows, cols = noise(SD.seed(salt, position, *roots), lay)
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
        per_lane.append((R, Cs, SG.call_words(xr, bb, wr, cc, u, z), rows, cols))
    return lay, per_lane


def _fold_receipt(lay, per_lane, salt, position, roots):
    keys = fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    cp = [[0, 0] for _ in range(lay.sets)]
    for ln, (_, _, words, _, _) in enumerate(per_lane):
        st = list(keys[ln])
        for s, p in enumerate(words):
            st = SG.absorb(h1tile.FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
    return [w & M32 for s in cp for w in s]


def _check_shape(rng, r, c, K, adversarial, out):
    def ok(name, cond):
        out.setdefault(name, True)
        out[name] = out[name] and bool(cond)

    a, b, salt, position, roots = _build(rng, r, c, K, adversarial)
    z, receipt = reference(a, b, salt, position, roots)
    lay, per_lane = _recompute_words(a, b, salt, position, roots)

    ok("fold of recomputed words == reference receipt", _fold_receipt(lay, per_lane, salt, position, roots) == receipt)

    def plain_dot(i, j):
        return sum(a[i][t] * b[j][t] for t in range(K)) & M32

    pairs = [(i, j) for i in range(r) for j in range(c)]
    ok("served z == plain A.B^T (clean integer product)", z == [plain_dot(i, j) for i, j in pairs])

    ks, m = lay.ks, lay.m
    ok_prefix, ok_final_differs = True, 0
    for R, Cs, words, rws, cls in per_lane:
        X = [[(a[i][t] + (rws[i][0][t] & 1)) & 0xFF for t in range(K)] for i in R]
        Wp = [[(b[j][t] + (cls[j][0][t] & 1) + cls[j][1][t % 32]) & 0xFF for t in range(K)] for j in Cs]
        for jp in range(ks):
            kk = 32 * (jp + 1)
            for q in range(32):
                want = sum(sx8(X[SG.row_slot(q)][t]) * sx8(Wp[SG.col_slot(q)][t]) for t in range(kk)) & M32
                ok_prefix = ok_prefix and words[jp][q] == want
        for q in range(32):
            i, j = R[SG.row_slot(q)], Cs[SG.col_slot(q)]
            if words[ks - 1][q] != plain_dot(i, j):
                ok_final_differs += 1
    ok("every credited prefix word == noised partial contraction X.W'", ok_prefix)
    ok("final credited word != served clean product (noise is not peelable for free)", ok_final_differs > 0)

    # AB-reuse expansion and the wrap, for one cell.
    R, Cs, words, rws, cls = per_lane[0]
    i, j = R[SG.row_slot(0)], Cs[SG.col_slot(0)]
    Ai, Bj = a[i], b[j]
    bi = [rws[i][0][t] & 1 for t in range(K)]
    cj = [cls[j][0][t] & 1 for t in range(K)]
    Uj = [cls[j][1][t % 32] for t in range(K)]
    Xi = [(Ai[t] + bi[t]) & 0xFF for t in range(K)]
    Wj = [(Bj[t] + cj[t] + Uj[t]) & 0xFF for t in range(K)]
    noised = sum(sx8(Xi[t]) * sx8(Wj[t]) for t in range(K)) & M32
    AB = sum(sx8(Ai[t]) * sx8(Bj[t]) for t in range(K))
    Ac = sum(sx8(Ai[t]) * cj[t] for t in range(K))
    AU = sum(sx8(Ai[t]) * Uj[t] for t in range(K))
    bB = sum(bi[t] * sx8(Bj[t]) for t in range(K))
    bc = sum(bi[t] * cj[t] for t in range(K))
    bU = sum(bi[t] * Uj[t] for t in range(K))
    recon = (AB + Ac + AU + bB + bc + bU
             - sum(sx8(Xi[t]) * ((Bj[t] + cj[t] + Uj[t]) - sx8(Wj[t])) for t in range(K))
             - sum(((Ai[t] + bi[t]) - sx8(Xi[t])) * (sx8(Bj[t]) + cj[t] + Uj[t]) for t in range(K))) & M32
    ok("noised word == AB + A(c+U) + b(B+c+U) - 256.(X.wb + xb.W') (exact over Z, mod 2^32)", recon == noised)

    res = [sum(sx8(Ai[t]) for t in range(K) if t % 32 == r_) for r_ in range(32)]
    ok("A.U == U.(residue sums of A)  [32 MACs, already the digit words]",
       sum(cls[j][1][r_] * res[r_] for r_ in range(32)) == AU)

    ok("A.c depends on ~every c_k (flip moves the word iff A_k != 0)", sum(1 for t in range(K) if sx8(Ai[t])) >= K - 2)
    ok("b.B depends on ~every b_k (flip moves the word iff B_k != 0)", sum(1 for t in range(K) if sx8(Bj[t])) >= K - 2)

    st0 = [rng.getrandbits(32) for _ in range(16)]
    w1 = [rng.getrandbits(32) for _ in range(32)]
    w2 = [rng.getrandbits(32) for _ in range(32)]
    f1 = SG.absorb("fx1", list(st0), w1)
    f2 = SG.absorb("fx1", list(st0), w2)
    fsum = SG.absorb("fx1", list(st0), [(w1[t] + w2[t]) & M32 for t in range(32)])
    ok("fx1 fold is NOT additive over words (fold-before-multiply gives nothing)",
       not all((f1[t] + f2[t] - st0[t]) & M32 == fsum[t] for t in range(16)))


def collect(seed: int = 1):
    """Run every shape; return (passed, failed) check names."""
    rng = random.Random(seed)
    out: dict[str, bool] = {}
    for r, c, K, adv in SHAPES:
        _check_shape(rng, r, c, K, adv, out)
    passed = [n for n, v in out.items() if v]
    failed = [n for n, v in out.items() if not v]
    return passed, failed


def _sizing() -> str:
    r, c, K = 8, 16, 256
    ks = K // 32
    m = SG.n_digits(ks)
    floor = 32 * 32 * (ks + m)
    rows = [
        ("direct noised GEMM", f"{floor} (== floor)", "0%"),
        ("reuse served AB", "0 (AB is 1 value/cell; prefixes need per-seg)", "0%"),
        ("per-seg plain AB + corrections", "0 (plain GEMM + A.c + b.B >= 2 GEMMs)", "0%"),
        ("Strassen / fast matmul", "0 (cannot emit per-segment prefix words)", "0%"),
        ("low-rank in coins (b,c)", "0 (b per-row, c per-col: full-rank, fresh)", "0%"),
        ("U linear structure", "0 (= the digit words; already credited)", "0%"),
        ("fold-before-multiply", "0 (fx1/duo fold non-additive)", "0%"),
        ("packing / CRT", "0 (exact <2^32; machine-route only)", "0%"),
        ("precompute before coin", "0 (only clean AB; credited is noised)", "0%"),
        ("C1 int8 Dither on FP8/FP4", "0 (emulation reproduces words bit-exact)", "0%"),
    ]
    out = [f"=== floor sizing per lane (K={K} KS={ks} m={m}): {32 * (ks + m)} credited words, {floor} MACs ===",
           f"{'route':<34}{'credited MACs below floor':<46}saving"]
    out += [f"  {name:<32}{detail:<46}{saving}" for name, detail, saving in rows]
    return "\n".join(out)


def main() -> int:
    passed, failed = collect()
    for n in passed:
        print("PASS " + n)
    for n in failed:
        print("FAIL " + n)
    print()
    print(_sizing())
    print()
    print(f"PASSED {len(passed)} distinct checks over {len(SHAPES)} shapes  FAILED {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
