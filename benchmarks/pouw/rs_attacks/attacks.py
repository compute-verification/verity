"""Cheapest algorithms for the committed prefix targets of (A + U Vᵀ)(B + W Zᵀ) (RS-D2A, CMP-13).

Masks E = U Vᵀ, F = W Zᵀ of rank r are drawn after A and B are committed; the prover commits
T_s = (A + E)[:, :sL] (B + F)[:sL, :] for s = 1..n/L. `check` verifies every attack route below bit for bit
against the honest targets over the integers on small instances and counts its operations; `table` prices
each route per committed word per block (one "output-block": n³/L of them) in int8 tensor MAC slots.

Run: python3 attacks.py check | table [--json OUT]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys

import numpy as np


def rng_pm1(g, shape):
    return g.choice(np.array([-1, 1], dtype=np.int64), size=shape)


def honest_targets(A, B, E, F, L):
    n = A.shape[0]
    At, Bt = A + E, B + F
    out, acc = [], np.zeros((n, n), dtype=np.int64)
    for s in range(n // L):
        acc = acc + At[:, s * L:(s + 1) * L] @ Bt[s * L:(s + 1) * L, :]
        out.append(acc.copy())
    return out


class Count:
    def __init__(self):
        self.mul = 0
        self.add = 0
        self.lookup = 0


def mm(X, Y, c):
    c.mul += X.shape[0] * X.shape[1] * Y.shape[1]
    c.add += X.shape[0] * X.shape[1] * Y.shape[1]
    return X @ Y


def attack_lowrank(U, V, W, Z, L, c, B=None):
    """A = 0. T_s = T_{s-1} + U (V_sᵀ B̃_s) with B̃ = B + W Zᵀ kept factored: inner dimension r a block.

    With B = 0 this is T_s = T_{s-1} + (U D_s) Zᵀ, D_s = V_sᵀ W_s."""
    n, r = U.shape
    T = np.zeros((n, n), dtype=np.int64)
    out = []
    for s in range(n // L):
        Vs, Ws = V[s * L:(s + 1) * L], W[s * L:(s + 1) * L]
        Ds = mm(Vs.T, Ws, c)                      # r x r, L r^2
        Gs = mm(U, Ds, c)                         # n x r, n r^2
        T = T + mm(Gs, Z.T, c)                    # n x n, n^2 r  <- the main term
        if B is not None:
            Hs = mm(Vs.T, B[s * L:(s + 1) * L], c)    # r x n, L r n
            T = T + mm(U, Hs, c)                  # n^2 r (fold into Gs Zᵀ as inner dim 2r, or Ds Zᵀ + Hs first)
        c.add += n * n
        out.append(T.copy())
    return out


def attack_lowrank_merged(U, V, W, Z, B, L, c):
    """A = 0, any B: T_s = U H_s, H_s = Σ_{j≤s} V_jᵀ (B_j + W_j Zᵀ) (r x n); inner dimension r, not 2r."""
    n, r = U.shape
    H = np.zeros((r, n), dtype=np.int64)
    out = []
    for s in range(n // L):
        Vs, Ws = V[s * L:(s + 1) * L], W[s * L:(s + 1) * L]
        H = H + mm(Vs.T, B[s * L:(s + 1) * L], c) + mm(mm(Vs.T, Ws, c), Z.T, c)
        out.append(mm(U, H, c))
    return out


def attack_duprows(A, U, V, W, Z, B, L, c, rep):
    """A has duplicated rows: rows i and rep[i] of A agree. Row rep[i] is computed honestly; row i is
    T_s[rep i] + (U[i] - U[rep i]) H_s, r MACs a word."""
    n, r = U.shape
    Bt = B + W @ Z.T
    out = []
    H = np.zeros((r, n), dtype=np.int64)
    reps = sorted(set(rep.tolist()))
    acc = np.zeros((len(reps), n), dtype=np.int64)
    At_rep = (A + U @ V.T)[reps]
    for s in range(n // L):
        sl = slice(s * L, (s + 1) * L)
        acc = acc + mm(At_rep[:, sl], Bt[sl], c)
        H = H + mm(V[sl].T, Bt[sl], c)
        T = np.zeros((n, n), dtype=np.int64)
        T[reps] = acc
        others = [i for i in range(n) if rep[i] != i]
        idx = {j: k for k, j in enumerate(reps)}
        if others:
            Du = U[others] - U[rep[others]]
            T[others] = acc[[idx[j] for j in rep[others]]] + mm(Du, H, c)
        out.append(T)
    return out


def strassen_thin(G, Zt, c):
    """One Strassen level on G (m x r) · Zt (r x n), r even: 7 products of (m/2 x r/2 x n/2), 18 block adds."""
    m, r = G.shape
    n = Zt.shape[1]
    a, b, d = m // 2, r // 2, n // 2
    A11, A12, A21, A22 = G[:a, :b], G[:a, b:], G[a:, :b], G[a:, b:]
    B11, B12, B21, B22 = Zt[:b, :d], Zt[:b, d:], Zt[b:, :d], Zt[b:, d:]
    c.add += 5 * a * b + 5 * b * d                # input adds (A side per block; B side shared over blocks)
    M1 = mm(A11 + A22, B11 + B22, c)
    M2 = mm(A21 + A22, B11, c)
    M3 = mm(A11, B12 - B22, c)
    M4 = mm(A22, B21 - B11, c)
    M5 = mm(A11 + A12, B22, c)
    M6 = mm(A21 - A11, B11 + B12, c)
    M7 = mm(A12 - A22, B21 + B22, c)
    c.add += 8 * a * d
    return np.block([[M1 + M4 - M5 + M7, M3 + M5], [M2 + M4, M1 - M2 + M3 + M6]])


def attack_strassen(U, V, W, Z, L, c):
    n, r = U.shape
    T = np.zeros((n, n), dtype=np.int64)
    out = []
    for s in range(n // L):
        Ds = mm(V[s * L:(s + 1) * L].T, W[s * L:(s + 1) * L], c)
        T = T + strassen_thin(mm(U, Ds, c), Z.T, c)
        c.add += n * n
        out.append(T.copy())
    return out


def attack_four_russians(U, V, W, Z, L, c, t):
    """Z ∈ {±1}: P_s[i, k] = Σ_g LUT_{s,i,g}[pattern_g(Z[k])], one lookup a group of t columns of G_s."""
    n, r = U.shape
    groups = [list(range(g, min(g + t, r))) for g in range(0, r, t)]
    pats = []
    for gi, cols in enumerate(groups):
        bits = (Z[:, cols] > 0).astype(np.int64)
        pats.append(bits @ (1 << np.arange(len(cols), dtype=np.int64)))
    signs = [np.array([[1 if (p >> j) & 1 else -1 for j in range(len(cols))] for p in range(1 << len(cols))],
                      dtype=np.int64) for cols in groups]
    T = np.zeros((n, n), dtype=np.int64)
    out = []
    for s in range(n // L):
        Ds = mm(V[s * L:(s + 1) * L].T, W[s * L:(s + 1) * L], c)
        Gs = mm(U, Ds, c)
        for gi, cols in enumerate(groups):
            lut = Gs[:, cols] @ signs[gi].T           # n x 2^t; Gray code makes it one add an entry
            c.add += n * (1 << len(cols))
            T = T + lut[:, pats[gi]]                  # one lookup and one add a word a group
            c.lookup += n * n
            c.add += n * n
        out.append(T.copy())
    return out


def check():
    g = np.random.default_rng(20261015)
    n, L = 128, 32
    rows = []
    for r in (1, 2, 4, 8, 16, 40):
        U, V, W, Z = (rng_pm1(g, (n, r)) for _ in range(4))
        E, F = U @ V.T, W @ Z.T
        zero = np.zeros((n, n), dtype=np.int64)
        Bany = g.integers(-50, 51, size=(n, n))
        honest0 = honest_targets(zero, zero, E, F, L)
        res = {}
        c = Count()
        res["lowrank A=B=0"] = (all(np.array_equal(x, y) for x, y in zip(attack_lowrank(U, V, W, Z, L, c), honest0)), c)
        honestB = honest_targets(zero, Bany, E, F, L)
        c = Count()
        res["lowrank A=0, any B"] = (all(np.array_equal(x, y) for x, y in
                                          zip(attack_lowrank_merged(U, V, W, Z, Bany, L, c), honestB)), c)
        rep = np.arange(n)
        rep[n // 2:] = rep[: n - n // 2]
        Adup = g.integers(-50, 51, size=(n, n))
        Adup[n // 2:] = Adup[: n - n // 2]
        c = Count()
        res["dup rows (half)"] = (all(np.array_equal(x, y) for x, y in
                                       zip(attack_duprows(Adup, U, V, W, Z, Bany, L, c, rep),
                                           honest_targets(Adup, Bany, E, F, L))), c)
        if r % 2 == 0:
            c = Count()
            res["strassen-1 thin"] = (all(np.array_equal(x, y) for x, y in
                                           zip(attack_strassen(U, V, W, Z, L, c), honest0)), c)
        c = Count()
        t = min(r, 8)
        res[f"four russians t={t}"] = (all(np.array_equal(x, y) for x, y in
                                            zip(attack_four_russians(U, V, W, Z, L, c, t), honest0)), c)
        for name, (ok, cnt) in res.items():
            words = n * n * (n // L)
            rows.append(dict(r=r, route=name, exact=bool(ok), mul_per_word=cnt.mul / words,
                             add_per_word=cnt.add / words, lookup_per_word=cnt.lookup / words))
    print(f"n = {n}, L = {L}, factors ±1, counts a committed word a block (honest: {L} MACs)")
    print(f"{'r':>3} {'route':<22} {'exact':<6} {'mul/word':>9} {'add/word':>9} {'lookup/word':>11}")
    for row in rows:
        print(f"{row['r']:>3} {row['route']:<22} {str(row['exact']):<6} {row['mul_per_word']:>9.2f} "
              f"{row['add_per_word']:>9.2f} {row['lookup_per_word']:>11.2f}")
    bad = [row for row in rows if not row["exact"]]
    if bad:
        sys.exit(f"inexact routes: {bad}")
    return rows


# Prices a committed word a block, in int8 tensor MAC slots (an m16n8k32 IMMA is 4,096 slots for 128 words).
# ledger: the converged model's dual weights (y_T 63/64, y_H 1/8, y_I 1/4, y_M 1/256; L, A, U and shared memory
# unpriced), i.e. priced use. time: the census rates (r20261015-064426-11ec), each op alone on the SM; prefix.cu
# measured IDP.4A at 16.6 and IMAD at 16.2-16.6 a word (r20261015-191053-26cf). sm_120a has no packed FFMA2:
# fma.rn.f32x2 lowers to two FFMAs (the same run's SASS), so an FP32 FMA or add is one lane-op.
PRICES = {
    "ledger": dict(fma=2.0, add=2.0, iop=2.0, imad=4.0, dp4a_op=4.0, lookup=2.0),
    "time": dict(fma=1024 / 124, add=1024 / 124, iop=1024 / 64, imad=1024 / 64, dp4a_op=1024 / 64,
                 lookup=1024 / 16),
}


def overflow_rows(r, L, cap=127):
    """Fraction of G_s rows (r entries, each about N(0, r L) for ±1 factors) with an entry beyond cap."""
    p = math.erfc((cap + 0.5) / math.sqrt(2 * r * L))
    return 1 - (1 - p) ** r


def bits_signed(v):
    return int(math.ceil(math.log2(v + 1))) + 1


def routes(r, L, n, price, beta=1):
    """Cost a committed word a block of each route for the A = 0 instance, factors bounded by beta."""
    p = PRICES[price]
    g_max = r * L * beta ** 3           # |G_s| = |U D_s|
    p_max = r * g_max * beta            # |P_s| = |G_s Zᵀ|, the block increment
    out = {}
    out["honest-shaped (E_s F_s, tensor)"] = L
    sig8 = math.ceil(bits_signed(g_max) / 8)
    out["tensor int8 (K = 32 or 16: both hold the pipe 16 cycles)"] = 32 * math.ceil(r / 32)
    if g_max <= 2048:
        out["tensor FP16 (k16), FP32 accumulate"] = 32 * math.ceil(r / 16)
    conv = p["iop"]                     # magic-number FP32 -> s32 fold into the running word: one IADD3
    lower = (r * r / n + r * r * L / (n * n)) * p["fma"]   # U D_s and D_s, spread over the block's n^2 words
    if p_max < 2 ** 22:
        out["CUDA FP32 FMA, r a word"] = r * p["fma"] + conv + lower
        if r >= 2 and r % 2 == 0:
            out["CUDA FP32, Strassen 1 level on r"] = 7 * r / 8 * p["fma"] + 2 * p["add"] + conv + lower
    out["CUDA IMAD (exact mod 2^32)"] = r * p["imad"] + lower
    # one int8 slice of G_s for the rows that fit (|G| is about sqrt(r L), far below its bound r L); the rest take
    # sig8 slices, a second accumulator and one op to merge them
    q = overflow_rows(r, L) if beta == 1 else 1.0
    one = math.ceil(r / 4) * p["dp4a_op"]
    out["CUDA IDP.4A, G_s in one int8 slice (rare rows in two)"] = (
        (1 - q) * one + q * (math.ceil(r * sig8 / 4) * p["dp4a_op"] + p["iop"]) + lower)
    if g_max < 2 ** 15:
        out["CUDA IDP.2A (16-bit G_s × 8-bit Z)"] = math.ceil(r / 2) * p["dp4a_op"] + lower
    # RS-D2F's rank-metric Steiner trees (note:20261015T1816Z-draft-rs-d2-lowrank-floor): their Steiner points are
    # rational, so exact arithmetic is mod 2^32, one IMAD a rank-one step a word
    if 2 <= r <= L and r % 2 == 0:
        out["Steiner hub tree mod 2^32 (IMAD), 3r/4 steps"] = 3 * r / 4 * p["imad"] + lower
    if L <= r <= 2 * L:
        out["Steiner median tree mod 2^32 (IMAD), (2L + r)/4 steps"] = (2 * L + r) / 4 * p["imad"] + lower
    if beta == 1:
        best = None
        for t in range(1, 17):
            pre = (math.ceil(r / t) * (2 ** min(t, r))) / n * p["add"]
            cost = math.ceil(r / t) * (p["lookup"] + p["add"]) + pre + lower
            if best is None or cost < best[0]:
                best = (cost, t)
        out[f"Four Russians on Zᵀ (±1), t={best[1]}"] = best[0]
    return out


def best_route(r, L, n, price, beta=1, lookups=True):
    rs = routes(r, L, n, price, beta)
    if not lookups:
        rs = {k: v for k, v in rs.items() if not k.startswith("Four")}
    name = min(rs, key=rs.get)
    return name, rs[name]


def instances(r, L, n, price, lookups=True):
    """Best cost a committed word a block for each choice of A, B (B̃ = B + W Zᵀ; rank(B) = ρ gives inner
    dimension r + ρ: T_s = (Ã_{≤s} [P W]_{≤s}) [Q Z]ᵀ for B = P Qᵀ)."""
    p = PRICES[price]
    best = lambda rr: min(best_route(rr, L, n, price, lookups=lookups)[1], L)
    out = {
        "A = 0 or B = 0 (other matrix anything)": best(r),
        "A or B of rank 1": best(r + 1),
        "A or B of rank 8": best(r + 8),
        "A with n/2 distinct rows (duplicated rows)": 0.5 * L + 0.5 * best(r),
        "A sparse, 8 nonzeros a row, B = W Zᵀ only": min(L, best(r) + 8 * L / n * p["fma"] + p["add"]),
    }
    if lookups:
        t = 11
        fr = math.ceil(L / t) * (p["lookup"] + p["add"]) + math.ceil(L / t) * 2 ** t / n * p["add"]
        out["A = ±1 dense, Four Russians on A's columns + low rank on E"] = min(L, fr + best(r))
    else:
        out["A = ±1 dense (no lookups: no structure left)"] = L
    out["A, B generic"] = L
    return out


def hk_rank_per_word(r):
    """Hopcroft–Kerr <m,2,m> (3m²+m)/2 products, tensored: (3/2)^d a word for r = 2^d·q (asymptotic in m)."""
    def direct(x):
        d, q = 0, x
        while q % 2 == 0:
            d, q = d + 1, q // 2
        return 1.5 ** d * q
    return min(direct(r), direct(1 << (r - 1).bit_length()))


def algebraic(L):
    rows = []
    for r in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96):
        # the chain, or RS-D2F's hub / median trees (rank-one steps a prefix): (2 min(r, L) + min(r, 2L)) / 4
        steiner = min(min(r, L), (2 * min(r, L) + min(r, 2 * L)) / 4 if r >= 2 else 1)
        strassen = min(r * (7 / 8) ** int(math.log2(r)) if r > 1 else 1, L * (7 / 8) ** int(math.log2(L)))
        hk = min(hk_rank_per_word(r), hk_rank_per_word(L))
        rows.append(dict(r=r, L=L, steiner=steiner / L, strassen=strassen / (L * (7 / 8) ** int(math.log2(L))),
                         hk=hk / hk_rank_per_word(L)))
    return rows


def algebraic_threshold(L, eps):
    """Model (1): smallest r whose best known Steiner length a prefix is at least (1 - eps) L."""
    r = 1
    while min(min(r, L), (2 * min(r, L) + min(r, 2 * L)) / 4 if r >= 2 else 1) < (1 - eps) * L:
        r += 1
    return r


# prefix.cu's mix48 (r20261015-191053-26cf): the tensor pipe keeps 0.861 of its peak beside IDP.4A words at 0.67 of
# their rate alone (16.6 a word); mix16 and mix32 give (0.952, 0.247) and (0.925, 0.48). Beyond 0.67 is unmeasured.
MIX = (0.861, 0.67)
DP4A_TIME = 16.6


def time_overlap(r, L):
    """Units a word a block with both pipes busy at MIX: the tensor route beside the one-slice IDP.4A route, for the
    attack (inner dimension r) and for the honest prover (inner dimension L), and their ratio."""
    fm, fi = MIX
    att = 1 / (fm / (32 * math.ceil(r / 32)) + fi / (DP4A_TIME * math.ceil(r / 4)))
    hon = 1 / max(1 / L, fm / L + fi / (DP4A_TIME * L / 4))
    return att, hon, min(att, hon) / hon


def time_overlap_threshold(L, eps):
    r = 1
    while time_overlap(r, L)[2] < 1 - eps:
        r += 1
    return r


def threshold(L, n, price, eps, rho, beta=1, rmax=4096, lookups=True):
    """Smallest r from which no route beats (1 - eps)·honest, receipts rho a word a block added to both."""
    honest = L + rho
    rstar = None
    for r in range(rmax, 0, -1):
        rs = routes(r, L, n, price, beta)
        if not lookups:
            rs = {k: v for k, v in rs.items() if not k.startswith("Four")}
        if min(rs.values()) + rho < (1 - eps) * honest:
            return r + 1
        rstar = r
    return rstar


def table(json_out=None):
    n = 8192
    res = {"n": n, "prices": PRICES, "routes": [], "thresholds": []}
    for price in ("ledger", "time"):
        print(f"\n== {price} prices, n = {n}, factors ±1; cost a committed word a block (honest = L)")
        for L in (32, 64, 128):
            print(f"-- L = {L}")
            for r in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
                rs = routes(r, L, n, price)
                name = min(rs, key=rs.get)
                res["routes"].append(dict(price=price, L=L, r=r, best=name, cost=rs[name], ratio=rs[name] / L,
                                          all=rs))
                print(f"   r={r:>4}  best {rs[name]:7.2f} ({rs[name] / L:5.2f} x honest)  {name}")
    print("\n== instance choices, L = 32, ratio to honest (ledger without lookups / ledger / time)")
    res["instances"] = []
    for r in (4, 16, 32):
        a = instances(r, 32, n, "ledger", lookups=False)
        b = instances(r, 32, n, "ledger")
        c = instances(r, 32, n, "time")
        for k in a:
            kb = k if k in b else next(x for x in b if x.startswith(k.split(" (")[0][:8]))
            kc = k if k in c else next(x for x in c if x.startswith(k.split(" (")[0][:8]))
            res["instances"].append(dict(r=r, choice=k, ledger_nolookup=a[k] / 32, ledger=b[kb] / 32,
                                         time=c[kc] / 32))
            print(f"   r={r:>3} {k:<52} {a[k] / 32:5.2f} {b[kb] / 32:5.2f} {c[kc] / 32:5.2f}")
    print("\n== model (1): products a committed word a block, ratio to the honest prefix family's own count")
    print("   (Steiner rank metric: honest n³; Strassen-only and Hopcroft–Kerr tensor powers: both sides use them)")
    res["algebraic"] = []
    for L in (32, 64):
        for row in algebraic(L):
            res["algebraic"].append(row)
            print(f"   L={L:>3} r={row['r']:>3}  Steiner {row['steiner']:5.2f}  Strassen {row['strassen']:5.2f}  "
                  f"HK {row['hk']:5.2f}")
    res["algebraic_thresholds"] = []
    for L in (32, 64, 128):
        for eps in (0.0, 0.05, 0.1):
            rs = algebraic_threshold(L, eps)
            res["algebraic_thresholds"].append(dict(L=L, eps=eps, rstar=rs))
            print(f"   model (1) r* (Steiner, best known tree): L={L:>3} eps={eps:<4} r*={rs}")
    print("\n== time with the tensor and integer pipes overlapped at the measured mix point "
          f"(tensor {MIX[0]}, IDP.4A {MIX[1]})")
    res["time_overlap"] = []
    for L in (32, 64):
        for r in (1, 4, 8, 16, 24, 28, 32, 48, 64):
            att, hon, ratio = time_overlap(r, L)
            res["time_overlap"].append(dict(L=L, r=r, attack=att, honest=hon, ratio=ratio))
            print(f"   L={L:>3} r={r:>3}  attack {att:6.2f}  honest {hon:6.2f}  ratio {ratio:5.2f}")
        for eps in (0.0, 0.05, 0.1):
            rs = time_overlap_threshold(L, eps)
            res["time_overlap"].append(dict(L=L, eps=eps, rstar=rs))
            print(f"   time-overlap r*: L={L:>3} eps={eps:<4} r*={rs}")
    print("\n== threshold r*: smallest r at which no route beats (1 - eps)·(honest + receipts)")
    print(f"{'prices':<7} {'L':>4} {'eps':>5} {'rho':>5} {'r* (all)':>9} {'r* (no lookups)':>16}")
    for price in ("ledger", "time"):
        for L in (32, 64, 128):
            for eps in (0.0, 0.05, 0.1):
                for rho in (0, L):
                    a = threshold(L, n, price, eps, rho)
                    b = threshold(L, n, price, eps, rho, lookups=False)
                    res["thresholds"].append(dict(price=price, L=L, eps=eps, rho=rho, rstar=a, rstar_no_lookup=b))
                    print(f"{price:<7} {L:>4} {eps:>5} {rho:>5} {a:>9} {b:>16}")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "table"])
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.what == "check":
        rows = check()
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rows, f, indent=1)
    else:
        table(a.json)


if __name__ == "__main__":
    main()
