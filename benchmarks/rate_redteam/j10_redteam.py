"""J10 red team (cursor/j10-shared-noise-9103 at 127bb8f7cc): one noise seed a call, row streams shared along a band
of at most 64 column tiles, column streams and fold keys per tile.

1. GridLuck at small sizes: exact moments over every coin, adversarial dead and lucky sets (random and local search),
   at the largest θ its condition allows; and its two load-bearing hypotheses broken one at a time (dead events of
   different rows correlated; a lucky event reading another skipped column's coins), at h1tile's decode grid.
2. GridTemplateCondition on a grid of ratios e = n/r and every w ≤ 32.
3. The copy channel at K = 64 and K = 4096 over a band of 32·64 = 2048 columns: the rank of the band's W' columns, an
   exhaustive search for short ±1 relations (one or two neighbours, whole vector and one segment), the per-cell
   multiply-add floor of any copy (the U term), four-Russians over the band-shared row, and an adaptive copy's luck.
4. The noise hashing per unit at decode (612 → 322 SHA-512 compressions).

Runs on a CPU in under a minute: ``uv run python benchmarks/rate_redteam/j10_redteam.py``.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np

RNG = np.random.default_rng(20261019)
RAND = random.Random(20261019)


# 1. GridLuck --------------------------------------------------------------------------------------------------------


def lm(p: float, x: float, kappa: float) -> float:
    return p * math.exp(x) + (1 - p) * math.exp(-kappa * x)


def condition(q: float, p: float, w: float, theta: float, r: int, n: int, kappa: float) -> float:
    return q * math.exp(n * theta * w) + (1 - q) * lm(p, r * theta * w, kappa) ** (n / r)


def max_theta(units, r: int, n: int, kappa: float) -> float:
    """The largest θ at which every unit (q, p, w) meets GridLuck's condition (it is convex in θ, 1 at θ = 0)."""
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if all(condition(q, p, w, mid, r, n, kappa) <= 1 for q, p, w in units):
            lo = mid
        else:
            hi = mid
    return lo


def grid_moment(r, n, nR, nC, nZ, w, dead, lucky, theta, kappa):
    """E over (ρ, z, γ) of exp(θ Σ w((1+κ)·lucky − κ)); dead[i][j] ⊆ range(nR); lucky[i][j][(ρ, z)] ⊆ range(nC)."""
    total = 0.0
    count = 0
    for rho in itertools.product(range(nR), repeat=r):
        for z in range(nZ):
            # given (ρ, z) the columns are independent: the moment factors over j
            prod = 1.0
            for j in range(n):
                s = 0.0
                for g in range(nC):
                    e = sum(w[i][j] * ((1 + kappa) * (g in lucky[i][j][(rho, z)]) - kappa) for i in range(r))
                    s += math.exp(theta * e)
                prod *= s / nC
            total += prod
            count += 1
    return total / count


def random_instance(r, n, nR, nC, nZ, p, correlate):
    w = [[RAND.choice([1, 2, 4, 8]) for _ in range(n)] for _ in range(r)]
    dead = [[set(RAND.sample(range(nR), RAND.choice([0, 1]))) for _ in range(n)] for _ in range(r)]
    size = int(p * nC)
    lucky = [[{} for _ in range(n)] for _ in range(r)]
    common = {j: set(RAND.sample(range(nC), size)) for j in range(n)}
    for i in range(r):
        for j in range(n):
            for rho in itertools.product(range(nR), repeat=r):
                for z in range(nZ):
                    if rho[i] in dead[i][j]:
                        lucky[i][j][(rho, z)] = set(range(nC))
                    elif correlate:
                        lucky[i][j][(rho, z)] = common[j]
                    else:
                        lucky[i][j][(rho, z)] = set(RAND.sample(range(nC), size))
    return w, dead, lucky


def gridluck_search(r, n, nR, nC, nZ, p, kappa, tries):
    worst = 0.0
    for t in range(tries):
        w, dead, lucky = random_instance(r, n, nR, nC, nZ, p, correlate=t % 2 == 0)
        units = [(len(dead[i][j]) / nR, p, w[i][j]) for i in range(r) for j in range(n)]
        theta = max_theta(units, r, n, kappa)
        worst = max(worst, grid_moment(r, n, nR, nC, nZ, w, dead, lucky, theta, kappa))
    return worst


def gridluck_local_search(r, n, nR, nC, p, kappa, steps):
    """Hill-climb the lucky sets (one (unit, ρ) flip a step) to maximise the moment at the condition's θ."""
    w, dead, lucky = random_instance(r, n, nR, nC, 1, p, correlate=False)
    units = [(len(dead[i][j]) / nR, p, w[i][j]) for i in range(r) for j in range(n)]
    theta = max_theta(units, r, n, kappa)
    size = int(p * nC)
    best = grid_moment(r, n, nR, nC, 1, w, dead, lucky, theta, kappa)
    for _ in range(steps):
        i, j = RAND.randrange(r), RAND.randrange(n)
        key = (tuple(RAND.randrange(nR) for _ in range(r)), 0)
        if key[0][i] in dead[i][j]:
            continue
        old = lucky[i][j][key]
        lucky[i][j][key] = set(RAND.sample(range(nC), size))
        m = grid_moment(r, n, nR, nC, 1, w, dead, lucky, theta, kappa)
        if m >= best:
            best = m
        else:
            lucky[i][j][key] = old
    return best


def broken_hypotheses():
    """h1tile's decode grid r = 32 rows, n = 32·64 columns, at GridTemplateMoment's θ = 1/(4·max(64r, n)), w = 32,
    κ = 4, p = ½, q = 2^-w: the exact log-moment of the worst instance each hypothesis allows, and with it broken."""
    r, n, w, kappa, p = 32, 2048, 32, 4.0, 0.5
    q = 2.0**-w
    theta = 1 / (4 * max(64 * r, n))
    out = {}
    # GridLuck's own bound (independent rows, column-local lucky): per row ≤ E[φ^{n/r}] ≤ 1
    unit = condition(q, p, w, theta, r, n, kappa)
    out["GridLuck condition per unit"] = unit
    # dead events of all rows together (a shared dead coin): with prob q all r·n units dead
    lm_unit = lm(p, theta * w, kappa)
    dead_all = q * math.exp(theta * w * r * n) + (1 - q) * lm_unit ** (r * n)
    out["log moment, every row's dead event one coin"] = math.log(dead_all)
    # lucky correlated along a row (each skipped column's guess reads another skipped column's coins so that a row's
    # lucky events coincide), each still ≤ p marginally
    row = q * math.exp(theta * w * n) + (1 - q) * lm(p, theta * w * n, kappa)
    out["log moment, a row's lucky events one event"] = r * math.log(row)
    return out


def template_lhs(e: float, w: int) -> float:
    """GridTemplateCondition's left side, the power taken as exp(e·log1p(·)) so that e up to 2^50 keeps its digits."""
    x = 1 / (4 * max(64.0, e))
    base = 0.5 * math.expm1(x * w) + 0.5 * math.expm1(-4 * x * w)
    return 0.5**w * math.exp(e * x * w) + (1 - 0.5**w) * math.exp(e * math.log1p(base))


def template_lhs_naive(e: float, w: int) -> float:
    x = 1 / (4 * max(64.0, e))
    return 0.5**w * math.exp(e * x * w) + (1 - 0.5**w) * (0.5 * math.exp(x * w) + 0.5 * math.exp(-4 * x * w)) ** e


def template_condition_scan(f):
    worst, at = 0.0, None
    for e in np.concatenate([np.logspace(-3, 15.06, 4000), np.linspace(60, 70, 1001)]):
        e = float(e)
        for w in range(1, 33):
            v = f(e, w)
            if v > worst:
                worst, at = v, (e, w)
    return worst, at


def template_limit(w: int) -> float:
    """The left side as e → ∞ (x = 1/(4e)): ½^w·e^{w/4} + (1 − ½^w)·e^{−3w/8}."""
    return 0.5**w * math.exp(w / 4) + (1 - 0.5**w) * math.exp(-3 * w / 8)


# 3. The copy channel ------------------------------------------------------------------------------------------------


def u_table(n_cols):
    """A stand-in for PouwU129: 32 values a column in ±64, period 32 along k."""
    return RNG.integers(-64, 65, size=(n_cols, 32))


def band_columns(K, n_cols, data):
    """W'_j = B_j + c_j + U_j as signed int8 (wrapped), one column a row of the result."""
    if data == "zero":
        B = np.zeros((n_cols, K), dtype=np.int64)
    elif data == "dup":
        B = np.tile(RNG.integers(-128, 128, size=(1, K)), (n_cols, 1))
    else:
        B = RNG.integers(-128, 128, size=(n_cols, K))
    c = RNG.integers(0, 2, size=(n_cols, K))
    U = np.tile(u_table(n_cols), (1, K // 32))
    W = (B + c + U + 128) % 256 - 128
    return W


def short_relations(W, targets=8):
    """Columns j with W_j = s·W_a or W_j = s·W_a + t·W_b over the integers, s, t ∈ ±1, a, b ≠ j (exhaustive)."""
    n = W.shape[0]
    index = {}
    for b in range(n):
        for t in (1, -1):
            index.setdefault((t * W[b]).tobytes(), []).append(b)
    singles = pairs = 0
    for j in RAND.sample(range(n), targets):
        for a in range(n):
            if a == j:
                continue
            for s in (1, -1):
                if np.array_equal(W[j], s * W[a]):
                    singles += 1
                rest = (W[j] - s * W[a]).tobytes()
                pairs += sum(1 for b in index.get(rest, []) if b not in (a, j))
    return singles, pairs, targets * (n - 1) * 2 * 2 * (n - 2)


def four_russians(K, g):
    """Per-cell additions for X_i·c_j (c bits) by tables over the row's X_i shared by 32·g columns, best chunk t."""
    best = None
    for t in range(1, 13):
        build = (K / t) * 2**t / (32 * g)
        look = K / t
        if best is None or build + look < best[0]:
            best = (build + look, t)
    return best


def adaptive_copy(K, trials):
    """A skipped cell copies its row neighbour's computed T (same X_i under J10, B_J1 = B_J0, A = 0), paying for no
    product: right iff X_i·(Δc + ΔU) = 0 on the prefix. Per word, over the call's coin."""
    X = RNG.integers(0, 2, size=(trials, K))
    dc = RNG.integers(0, 2, size=(trials, K)) - RNG.integers(0, 2, size=(trials, K))
    du = np.tile(RNG.integers(-64, 65, size=(trials, 32)) - RNG.integers(-64, 65, size=(trials, 32)), (1, K // 32))
    res = (X * (dc + du)).sum(axis=1)
    free_u = (X * dc).sum(axis=1)
    return float((res == 0).mean()), float((free_u == 0).mean())


# 4. Noise hashing at decode -----------------------------------------------------------------------------------------


def compressions(K=4096, m=2, g=64, other=36):
    rb = math.ceil((K + 32 * (m - 1)) / 512)
    cb = math.ceil((K + 256) / 512)
    per_tile = 32 * rb + 32 * cb + other
    shared = 32 * rb / g + 32 * cb + other
    return rb, cb, per_tile, shared


def main():
    print("== 1. GridLuck at small sizes (moment / |Ω|; the lemma says ≤ 1) ==")
    for (r, n, nR, nC, nZ, tries) in [(1, 2, 2, 2, 1, 200), (2, 2, 2, 2, 1, 200), (2, 3, 2, 4, 1, 60),
                                      (3, 2, 2, 2, 1, 60), (2, 2, 2, 4, 2, 60)]:
        worst = gridluck_search(r, n, nR, nC, nZ, 0.5, 4.0, tries)
        tag = " (lucky also reads a shared Ω₀, |Ω₀| = 2)" if nZ > 1 else ""
        print(f"r={r} n={n} |Rw|={nR} |Cl|={nC}{tag}: worst {worst:.9f} (worst − 1 = {worst - 1:.2e}) over {tries} random instances")
    for (r, n, nR, nC) in [(2, 2, 2, 4), (2, 3, 2, 2)]:
        print(f"r={r} n={n} |Rw|={nR} |Cl|={nC}: local search worst {gridluck_local_search(r, n, nR, nC, 0.5, 4.0, 400):.6f}")
    print()
    print("== 1b. GridLuck's hypotheses broken one at a time, decode grid r=32, n=2048, θ=1/(4·max(64r,n)), w=32 ==")
    for k, v in broken_hypotheses().items():
        print(f"{k}: {v:.6g}")
    print()
    print("== 2. GridTemplateCondition: max over e ∈ [1e-3, 2^50] (and 60..70), w ≤ 32 ==")
    worst, at = template_condition_scan(template_lhs_naive)
    print(f"naive float power: max LHS {worst:.12f} at e={at[0]:.4g}, w={at[1]} (a rounding artefact at large e)")
    worst, at = template_condition_scan(template_lhs)
    print(f"log1p/expm1: max LHS {worst:.12f} at e={at[0]:.4g}, w={at[1]} (≤ 1 holds: {worst <= 1})")
    lim = max((template_limit(w), w) for w in range(1, 33))
    print(f"limit e → ∞: max over 1 ≤ w ≤ 32 of ½^w·e^(w/4) + (1 − ½^w)·e^(−3w/8) = {lim[0]:.6f} at w={lim[1]}")
    print()
    print("== 3a. rank of a band's W' columns (32·64 = 2048 columns) ==")
    for K in (64, 4096):
        W = band_columns(K, 2048, "uniform")
        full = np.linalg.matrix_rank(W.astype(np.float64))
        seg = np.linalg.matrix_rank(W[:, :32].astype(np.float64))
        print(f"K={K}: rank {full} of min(K, n) = {min(K, 2048)}; one segment's rank {seg} of 32")
    print("A cell's word from other cells' words costs one coefficient per dimension of the span it needs: K per full")
    print("word, 32 per segment word; the honest cell is K multiply-adds (32 a segment). No relation is cheaper.")
    print()
    print("== 3b. short ±1 relations among a band's columns (exhaustive over 1 and 2 neighbours, 8 targets) ==")
    for K in (64, 4096):
        for data in ("zero", "dup", "uniform"):
            W = band_columns(K, 2048, data)
            s1, s2, cand = short_relations(W)
            s1g, s2g, _ = short_relations(np.ascontiguousarray(W[:, :32]))
            print(f"K={K} data={data}: whole word {s1} single, {s2} pair matches; first segment {s1g} single, "
                  f"{s2g} pair matches; of {cand:.3g} pair candidates")
    print()
    print("== 3c. per-cell work of a copy, multiply-adds (an honest cell is K int8 MACs on tensor cores) ==")
    for K in (64, 4096):
        KS = K // 32
        for g in (1, 4, 16, 64):
            fr, t = four_russians(K, g)
            print(f"K={K} g={g}: X·c by four-Russians {fr:.1f} adds (chunk {t}, {fr / K:.3f}K); "
                  f"U term over the {KS} prefixes {32 * KS} MACs (= K); X·B {K} MACs unless B = 0")
    print()
    print("== 3d. an adaptive copy's luck per word, K = 64, A = 0, B_J1 = B_J0 ==")
    for K in (64, 4096):
        paid, free = adaptive_copy(K, 200_000 if K == 64 else 20_000)
        print(f"K={K}: copy right (no product paid) {paid:.5f}; with the U term given free {free:.5f}")
    print()
    print("== 4. SHA-512 compressions a unit at decode (K = 4096, m = 2, other = 36) ==")
    rb, cb, per_tile, shared = compressions()
    print(f"row block {rb}, column block {cb} compressions; per tile {per_tile}; rows shared over g = 64: {shared:.1f}")
    print(f"ratio {per_tile / shared:.3f} (kernels measured 612 → 322: {612 / 322:.3f}); the noise half shared is the "
          f"rows' {32 * rb} of {32 * rb + 32 * cb}")


if __name__ == "__main__":
    main()
