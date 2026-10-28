"""The narrow-U check: co32c (checked_strip.py's CO-Dither with committed strip digits) with U on ±N, N = 8, 16, 32,
so that the peeled side keeps 127 − N of its int8 range (the bench's `co32c-u8`, `co32c-u16`, `co32c-u32`), beside
N = 64, the U that checked_strip.py fixes.

U(i, ρ) = −N + Σ_p w_p β_p(i, ρ) over p = log2(N) + 1 planes, w = (1, 2, …, N/2, N + 1): 2N letters ±1..±N, never 0,
and U's parity varies (the odd top weight, as checked_strip.py's 65). For each N:
- A. Ranges: P ∈ [−(127 − N), 126 − N], so P̂ = P + c and P̃ = P̂ + U are int8. The served output O = O′ − Σ 128^d D_d
  in s32 at a random and at the zero instance, and at A = B = 0 every committed word of every cell varies with the
  coins (none is a constant a cheater could serve), with D_1 = U·z_0 nonzero in ≥ 95% of cells.
- B. co32c's cross ranks: k + r in both orientations at the zero, a random and a structured instance, same-side blocks
  0 over F₂ (checked_strip.py's part B for co32c, at U's narrow planes). Mutation: an even top weight, which makes U's
  parity constant, loses rank.
- C. Dither-C's cohalf Q Kᵀ (checked_strip.py's design `co` at r = k/2) at U's narrow planes, the Q Kᵀ H1
  carries (`co32c-u16-cohalf`, Q and K within ±(127 − N)): cross ranks (k, k + r) at k = 64, 96, 128 at the zero, a
  random and a structured instance, same-side blocks 0, and at Q = K = 0 every committed word varies with the coins.
  Mutation: an even top weight loses rank.
- H. The tensor–scalar model (checked_strip.py's TS facets and tiles) at A = B = 0 with every committed word bit-sliced
  at once: a T increment is 1 + p binary products (c and U's p planes), D_d is p, O′'s c·b is k/32 (checked_strip.py's
  `word_routes` at p planes). On the Llama-3.1 8B and 70B linears no route gains at the duo fold (3) for x ≥ 4 warp
  instructions per 4,096 bit products, at every N. At folds 1.5 and 1 the breakeven x rises as U narrows (6.77 at
  N = 64, 9.48 at 16, 11.86 at 8): at the measured x = 10.66 U on ±8 loses, ±16 holds.
- T. Byte packing for the packed shared-memory table route (the closest table construction to design A's
  threshold): a sum of e entries of (c + U)·b spans e(2N + 1) + 1 values, so a byte field holds at most ⌊255/(2N + 1)⌋
  entries' sum, against 255 for design A's c·b. For N ≥ 8 two 8-entry chunks never share a byte (each chunk needs its
  own extracting writer, or halfword fields), where design A's four chunks fit one byte.

Run from the repository root: `uv run --no-project --with numpy --with python-sat --with-editable
benchmarks/pouw/exact_kit python -B benchmarks/pouw/theory/narrow_u.py --mutations`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checked_strip import (MASK, TS_SHAPES, DesignBlock, co32c_words, cross_ranks, instance, n_digits, o_routes,  # noqa: E402
                           s32, ts_best)
from exact_kit.checks import Checks  # noqa: E402

WIDTHS = (8, 16, 32, 64)
MEASURED_X = 10.66


def planes(N, even=False):
    """U's plane weights for U on ±1..±N: (1, 2, …, N/2, N + 1); `even` doubles the first and drops the odd top."""
    p = N.bit_length()
    w = [1 << j for j in range(p - 1)] + [N + 1]
    if even:
        w = [2] + w[1:-1] + [N]
    return np.array(w, dtype=np.int64)


class NarrowBlock(DesignBlock):
    """checked_strip.py's DesignBlock with U = −N + β·w, N the half-width the planes span."""

    def __init__(self, design, n, k, n2, r, L, w):
        super().__init__(design, n, k, n2, r, L, w=w)
        self.off = int(np.sum(w)) // 2

    def fields(self, Z):
        U, c, b, z = super().fields(Z)
        return U + 64 - self.off, c, b, z


def draw_u(rng, n, r, w):
    beta = rng.integers(0, 2, (n, r, len(w)))
    return -(int(np.sum(w)) // 2) + beta @ w


def word_routes_p(kk, x, p):
    m = n_digits((kk // 32) * -128, (kk // 32) * 127)
    routes = [(-(kk // 32), (1 + p) * x * (kk // 32))]
    if m > 1:
        routes.append((-(m - 1), p * x * (m - 1)))
    return routes + [o_routes("co32c", kk)["scalar"](x)]


def weighted_best(fold, x, p, mname):
    h = c_ = 0.0
    for _, kk, nn in TS_SHAPES[mname]:
        th, tb = ts_best("co32c", kk, nn, fold, word_routes_p(kk, x, p) if x else [])
        h, c_ = h + th * nn, c_ + tb * nn
    return h / c_ - 1


def breakeven(fold, p, mname):
    """The least x (to 0.01) at which no bit-sliced route gains over the cheater that does every IMMA."""
    base = weighted_best(fold, None, p, mname)
    lo, hi = 0.0, 64.0
    while hi - lo > 0.01:
        mid = (lo + hi) / 2
        if weighted_best(fold, mid, p, mname) > base + 1e-12:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    checks = Checks()
    rng = np.random.default_rng(20261017)
    out = {}
    L = 32
    for N in WIDTHS:
        w = planes(N)
        p = len(w)
        letters = sorted({-N + int(np.dot(bits, w)) for bits in np.ndindex(*(2,) * p)})
        P_rng, Pt_rng = (-(127 - N), 126 - N), (-(127 - N) + letters[0], 126 - N + 1 + letters[-1])
        checks.check(f"A{N} U's {2 * N} letters are ±1..±{N} (no 0, both parities); P̂ = P + c and P̃ = P̂ + U are "
                     f"int8 with P in {list(P_rng)}",
                     letters == [v for v in range(-N, N + 1) if v] and -128 <= Pt_rng[0] and Pt_rng[1] <= 127,
                     f"P̃ in {list(Pt_rng)}")

        nz, k6, r6 = 16, 256, 32
        m6 = n_digits((k6 // r6) * -128, (k6 // r6) * 127)
        U6 = draw_u(rng, nz, r6, w)
        c6, b6 = rng.integers(0, 2, (nz, k6)), rng.integers(0, 2, (k6, nz))
        z6 = rng.integers(0, 2, (m6 - 1, r6, nz))
        Pz, Vz = np.zeros((nz, k6), np.int64), np.zeros((k6, nz), np.int64)
        Pr, Vr = rng.integers(P_rng[0], P_rng[1] + 1, (nz, k6)), rng.integers(-128, 127, (k6, nz))
        served = lambda Op, D: s32(Op.astype(np.int64) - sum(Dd.astype(np.int64) << (7 * (d + 1))
                                                             for d, Dd in enumerate(D)))
        T6, D6, Op6 = co32c_words(Pz, Vz, U6, c6, b6, z6, r6, L, m6)
        Tr, Dr, Opr = co32c_words(Pr, Vr, U6, c6, b6, z6, r6, L, m6)
        draws = 64
        zero_words = []
        for _ in range(draws):
            Ud, cd, bd = draw_u(rng, 4, r6, w), rng.integers(0, 2, (4, k6)), rng.integers(0, 2, (k6, 4))
            zd = rng.integers(0, 2, (m6 - 1, r6, 4))
            Td, Dd, Opd = co32c_words(np.zeros((4, k6), np.int64), np.zeros((k6, 4), np.int64), Ud, cd, bd, zd, r6, L,
                                      m6)
            zero_words.append(np.stack(Td + Dd + [Opd]))
        zw = np.stack(zero_words)
        varying = np.all(zw.min(0) != zw.max(0))
        nonzero = np.count_nonzero(D6[0])
        checks.check(f"A{N} committed digits: O = O′ − Σ 128^d D_d in s32 (random and zero instance); at A = B = 0 "
                     f"D_1 = U·z_0 is nonzero in ≥ 95% of cells and every committed word of every cell varies with the "
                     f"coins ({draws} draws)",
                     np.array_equal(served(Opr, Dr), s32((Pr + c6) @ (Vr + b6)))
                     and np.array_equal(served(Op6, D6), s32(c6 @ b6)) and np.array_equal(D6[0], s32(U6 @ z6[0]))
                     and nonzero >= 0.95 * nz * nz and varying,
                     f"m = {m6}, D_1 nonzero in {nonzero} of {nz * nz} cells, {zw.shape[1]} words a cell")

        ranks = {}
        for (kk, rr, LL) in [(128, 32, 32), (64, 32, 32), (96, 16, 32)]:
            Bc = NarrowBlock("co32c", 1, kk, 1, rr, LL, w)
            res = [cross_ranks(Bc, *instance(Bc, kind, rng), mod=1) for kind in ("zero", "random", "balanced")]
            ranks[f"k={kk} r={rr} L={LL}"] = [list(x[:2]) for x in res]
            checks.check(f"B{N} co32c (k, r, L) = ({kk}, {rr}, {LL}), U on ±{N}: both cross ranks k + r = {kk + rr} at "
                         f"the zero, a random and a structured instance, same-side blocks 0 over F₂",
                         all(x[:2] == (kk + rr, kk + rr) and x[2] == 0 for x in res),
                         f"{[x[:2] for x in res]}, F₂ same-side nonzero {[x[2] for x in res]}")
        out[f"N={N} ranks"] = ranks
        if checks.mutations:
            Bm = NarrowBlock("co32c", 1, 64, 1, 32, 32, planes(N, even=True))
            r1, r2, _ = cross_ranks(Bm, *instance(Bm, "random", rng), mod=1)
            with checks.mutant(f"co32c with an even top weight, U on ±{N} (U's parity constant)"):
                checks.check(f"M{N} both cross ranks k + r = 96", r1 == r2 == 96, f"ranks {r1}, {r2}")

        half = {}
        for (kk, LL) in [(64, 32), (96, 32), (128, 32)]:
            rr = kk // 2
            Bh = NarrowBlock("co", 1, kk, 1, rr, LL, w)
            res = [cross_ranks(Bh, *instance(Bh, kind, rng)) for kind in ("zero", "random", "balanced")]
            half[f"k={kk} r={rr} L={LL}"] = [list(x[:2]) for x in res]
            checks.check(f"C{N} cohalf (co at r = k/2) (k, r, L) = ({kk}, {rr}, {LL}), U on ±{N}: cross ranks (peeled "
                         f"rows, coin rows) = (k, k + r) = ({kk}, {kk + rr}) at the zero, a random and a structured "
                         f"instance, same-side blocks 0",
                         all(x[:2] == (kk, kk + rr) and x[2] == 0 for x in res),
                         f"{[x[:2] for x in res]}, same-side nonzero {[x[2] for x in res]}")
        out[f"N={N} cohalf ranks"] = half
        Bz = NarrowBlock("co", 4, 64, 4, 32, 32, w)
        zw = Bz.words(np.zeros((4, 64), np.int64), np.zeros((64, 4), np.int64), rng.integers(0, 2, (draws, Bz.nbits)))
        checks.check(f"C{N} cohalf at Q = K = 0, U on ±{N}: every committed word of every cell (T_t and O) varies with "
                     f"the coins ({draws} draws)", bool(np.all(zw.min(0) != zw.max(0))), f"{zw.shape[1]} words")
        if checks.mutations:
            Bm = NarrowBlock("co", 1, 64, 1, 32, 32, planes(N, even=True))
            r1, r2, _ = cross_ranks(Bm, *instance(Bm, "random", rng))
            with checks.mutant(f"cohalf with an even top weight, U on ±{N} (U's parity constant)"):
                checks.check(f"MC{N} cross ranks (k, k + r) = (64, 96)", (r1, r2) == (64, 96), f"ranks {r1}, {r2}")

        gain3 = [weighted_best(3, x, p, m) - weighted_best(3, None, p, m) for m in TS_SHAPES for x in (4, 8, 12, 16)]
        be = {f"fold={f} {m}": breakeven(f, p, m) for f in (3, 1.5, 1) for m in TS_SHAPES}
        out[f"N={N} breakeven x"] = {k_: round(v, 2) for k_, v in be.items()}
        checks.check(f"H{N} co32c at A = B = 0, U on ±{N} ({p} planes), every word bit-sliced at once (T at {1 + p}x, "
                     f"D at {p}x, O′'s c·b at x·k/32): no gain on the Llama-3.1 linears at the duo fold (3) for x ≥ 4",
                     all(abs(g) < 1e-12 for g in gain3), f"breakeven x {out[f'N={N} breakeven x']}")
    cheap = {N: max(v for k_, v in out[f"N={N} breakeven x"].items() if not k_.startswith("fold=3 ")) for N in WIDTHS}
    checks.check(f"H at folds 1.5 and 1 the breakeven x falls as U widens; at the measured x = {MEASURED_X} a "
                 f"bit-sliced cheater gains with U on ±8 and with no wider U",
                 cheap[8] > MEASURED_X and all(cheap[N] < MEASURED_X for N in WIDTHS if N > 8)
                 and all(cheap[a] > cheap[b] for a, b in zip(WIDTHS, WIDTHS[1:])), str(cheap))

    fit = {}
    for N in (0,) + WIDTHS:
        span = 1 if N == 0 else 2 * N + 1
        fit["A (c·b)" if N == 0 else f"N={N}"] = {"entries a byte": 255 // span,
                                                  "8-entry chunks a byte": (255 // span) // 8}
    out["byte fields"] = fit
    checks.check("T byte fields: design A's c·b sums all four 8-entry chunks of a segment in one byte (the ¼ LDS.128 + "
                 "½ IADD3 + 1 IDP.4A core); with U on ±N, N ≥ 8, no two 8-entry chunks share a byte",
                 fit["A (c·b)"]["8-entry chunks a byte"] >= 4
                 and all(fit[f"N={N}"]["8-entry chunks a byte"] <= 1 for N in WIDTHS), str(fit))

    print("\nRESULTS " + json.dumps(out, default=str, sort_keys=True))
    sys.exit(checks.finish())


if __name__ == "__main__":
    main()
