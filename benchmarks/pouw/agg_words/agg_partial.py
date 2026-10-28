"""Partial computation plus guessing: the prover computes every part of each cell's word except one piece of low entropy,
and guesses the aggregate. The piece is Dither's dither product d_A,s · d_B,s over a segment of L inner indices, which
the floor's split prices at 1/8 of G (`note:compute-research/20261028T0523Z-finding-dither-noised-floor`). Skipping it
leaves each cell known up to beta_j ~ Binomial(L, 1/4) about its mean, so the prover's best guess of
W = sum_j f(z0_j + beta_j) mod 2^16 is the mode of the convolution of the g pushforwards of beta_j under f(z0_j + .),
with z0_j uniform (the computed rest). Reported: the mean of that mode's probability over trials, for each f and g,
against 1/N. A per-cell word (g = 1, f = z) is the scheme without aggregation."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402

N, M16 = F.N, F.M16


def pmf(L: int):
    k = np.arange(L + 1)
    p = binom.pmf(k, L, 0.25)
    keep = p > 1e-12
    return (k[keep] - L // 4).astype(np.int64), p[keep]


def success(f, g: int, L: int, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    off, p = pmf(L)
    got = []
    for _ in range(trials):
        acc = None
        for z0 in rng.integers(0, N, size=g):
            nu = np.bincount(f((z0 + off) & M16).astype(np.int64), weights=p, minlength=N)
            s = np.fft.rfft(nu)
            acc = s if acc is None else acc * s
        dist = np.fft.irfft(acc, n=N)
        got.append(float(dist.max()))
    got = np.array(got)
    return {"mean": float(got.mean()), "max": float(got.max()), "over_1_N": float(got.mean() * N)}


def main(out: str) -> None:
    res = {"N": N, "one_over_N": 1 / N}
    for L in (1024, 256):
        for name in ("ident", "mod8", "square", "tfun", "mix1", "mix", "mix3", "mix4", "m32a", "m32b", "feistel3", "rand"):
            f = F.CANDIDATES[name][0]
            for g in (1, 2, 4, 8, 16, 64):
                trials = 200 if g <= 16 else 50
                r = success(f, g, L, trials, 7 * g + L)
                res[f"L{L}_{name}_g{g}"] = r
                print(L, name, g, r, flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/partial.json")
