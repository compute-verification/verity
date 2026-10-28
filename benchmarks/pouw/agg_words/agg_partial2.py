"""Partial computation with the linear collapse on the skipped piece: as agg_partial.py, but the prover also computes the
group's sum S = sum_j beta_j of the skipped dither products, which costs 1/g of them (a row group's d_A,s · sum_j d_B,s,j).
Its best guess of W given S is the mode of P(W = . | S), so its success is sum_s max_w P(W = w, S = s), from the joint
distribution of (W mod 2^16, S mod M) as the 2D convolution of the g cells' pushforwards. Without f (ident) W is then
exactly determined."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402
from agg_partial import pmf  # noqa: E402

N, M16 = F.N, F.M16


def success(f, g: int, L: int, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    off, p = pmf(L)
    sd = np.sqrt(L * 3 / 16 * g)
    M = 1 << int(np.ceil(np.log2(14 * sd + 1)))
    got = []
    for _ in range(trials):
        acc = None
        for z0 in rng.integers(0, N, size=g):
            nu = np.zeros((M, N), dtype=np.float64)
            nu[off % M, f((z0 + off) & M16).astype(np.int64)] = p
            s = sfft.rfft2(nu, workers=4)
            acc = s if acc is None else acc * s
        joint = sfft.irfft2(acc, s=(M, N), workers=4)
        got.append(float(np.clip(joint, 0, None).max(axis=1).sum()))
    got = np.array(got)
    return {"M": M, "mean": float(got.mean()), "max": float(got.max()), "over_1_N": float(got.mean() * N)}


def main(out: str) -> None:
    res = {}
    L = 256
    for name in ("ident", "mix", "mix4", "m32a", "rand"):
        f = F.CANDIDATES[name][0]
        for g in (2, 4, 8, 16):
            r = success(f, g, L, 3, 11 * g)
            res[f"L{L}_{name}_g{g}"] = r
            print(L, name, g, r, flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/partial2.json")
