"""The extreme linear route: the prover knows g − 1 independent linear statistics of a group's g cells, so the cells are
known up to one direction, z = z0 + t·v with t uniform in Z/2^16 (uniform cells). It skipped 1/g of the group's dots.
Its best guess of W(t) = sum_j f(z0_j + t v_j) mod 2^16 succeeds with max_w #{t : W(t) = w} / 2^16, computed exactly
over all t. Directions: v = (1, ..., 1) (it knows every difference z_j − z_1), v random with odd entries, and v random.
The lemma's bound is vacuous here (r = g − 1 > r_max); this measures what the word keeps."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402

N, M16 = F.N, F.M16
T = np.arange(N, dtype=np.int64)


def success(f, g: int, kind: str, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    got = []
    for _ in range(trials):
        z0 = rng.integers(0, N, size=g)
        v = np.ones(g, dtype=np.int64) if kind == "ones" else rng.integers(0, N, size=g)
        if kind == "odd":
            v |= 1
        W = np.zeros(N, dtype=np.int64)
        for j in range(g):
            W += f((z0[j] + T * v[j]) & M16).astype(np.int64)
        got.append(int(np.bincount(W & M16, minlength=N).max()))
    got = np.array(got)
    return {"mean_max_count": float(got.mean()), "max_max_count": int(got.max()), "success_over_1_N": float(got.mean())}


def main(out: str) -> None:
    res = {"note": "success x N: a uniform word over 2^16 values of t has max count about 8 (balls in bins)"}
    u = np.random.default_rng(5).integers(0, N, size=(20, N))
    res["uniform_baseline_max_count"] = float(np.mean([np.bincount(r, minlength=N).max() for r in u]))
    for name in ("ident", "square", "tfun", "mix", "mix4", "m32a", "feistel3", "rand"):
        f = F.CANDIDATES[name][0]
        for kind in ("ones", "odd", "any"):
            for g in (2, 8, 32, 128):
                r = success(f, g, kind, 20 if g <= 32 else 8, 13 * g + len(kind))
                res[f"{name}_{kind}_g{g}"] = r
                print(name, kind, g, r, flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "line.json")
