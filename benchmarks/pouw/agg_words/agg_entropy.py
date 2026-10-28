"""How predictable an aggregated word is at Dither's most predictable segment: L = 64 inner indices at A = B = 0, noise
na = d + V and nb = e + U with V, U uniform on ±80 and d, e bits (i.i.d. inside one noise tile, L <= L′).

For each f and g (a row group: one x, g fresh columns), over n groups:
- p_max: the most frequent word's share, against p_max_uniform, the same statistic for n uniform words;
- coll: N · (collision probability), unbiased (1 for uniform words);
- coll_given_S: N · the collision probability of words among groups whose one-dot statistic S = x · sum_j w′_j mod 2^16
  agrees, the linear collapse's information (1 if S says nothing).
The single cell (g = 1, f = z) is Dither's T_1 at L = 64, which v3.2 measures at p <= 2.79e-5."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402

N, M16 = F.N, F.M16


def sample(rng, n, g, L):
    x = (rng.integers(-80, 81, size=(n, L), dtype=np.int16) + rng.integers(0, 2, size=(n, L), dtype=np.int16))
    w = (rng.integers(-80, 81, size=(n, g, L), dtype=np.int16) + rng.integers(0, 2, size=(n, g, L), dtype=np.int16))
    z = np.matmul(w.astype(np.float32), x.astype(np.float32)[:, :, None])[:, :, 0].astype(np.int64)
    S = (w.astype(np.int64).sum(axis=1) * x).sum(axis=1)
    return z, S


def coll(c, n):
    c = c.astype(np.float64)
    return float((c * (c - 1)).sum() / (n * (n - 1)) * N)


def coll_given(words, S):
    """N · P[words equal | S equal] over pairs, unbiased."""
    key = (S & M16).astype(np.int64) * N + words
    _, kc = np.unique(key, return_counts=True)
    _, sc = np.unique(S & M16, return_counts=True)
    pairs_eq = float((kc.astype(np.float64) * (kc - 1)).sum())
    pairs_s = float((sc.astype(np.float64) * (sc - 1)).sum())
    return pairs_eq / pairs_s * N if pairs_s else None


def run(f, g, L, n, seed):
    rng = np.random.default_rng(seed)
    words, Ss = [], []
    chunk = max(1, (1 << 22) // (g * L))
    done = 0
    while done < n:
        b = min(chunk, n - done)
        z, S = sample(rng, b, g, L)
        words.append((f(z & M16).astype(np.int64).sum(axis=1)) & M16)
        Ss.append(S)
        done += b
    w = np.concatenate(words)
    S = np.concatenate(Ss)
    c = np.bincount(w, minlength=N)
    u = np.bincount(np.random.default_rng(seed + 1).integers(0, N, size=n), minlength=N)
    return {"n": n, "p_max": float(c.max() / n), "p_max_uniform": float(u.max() / n), "coll": coll(c, n),
            "coll_uniform": coll(u, n), "coll_given_S": coll_given(w, S)}


def main(out: str, n: int = 1 << 22) -> None:
    res = {}
    L = 64
    res["cell_T1_L64"] = run(F.ident, 1, L, n, 1)
    print("cell", res["cell_T1_L64"], flush=True)
    for name in ("ident", "mod8", "mix", "mix3", "mix4", "m32a", "feistel3", "rand"):
        f = F.CANDIDATES[name][0]
        for g in (2, 8, 64):
            r = run(f, g, L, n if g <= 8 else n // 4, 10 + g)
            res[f"{name}_g{g}"] = r
            print(name, g, r, flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/entropy.json")
