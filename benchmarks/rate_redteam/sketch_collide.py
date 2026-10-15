"""Two int8 words with the same k_s-row ±1 sketch: is one skipped word bound exactly?

A tile's sketch is S·v over its n stored bytes, S a fresh k_s × n sign matrix. A word u of B bytes enters through
its k_s × B block S_u, so two words u ≠ u' give the same sketch, with every other byte of the tile equal, iff
S_u·(u − u') = 0 with u − u' in [−255, 255]^B. This finds such a difference d by LLL on the lattice
{(d, W·S_u·d)}, builds u and u' = u + d inside int8, and checks the sketch against a direct int64 product.

`uv run --no-project --with python-flint --with numpy python sketch_collide.py [SEEDS]` prints one JSON line a
(B, k_s, seed): the shortest kernel vector's max entry, whether it fits the int8 box, and the Gaussian-heuristic
length of the kernel lattice's shortest vector.
"""

from __future__ import annotations

import json
import math
import sys

import flint
import numpy as np

CASES = [(64, 64), (96, 16), (96, 64), (96, 72), (96, 80), (96, 82), (96, 84), (96, 86), (96, 88), (96, 92),
         (96, 96), (192, 16), (192, 64), (192, 160), (192, 168), (192, 176), (256, 16), (256, 64), (256, 224),
         (256, 232), (256, 240)]
WEIGHT = 1 << 40


def heuristic_l2(b: int, ks: int) -> float:
    """Gaussian heuristic for the kernel lattice of a random ±1 ks × b block: det ≈ √(b!/(b − ks)!) / 2^(ks − 1)."""
    r = b - ks
    if r == 0:
        return math.inf
    log2_det = 0.5 * (math.lgamma(b + 1) - math.lgamma(b - ks + 1)) / math.log(2) - (ks - 1)
    return math.sqrt(r / (2 * math.pi * math.e)) * 2 ** (log2_det / r)


def shortest_kernel(s: np.ndarray) -> np.ndarray | None:
    """The reduced basis's kernel vector of least max entry, or None when no basis vector lies in the kernel."""
    ks, b = s.shape
    rows = [[1 if j == i else 0 for j in range(b)] + [int(WEIGHT * s[r, i]) for r in range(ks)] for i in range(b)]
    red = flint.fmpz_mat(rows).lll()
    best = None
    for i in range(red.nrows()):
        v = [int(red[i, j]) for j in range(b + ks)]
        if any(v[b:]) or not any(v[:b]):
            continue
        d = np.array(v[:b], dtype=np.int64)
        if best is None or np.abs(d).max() < np.abs(best).max():
            best = d
    return best


def main(seeds: int) -> None:
    for b, ks in CASES:
        for seed in range(seeds):
            rng = np.random.default_rng([20261012, b, ks, seed])
            s = rng.choice(np.array([-1, 1], dtype=np.int64), size=(ks, b))
            full = ks >= b and np.linalg.matrix_rank(s.astype(float)) == b
            d = None if full else shortest_kernel(s)
            out = {"bytes": b, "k_s": ks, "seed": seed, "heuristic_l2": round(heuristic_l2(b, ks), 1),
                   "full_rank": bool(full)}
            if d is None:
                out.update(collision=False, kernel_found=False)
            else:
                fits = int(np.abs(d).max()) <= 255
                out.update(kernel_found=True, max_entry=int(np.abs(d).max()),
                           l2=round(float(np.linalg.norm(d)), 1), collision=fits)
                if fits:
                    lo = np.maximum(-128, -128 - d)
                    hi = np.minimum(127, 127 - d)
                    u = rng.integers(lo, hi + 1)
                    u2 = u + d
                    assert u.min() >= -128 and u.max() <= 127 and u2.min() >= -128 and u2.max() <= 127
                    assert not np.array_equal(u, u2)
                    assert np.array_equal(s @ u, s @ u2)
                    out["bytes_differing"] = int(np.count_nonzero(d))
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
