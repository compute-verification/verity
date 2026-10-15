"""Hill-climb for violations of the star inequality for <n,n,n> matmul (L = n).

Star inequality (a necessary consequence of Conjecture A: the tree 0 -> Z -> every target):
    sum_t max(0, rank T_t - rank(T_t - Z)) <= rank Z.
Also reports the weaker cell bound sum_t max(0, rank Z - rank Z_t^o) where Z_t^o zeroes t's cell.
Usage: star_search.py n p iters seed
"""

import sys

import numpy as np

import fq


def main():
    n, p, iters, seed = (int(a) for a in sys.argv[1:5])
    rng = np.random.default_rng(seed)
    T = fq.matmul_targets(n)
    keys = sorted(T)
    Ts = np.stack([T[k] for k in keys])
    N = n * n

    def excess(Zb):
        rz = fq.rank(Zb, p)
        tot = np.zeros(len(Zb), dtype=np.int64)
        for i in range(len(keys)):
            a = n - fq.rank((Ts[i][None] - Zb) % p, p)
            tot += np.maximum(a, 0)
        return tot - rz, rz

    best = -10**9
    B = 64
    for it in range(iters):
        # start: sum of a few cell-local rank-1 pieces plus a few dense rank-1s
        Z = np.zeros((B, N, N), dtype=np.int64)
        for b in range(B):
            for _ in range(rng.integers(1, 2 * n + 1)):
                u = np.zeros(N, dtype=np.int64)
                v = np.zeros(N, dtype=np.int64)
                if rng.random() < 0.7:
                    r, c = rng.integers(0, n, 2)
                    u[r * n:(r + 1) * n] = rng.integers(0, p, n)
                    v[c::n] = rng.integers(0, p, n)
                else:
                    u = rng.integers(0, p, N)
                    v = rng.integers(0, p, N)
                Z[b] = (Z[b] + np.outer(u, v)) % p
        ex, _ = excess(Z)
        for step in range(60):
            Zn = Z.copy()
            for b in range(B):
                k = rng.integers(1, 3)
                for _ in range(k):
                    i, j = rng.integers(0, N, 2)
                    Zn[b, i, j] = rng.integers(0, p)
            exn, _ = excess(Zn)
            acc = exn >= ex
            Z[acc] = Zn[acc]
            ex[acc] = exn[acc]
        if ex.max() > best:
            best = ex.max()
            print("iter", it, "best excess", best, flush=True)
        if best > 0:
            b = int(np.argmax(ex))
            print("VIOLATION", Z[b].tolist())
            break
    print("final best excess", best)


if __name__ == "__main__":
    main()
