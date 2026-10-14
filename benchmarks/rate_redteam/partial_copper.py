"""The partial-root attack at k = 2 by Coppersmith's method: does a bigger lattice buy bits cheaper than the 4D one?

`partial_root_k2.py` linearizes the quartic f(u) ≡ 0 (mod p) into 4 dimensions, which completes x < b/10. Howgrave-
Graham's lattice for the same f, with p·u^i (i < 4), f and u·f, …, u^(t-1)·f, has dimension 4 + t and reaches
x < b·2t/((4+t)(3+t)): b/10 at t = 1, b/7.5 at t = 2 (dimension 6), b/7 at t = 3. Its shortest vector is an integer polynomial h with h(u) = 0 over
the integers. The adversary stores both roots' signs (2 bits a word), so each word takes one reduction. This times
fplll's fast LLL on it against one GMP root and checks h(u) = 0 at the true u (the integer root-finding of h, which
the adversary would also pay, is not timed).

`uv run --no-project --with gmpy2 --with python-flint --with fpylll --with cysignals python partial_copper.py
[WORDS [FIELD ...]]` prints one JSON line a (field, t, x).
"""

from __future__ import annotations

import json
import random
import sys
import time

from fpylll import LLL, IntegerMatrix
from gmpy2 import mpz

from partial_root import FIELDS, rho


def quartic(h: mpz, x: int, a: mpz, cp: mpz, s1: int, s2: int, p: mpz) -> list[int]:
    """f's coefficients, constant first: s1·(s2·(A + u)² − c′)² − a with A = h·2^x, monic in u (mod p)."""
    A = h << x
    f0 = (A ** 4 - 2 * s2 * cp * A * A + cp * cp - s1 * a) % p
    f1 = (4 * A ** 3 - 4 * s2 * cp * A) % p
    f2 = (6 * A * A - 2 * s2 * cp) % p
    f3 = 4 * A % p
    return [int(f0), int(f1), int(f2), int(f3), 1]


def lattice(f: list[int], x: int, t: int, p: int) -> list[list[int]]:
    n = 4 + t
    X = [1 << (x * i) for i in range(n)]
    rows = []
    for i in range(4):
        r = [0] * n
        r[i] = p * X[i]
        rows.append(r)
    for j in range(t):
        r = [0] * n
        for i, c in enumerate(f):
            r[i + j] = c * X[i + j]
        rows.append(r)
    return rows


REDUCERS = [("fplll-fast-double", {"method": "fast", "float_type": "double"}),
            ("fplll-heuristic-dpe", {"method": "heuristic", "float_type": "dpe"}), ("fplll-default", {})]


def main(words: int, fields: list[str]) -> None:
    rng = random.Random(5)
    for name in fields:
        n, cc = FIELDS[name]
        p = mpz((1 << n) - cc)
        e = (p + 1) // 4
        batch = [(mpz(rng.randrange(int(p))), mpz(rng.randrange(int(p)))) for _ in range(words)]
        t0 = time.perf_counter()
        ys = [rho((rho(a, p, e) + cp) % p, p, e) for a, cp in batch]
        root_us = (time.perf_counter() - t0) / words / 2 * 1e6
        for t, frac in ((1, (10, 9)), (2, (10, 8)), (2, (10, 7)), (3, (10, 8))):
            dim = 4 + t
            bound = n * 2 * t / (dim * (dim - 1))
            x = int(bound * frac[1] / frac[0])
            cases = []
            for (a, cp), y in zip(batch, ys):
                s2 = 1 if y % 2 == 0 else -1
                y1 = (s2 * y * y - cp) % p
                s1 = 1 if y1 % 2 == 0 else -1
                cases.append((y, quartic(y >> x, x, a, cp, s1, s2, p)))
            for method, kw in REDUCERS:
                mats = [IntegerMatrix.from_matrix(lattice(f, x, t, int(p))) for _, f in cases]
                try:
                    t0 = time.perf_counter()
                    for m in mats:
                        LLL.reduction(m, **kw)
                    red_us = (time.perf_counter() - t0) / words * 1e6
                    break
                except Exception:  # fplll's fast modes fail on wide entries; fall back to the next
                    continue
            ok = 0
            for m, (y, _) in zip(mats, cases):
                u = int(y) & ((1 << x) - 1)
                for row in range(2):
                    v = [int(m[row, i]) for i in range(dim)]
                    if any(v) and sum((v[i] >> (x * i)) * u ** i for i in range(dim)) == 0:
                        ok += 1
                        break
            print(json.dumps({"field": name, "bits": n, "k": 2, "method": "coppersmith", "reducer": method, "t": t,
                              "dim": dim,
                              "x": x, "x_over_b": round(x / n, 4), "x_bound_over_b": round(bound / n, 4),
                              "words": words, "completed": ok, "root_us": round(root_us, 2),
                              "reduce_us": round(red_us, 2), "reduce_over_root": round(red_us / root_us, 4),
                              "claim_over_root": round(2 * x / n, 4)}), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50, sys.argv[2:] or list(FIELDS))
