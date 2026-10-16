"""The partial-root attack at its arithmetic floor: how far above a root are the reductions, tuned?

`partial_root.py` and `partial_root_k2.py` time FLINT's LLL and Python's control flow, whose fixed cost is large
next to a 512-bit root (FLINT's 4D LLL is 127 µs at 512 bits and 527 µs at 2,048, while the root grows 27×). So a
margin measured with them overstates the method's margin at small widths. Here, against one GMP root:
- k = 2: the 4-dimensional lattice reduced by FLINT and by fplll (fpylll) in its fast modes, each basis checked by
  completing the word against the encode (both signs of y's parity kept, as an adversary that stores it would);
- k = 1: GMP's extended GCD of (2·H·2^x, p), which the 2-dimensional Gauss reduction is a half of, as that
  method's floor.

`uv run --no-project --with gmpy2 --with python-flint --with fpylll --with cysignals python partial_floor.py
[WORDS [FIELD ...]]` prints one JSON line a (field, reducer).
"""

from __future__ import annotations

import json
import random
import sys
import time

import flint
import gmpy2
from fpylll import LLL, IntegerMatrix
from gmpy2 import mpz

from partial_root import FIELDS, rho


def rows_k2(h: mpz, x: int, cp: mpz, s2: int, p: mpz) -> list[list[int]]:
    A = h << x
    f3 = 4 * A % p
    f2 = (6 * A * A - 2 * s2 * cp) % p
    f1 = (4 * A * A * A - 4 * s2 * cp * A) % p
    return [[1 << (3 * x), 0, 0, int(-f1 % p)], [0, 1 << (2 * x), 0, int(-f2 % p)], [0, 0, 1 << x, int(-f3 % p)],
            [0, 0, 0, int(p)]]


def babai_k2(basis: list[list[int]], h: mpz, x: int, a: mpz, cp: mpz, s2: int, p: mpz) -> mpz | None:
    """Complete y from a reduced basis of the k = 2 lattice (s2 fixed: the adversary stores y's parity)."""
    A = h << x
    q = flint.fmpq_mat(basis).inv()
    last = [(mpz(int(q[3, i].p)), mpz(int(q[3, i].q))) for i in range(4)]
    w0 = mpz(1) << (3 * x)
    for s1 in (1, -1):
        f0 = (A ** 4 - 2 * s2 * cp * A * A + cp * cp - s1 * a) % p
        coeffs = [(2 * f0 * num + den) // (2 * den) for num, den in last]
        v0 = sum(coeffs[i] * basis[i][0] for i in range(4))
        u = gmpy2.f_div(v0 + w0 // 2, w0)
        for du in (0, -1, 1):
            uu = u + du
            if not 0 <= uu < (mpz(1) << x):
                continue
            y = A + uu
            y1 = (s2 * y * y - cp) % p
            if (y1 % 2 == 0) == (s1 == 1) and (s1 * y1 * y1 - a) % p == 0:
                return y
    return None


def reduce_flint(rows: list[list[int]]) -> list[list[int]]:
    red = flint.fmpz_mat(rows).lll()
    return [[int(red[i, j]) for j in range(4)] for i in range(4)]


def reduce_fplll(rows: list[list[int]], **kw) -> list[list[int]]:
    m = IntegerMatrix.from_matrix(rows)
    LLL.reduction(m, **kw)
    return [[int(m[i, j]) for j in range(4)] for i in range(4)]


REDUCERS = {
    "flint": reduce_flint,
    "fplll-default": lambda r: reduce_fplll(r),
    "fplll-fast-double": lambda r: reduce_fplll(r, method="fast", float_type="double"),
    "fplll-fast-longdouble": lambda r: reduce_fplll(r, method="fast", float_type="long double"),
    "fplll-heuristic-dpe": lambda r: reduce_fplll(r, method="heuristic", float_type="dpe"),
}


def main(words: int, fields: list[str]) -> None:
    rng = random.Random(3)
    for name in fields:
        n, cc = FIELDS[name]
        p = mpz((1 << n) - cc)
        e = (p + 1) // 4
        batch = [(mpz(rng.randrange(int(p))), mpz(rng.randrange(int(p)))) for _ in range(words)]
        t0 = time.perf_counter()
        ys = [rho((rho(a, p, e) + cp) % p, p, e) for a, cp in batch]
        root_us = (time.perf_counter() - t0) / words / 2 * 1e6
        x = n // 10 - 8
        cases = []
        for (a, cp), y in zip(batch, ys):
            s2 = 1 if y % 2 == 0 else -1
            cases.append((a, cp, y, s2, rows_k2(y >> x, x, cp, s2, p)))
        for rname, red in REDUCERS.items():
            try:
                t0 = time.perf_counter()
                bases = [red(r) for *_, r in cases]
                red_us = (time.perf_counter() - t0) / words * 1e6
            except Exception as exc:  # a float type fplll can't run at this width
                print(json.dumps({"field": name, "k": 2, "reducer": rname, "error": str(exc)[:120]}), flush=True)
                continue
            ok = sum(babai_k2(b, y >> x, x, a, cp, s2, p) == y for b, (a, cp, y, s2, _) in zip(bases, cases))
            print(json.dumps({"field": name, "bits": n, "k": 2, "x": x, "reducer": rname, "words": words,
                              "completed": ok, "root_us": round(root_us, 2), "reduce_us": round(red_us, 2),
                              "reduce_over_root": round(red_us / root_us, 4),
                              "claim_over_root": round(2 * x / n, 4)}), flush=True)
        # k = 1's floor: the 2D Gauss reduction is half of an extended GCD of (c, p).
        x1 = n // 3 - 4
        cs = [2 * ((y >> x1) << x1) % p for y in ys]
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            for c in cs:
                gmpy2.gcdext(c, p)
        g_us = (time.perf_counter() - t0) / reps / words * 1e6
        print(json.dumps({"field": name, "bits": n, "k": 1, "x": x1, "reducer": "gmp-gcdext (full; Gauss is half)",
                          "root_us": round(root_us, 2), "gcdext_us": round(g_us, 3),
                          "gcdext_over_root": round(g_us / root_us, 4), "claim_over_root": round(x1 / n, 4)}),
              flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, sys.argv[2:] or list(FIELDS))
