"""The partial-root attack's operation count: how many elementary steps the 4D reduction takes, against the 2D one.

`partial_floor.py` times reducers whose fixed cost hides the method's floor at small widths. This counts what any
LLL-type reduction of the k = 2 lattice does instead: the swaps and the size-reduction steps of a textbook LLL
(δ = 0.99, η = 0.51, exact GSO), each step's operand sizes (so its multiprecision work in 32-bit limbs), and fplll's
own swap count on the same basis, with the k = 1 lattice's Gauss steps beside it. Every reduced basis is checked by
completing the word against the encode.

`uv run --no-project --with gmpy2 --with python-flint --with fpylll --with cysignals python partial_count.py
[WORDS [FIELD ...]]` prints one JSON line a field.
"""

from __future__ import annotations

import json
import random
import sys
from fractions import Fraction

from fpylll import GSO, LLL, IntegerMatrix
from gmpy2 import mpz

from partial_floor import babai_k2, rows_k2
from partial_root import FIELDS, rho


def limbs(v: int) -> int:
    return max(1, (abs(v).bit_length() + 31) // 32)


def textbook_lll(rows: list[list[int]], delta: Fraction = Fraction(99, 100)) -> tuple[list[list[int]], dict]:
    b = [list(r) for r in rows]
    n = len(b)
    count = {"swaps": 0, "size_steps": 0, "size_limb_muls": 0, "q_bits": 0}

    def gso() -> tuple[list[list[Fraction]], list[Fraction]]:
        bs: list[list[Fraction]] = []
        mu = [[Fraction(0)] * n for _ in range(n)]
        nb: list[Fraction] = []
        for i in range(n):
            v = [Fraction(c) for c in b[i]]
            for j in range(i):
                mu[i][j] = sum(Fraction(b[i][t]) * bs[j][t] for t in range(n)) / nb[j]
                v = [v[t] - mu[i][j] * bs[j][t] for t in range(n)]
            bs.append(v)
            nb.append(sum(c * c for c in v))
        return mu, nb

    mu, nb = gso()
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                count["size_steps"] += 1
                count["q_bits"] += abs(q).bit_length()
                count["size_limb_muls"] += sum(limbs(c) for c in b[j]) * limbs(q)
                b[k] = [b[k][t] - q * b[j][t] for t in range(n)]
                mu, nb = gso()
        if nb[k] >= (delta - mu[k][k - 1] ** 2) * nb[k - 1]:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            count["swaps"] += 1
            mu, nb = gso()
            k = max(k - 1, 1)
    return b, count


def gauss_steps(c: int, p: int, x: int) -> int:
    """Gauss reduction of the k = 1 lattice [[2^x, c], [0, p]]: its quotient steps."""
    u, v = [1 << x, c], [0, p]
    steps = 0
    while True:
        if u[0] ** 2 + u[1] ** 2 > v[0] ** 2 + v[1] ** 2:
            u, v = v, u
        q = round(Fraction(u[0] * v[0] + u[1] * v[1], u[0] ** 2 + u[1] ** 2))
        if q == 0:
            return steps
        v = [v[0] - q * u[0], v[1] - q * u[1]]
        steps += 1


def main(words: int, fields: list[str]) -> None:
    rng = random.Random(4)
    for name in fields:
        n, cc = FIELDS[name]
        p = mpz((1 << n) - cc)
        e = (p + 1) // 4
        x = n // 10 - 8
        tot = {"swaps": 0, "size_steps": 0, "size_limb_muls": 0, "q_bits": 0}
        fswaps = ok = 0
        for _ in range(words):
            a, cp = mpz(rng.randrange(int(p))), mpz(rng.randrange(int(p)))
            y = rho((rho(a, p, e) + cp) % p, p, e)
            s2 = 1 if y % 2 == 0 else -1
            rows = rows_k2(y >> x, x, cp, s2, p)
            red, c = textbook_lll(rows)
            for key in tot:
                tot[key] += c[key]
            ok += babai_k2(red, y >> x, x, a, cp, s2, p) == y
            m = IntegerMatrix.from_matrix(rows)
            r = LLL.Reduction(GSO.Mat(m, float_type="dpe"))
            r()
            fswaps += r.nswaps
        x1 = n // 3 - 4
        g = sum(gauss_steps(int(2 * ((rho(mpz(rng.randrange(int(p))), p, e) >> x1) << x1) % p), int(p), x1)
                for _ in range(words))
        row = {"field": name, "bits": n, "k": 2, "x": x, "words": words, "completed": ok,
               "textbook_swaps": round(tot["swaps"] / words, 1), "textbook_size_steps": round(tot["size_steps"] / words, 1),
               "size_limb_muls_32": round(tot["size_limb_muls"] / words), "q_bits_mean":
               round(tot["q_bits"] / max(1, tot["size_steps"]), 1), "fplll_swaps": round(fswaps / words, 1),
               "k1_x": x1, "k1_gauss_steps": round(g / words, 1),
               "root_limb_muls_32": (n // 32) * (n // 32 + 1) // 2 * n + (n // 32) ** 2 * (n // 5)}
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50, sys.argv[2:] or list(FIELDS))
