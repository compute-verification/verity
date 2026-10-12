"""Complete a k = 1 Sloth word from its top bits: is a partial rebuild priced per bit?

At k = 1 a stored word is y = ρ(a), a = m + c_0 + t(i) mod p, so y² ≡ ±a (wide.py's ρ: the sign is y's parity).
Under the decode-free bound the adversary knows m and the seed, so it knows a. Keep only H = y >> x and write
y = H·2^x + u with u < 2^x: then c·u + w ≡ r (mod p) with c = 2H·2^x, r = ±a − H²·2^(2x) and w = u² < 2^(2x).
For x < b/3 that linear congruence has one small solution, which Gauss reduction of a 2-dimensional lattice and
Babai rounding find. `rate-codec-work` prices rebuilding those x bits at x/b of a root.

`uv run --no-project --with gmpy2 --with python-flint python partial_root.py [WORDS]` prints one JSON line a
(field, x): the share of words completed and checked against ρ, the reduction's iteration count, and the time of
the completion against a root (both in C through gmpy2 / FLINT, the completion's control flow in Python).
"""

from __future__ import annotations

import json
import random
import sys
import time

import flint
import gmpy2
from gmpy2 import mpz

FIELDS = {"p512": (512, 569), "p768": (768, 825), "p1536": (1536, 3453)}


def rho(a: mpz, p: mpz, e: mpz) -> mpz:
    """wide.py's ρ: the even root of a when a is a square, else the odd root of −a."""
    s = gmpy2.powmod(a, e, p)
    if s * s % p == a:
        return s if s % 2 == 0 else p - s
    return s if s % 2 == 1 else p - s


def gauss(b1: tuple[mpz, mpz], b2: tuple[mpz, mpz]) -> tuple[tuple[mpz, mpz], tuple[mpz, mpz], int]:
    """Lagrange-Gauss reduction of a 2-dimensional basis; returns the reduced pair and the iteration count."""
    n1, n2 = b1[0] * b1[0] + b1[1] * b1[1], b2[0] * b2[0] + b2[1] * b2[1]
    if n1 > n2:
        b1, b2, n1, n2 = b2, b1, n2, n1
    it = 0
    while True:
        it += 1
        q = gmpy2.f_div(2 * (b1[0] * b2[0] + b1[1] * b2[1]) + n1, 2 * n1)
        b2 = (b2[0] - q * b1[0], b2[1] - q * b1[1])
        n2 = b2[0] * b2[0] + b2[1] * b2[1]
        if n2 >= n1:
            return b1, b2, it
        b1, b2, n1, n2 = b2, b1, n2, n1


def complete(h: mpz, x: int, a: mpz, p: mpz, reducer: str) -> tuple[mpz | None, int]:
    """y from its top bits h and a, or None; tries both signs of y² ≡ ±a."""
    y0 = h << x
    wgt = mpz(1) << x
    c = 2 * y0 % p
    iters = 0
    for sign in (1, -1):
        r = (sign * a - y0 * y0) % p
        if reducer == "flint":
            m = flint.fmpz_mat([[int(wgt), int(-c % p)], [0, int(p)]]).lll()
            b1 = (mpz(int(m[0, 0])), mpz(int(m[0, 1])))
            b2 = (mpz(int(m[1, 0])), mpz(int(m[1, 1])))
        else:
            b1, b2, it = gauss((wgt, -c % p), (mpz(0), p))
            iters += it
        tx, ty = wgt << (x - 1), (mpz(1) << (2 * x - 1)) - r
        det = b1[0] * b2[1] - b1[1] * b2[0]
        al = gmpy2.f_div(2 * (tx * b2[1] - ty * b2[0]) + det, 2 * det)
        be = gmpy2.f_div(2 * (b1[0] * ty - b1[1] * tx) + det, 2 * det)
        for da in (0, -1, 1):
            for db in (0, -1, 1):
                v0 = (al + da) * b1[0] + (be + db) * b2[0]
                v1 = (al + da) * b1[1] + (be + db) * b2[1]
                u = v0 // wgt
                if 0 <= u < wgt and r + v1 == u * u:
                    y = y0 + u
                    if (y * y - sign * a) % p == 0 and (y % 2 == 0) == (sign == 1):
                        return y, iters
    return None, iters


def main(words: int) -> None:
    rng = random.Random(1)
    for name, (n, cc) in FIELDS.items():
        p = mpz((1 << n) - cc)
        e = (p + 1) // 4
        batch = [mpz(rng.randrange(int(p))) for _ in range(words)]
        t0 = time.perf_counter()
        ys = [rho(a, p, e) for a in batch]
        root_us = (time.perf_counter() - t0) / words * 1e6
        third = n // 3
        for x in sorted({third - 64, third - 16, third - 8, third - 4, third - 2, third, third + 4}):
            for reducer in ("gauss", "flint"):
                ok = iters = 0
                t0 = time.perf_counter()
                for a, y in zip(batch, ys):
                    got, it = complete(y >> x, x, a, p, reducer)
                    ok += got == y
                    iters += it
                dt = (time.perf_counter() - t0) / words * 1e6
                print(json.dumps({"field": name, "bits": n, "x": x, "x_over_b": round(x / n, 4), "reducer": reducer,
                                  "words": words, "completed": ok, "gauss_iters": iters / words if iters else None,
                                  "root_us": round(root_us, 2), "complete_us": round(dt, 2),
                                  "complete_over_root": round(dt / root_us, 4),
                                  "claim_over_root": round(x / n, 4)}), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
