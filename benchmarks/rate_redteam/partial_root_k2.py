"""The partial-root attack against its own fix: does a k = 2 word give up a tail cheaply too?

At k = 2 (wide.py's `encode_word`) the stored word is y = ρ(ρ(a) + c′), with a = m + c₁ + t and c′ = c₂ + t, so
a = s₁·(s₂·y² − c′)² for the signs s₁, s₂ of the two roots. With y = A + u, A = H·2^x known and u < 2^x, that is a
monic quartic f(u) ≡ 0 (mod p). Linearizing z_i = u^i (i = 1..4, z_i < 2^(ix)) leaves one linear congruence in
four small unknowns, a 4-dimensional lattice with a closest-vector target, which LLL and Babai rounding solve for
x < b/10. `rate-codec-work` per bit prices x bits of a k = 2 word at 2x/b of a root.

`uv run --no-project --with gmpy2 --with python-flint python partial_root_k2.py [WORDS [FIELD ...]]` prints one JSON line a
(field, x): words completed and checked against the encode, and the completion's time over one root.
"""

from __future__ import annotations

import json
import random
import sys
import time

import flint
import gmpy2
from gmpy2 import mpz

from partial_root import FIELDS, rho


def complete(h: mpz, x: int, a: mpz, cp: mpz, p: mpz) -> mpz | None:
    """y from its top bits h at k = 2, or None: the quartic's small root by a linearized 4-dimensional lattice."""
    A = h << x
    w = [mpz(1) << (3 * x), mpz(1) << (2 * x), mpz(1) << x]
    for s2 in (1, -1):
        f3 = 4 * A % p
        f2 = (6 * A * A - 2 * s2 * cp) % p
        f1 = (4 * A * A * A - 4 * s2 * cp * A) % p
        rows = [[int(w[0]), 0, 0, int(-f1 % p)], [0, int(w[1]), 0, int(-f2 % p)], [0, 0, int(w[2]), int(-f3 % p)],
                [0, 0, 0, int(p)]]
        red = flint.fmpz_mat(rows).lll()
        basis = [[mpz(int(red[i, j])) for j in range(4)] for i in range(4)]
        q = flint.fmpq_mat([[int(v) for v in r] for r in basis]).inv()
        last = [(mpz(int(q[3, i].p)), mpz(int(q[3, i].q))) for i in range(4)]
        for s1 in (1, -1):
            f0 = (A ** 4 - 2 * s2 * cp * A * A + cp * cp - s1 * a) % p
            coeffs = [(2 * f0 * num + den) // (2 * den) for num, den in last]
            v0 = sum(coeffs[i] * basis[i][0] for i in range(4))
            u = gmpy2.f_div(v0 + w[0] // 2, w[0])
            for du in (0, -1, 1):
                uu = u + du
                if not 0 <= uu < (mpz(1) << x):
                    continue
                y = A + uu
                y1 = (s2 * y * y - cp) % p
                if ((y % 2 == 0) == (s2 == 1) and (y1 % 2 == 0) == (s1 == 1)
                        and (s1 * y1 * y1 - a) % p == 0):
                    return y
    return None


def main(words: int, fields: list[str]) -> None:
    rng = random.Random(2)
    for name in fields:
        n, cc = FIELDS[name]
        p = mpz((1 << n) - cc)
        e = (p + 1) // 4
        batch = [(mpz(rng.randrange(int(p))), mpz(rng.randrange(int(p)))) for _ in range(words)]
        t0 = time.perf_counter()
        ys = [rho((rho(a, p, e) + cp) % p, p, e) for a, cp in batch]
        root_us = (time.perf_counter() - t0) / words / 2 * 1e6
        tenth = n // 10
        for x in sorted({tenth - 32, tenth - 8, tenth, tenth + 8}):
            ok = 0
            t0 = time.perf_counter()
            for (a, cp), y in zip(batch, ys):
                ok += complete(y >> x, x, a, cp, p) == y
            dt = (time.perf_counter() - t0) / words * 1e6
            print(json.dumps({"field": name, "bits": n, "k": 2, "x": x, "x_over_b": round(x / n, 4), "words": words,
                              "completed": ok, "root_us": round(root_us, 2), "complete_us": round(dt, 2),
                              "complete_over_root": round(dt / root_us, 4),
                              "claim_over_root": round(2 * x / n, 4)}), flush=True)
        # The floor of this method: one LLL (an adversary that stores y's parity needs one sign) and one inverse.
        x = tenth - 8
        (a, cp), y = batch[0], ys[0]
        A = (y >> x) << x
        rows = [[1 << (3 * x), 0, 0, int(-4 * (A ** 3 - cp * A) % p)], [0, 1 << (2 * x), 0, int(-(6 * A * A - 2 * cp) % p)],
                [0, 0, 1 << x, int(-4 * A % p)], [0, 0, 0, int(p)]]
        m = flint.fmpz_mat(rows)
        reps = max(10, words // 5)
        t0 = time.perf_counter()
        for _ in range(reps):
            red = m.lll()
        lll_us = (time.perf_counter() - t0) / reps * 1e6
        t0 = time.perf_counter()
        for _ in range(reps):
            flint.fmpq_mat(red).inv()
        inv_us = (time.perf_counter() - t0) / reps * 1e6
        print(json.dumps({"field": name, "bits": n, "k": 2, "x": x, "parts": True, "lll_over_root": round(lll_us / root_us, 4),
                          "inverse_over_root": round(inv_us / root_us, 4), "claim_over_root": round(2 * x / n, 4)}),
              flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, sys.argv[2:] or list(FIELDS))
