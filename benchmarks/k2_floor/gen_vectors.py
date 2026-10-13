"""Reference vectors for the k = 2 (and k = 3) partial-completion floor, from the trusted Python codec.

A k = 2 Sloth word over p = 2^b - c is y = ρ(ρ(a) + c′) (`partial_root_k2.py`). The adversary stores y's top
b - x bits H and its parity, knows (a, c′), and completes the low x bits by a 4-dimensional lattice reduction. This
dumps, for each (b, x), a batch of words as `A cp a y s2` in hex (A = H << x, s2 = y's stored sign), so a C or CUDA
implementation of the reduction can recover y and check it bit-for-bit against the true word here.

    research-free: `python gen_vectors.py OUT b c x words [k]` (k defaults to 2).
"""

from __future__ import annotations

import random
import sys

import gmpy2
from gmpy2 import mpz


def rho(a: mpz, p: mpz, e: mpz) -> mpz:
    s = gmpy2.powmod(a, e, p)
    if s * s % p == a:
        return s if s % 2 == 0 else p - s
    return s if s % 2 == 1 else p - s


def main() -> None:
    out, b, c, x, words = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    k = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    p = mpz((1 << b) - c)
    assert p % 4 == 3 and gmpy2.is_prime(p, 40), "p must be a 3-mod-4 prime"
    e = (p + 1) // 4
    rng = random.Random(0xC0DEC0DE ^ (b << 8) ^ x ^ (k << 20))
    lines = [f"{b} {c} {x} {words} {k}"]
    for _ in range(words):
        cs = [mpz(rng.randrange(int(p))) for _ in range(k)]
        xv = cs[0]
        for j in range(1, k):
            xv = (rho(xv, p, e) + cs[j]) % p
        y = rho(xv, p, e)
        a = cs[0]
        cp = cs[1] if k >= 2 else mpz(0)
        s2 = 1 if y % 2 == 0 else -1
        A = (y >> x) << x
        extra = " ".join(format(int(v), "x") for v in cs[2:])
        row = f"{format(int(A), 'x')} {format(int(cp), 'x')} {format(int(a), 'x')} {format(int(y), 'x')} {s2}"
        lines.append(row + (" " + extra if extra else ""))
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {words} words to {out}: b={b} c={c} x={x} k={k} claim_2x_b={2 * x / b:.4f}")


if __name__ == "__main__":
    main()
