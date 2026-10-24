"""The locality lemma's two algebraic steps, checked numerically on random supports.

For pieces a in S_A, b in S_B of Z_q^n and a consumer computing ||a + b||^2 mod 2Q exactly (Q = 2^ell, ell <= b):

- extraction: the matrix omega_{2Q}^{||a+b||^2} is diag(omega_{2Q}^{||a||^2}) . omega_Q^{<a,b>} . diag(omega_{2Q}^{||b||^2}),
  so it has the rank of the inner product's character table on the pieces' low ell planes;
- the rank bound: that rank is at least |low(S_A)| |low(S_B)| / Q^n (Frobenius norm squared over the full table's
  operator norm squared), so a cut's crossings number at least log|low(S_A)| + log|low(S_B)| - n ell.

    python bound_check.py
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np


def check(n, b, ell, nA, nB, rng):
    q, Q = 1 << b, 1 << ell
    full = list(itertools.product(range(q), repeat=n))
    SA, SB = rng.sample(full, nA), rng.sample(full, nB)
    w2, wq = np.exp(2j * np.pi / (2 * Q)), np.exp(2j * np.pi / Q)
    M = np.array([[w2 ** (sum(((x + y) % q) ** 2 for x, y in zip(a, bb)) % (2 * Q)) for bb in SB] for a in SA])
    IP = np.array([[wq ** (sum(x * y for x, y in zip(a, bb)) % Q) for bb in SB] for a in SA])
    DA = np.diag([w2 ** (sum(x * x for x in a) % (2 * Q)) for a in SA])
    DB = np.diag([w2 ** (sum(y * y for y in bb) % (2 * Q)) for bb in SB])
    low_a = {tuple(x % Q for x in a) for a in SA}
    low_b = {tuple(y % Q for y in bb) for bb in SB}
    bound = math.log2(len(low_a)) + math.log2(len(low_b)) - n * ell
    return np.allclose(M, DA @ IP @ DB), math.log2(np.linalg.matrix_rank(M)), bound


def main():
    rng = random.Random(1)
    print("| n | b | ell | S_A | S_B | extraction holds | log-rank | bound |")
    print("|---|---|---|---|---|---|---|---|")
    for n, b, ell, nA, nB in [(3, 2, 2, 64, 64), (3, 2, 2, 64, 16), (3, 3, 2, 200, 300), (4, 2, 1, 256, 256),
                              (2, 3, 3, 64, 64), (3, 3, 3, 100, 512), (3, 3, 3, 40, 40)]:
        ok, lr, bound = check(n, b, ell, nA, nB, rng)
        assert ok and lr >= bound - 1e-9, (n, b, ell, nA, nB)
        print(f"| {n} | {b} | {ell} | {nA} | {nB} | {ok} | {lr:.2f} | {bound:.2f} |")


if __name__ == "__main__":
    main()
