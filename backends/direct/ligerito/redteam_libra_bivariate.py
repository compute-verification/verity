"""red-team-ligerito: the Libra mask g = a_0 + sum_i g_i(x_i) (zk.SumcheckMask) does not blind a round that binds TWO
variables (ligerito-sumcheck's bivariate opening round): the message's X1^a X2^b coefficients with a, b >= 1 receive
nothing from g, so they are a fixed function of the witness.  Toy zero-check over BabyBear (base field; the argument is
field-agnostic), n = 6 variables, degree 3 (eq x (A B - C)), the round message sent in full as s(X1, X2).

    python -m backends.direct.ligerito.redteam_libra_bivariate
"""
from __future__ import annotations

import random

P = 2013265921
N_VARS = 6
D = 3                                 # per-variable degree of the masked summand (zk.py: ("zc", n, 3))
GRID = range(D + 1)


def lin(x, b):
    return x % P if b else (1 - x) % P


def mle(table, point):
    """Multilinear extension of ``table`` (index bit j = variable j) at ``point`` (list of field elements)."""
    acc = 0
    for idx, v in enumerate(table):
        w = 1
        for j, x in enumerate(point):
            w = w * lin(x, (idx >> j) & 1) % P
        acc = (acc + w * v) % P
    return acc


def eq(tau, x):
    w = 1
    for t, xi in zip(tau, x):
        w = w * ((t * xi + (1 - t) * (1 - xi)) % P) % P
    return w


def witness(rng):
    A = [rng.randrange(P) for _ in range(1 << N_VARS)]
    B = [rng.randrange(P) for _ in range(1 << N_VARS)]
    C = [a * b % P for a, b in zip(A, B)]              # every constraint A.B = C holds: the zero-check claim is 0
    return A, B, C


def libra(rng, cross: bool):
    g = {"a0": rng.randrange(P), "uni": [[rng.randrange(P) for _ in range(D)] for _ in range(N_VARS)]}
    if cross:   # the fix: a bivariate block for the packed pair (X1, X2)
        g["cross"] = [[rng.randrange(P) for _ in range(D)] for _ in range(D)]
    return g


def g_eval(g, x):
    v = g["a0"]
    for i, xi in enumerate(x):
        for e in range(D):
            v = (v + g["uni"][i][e] * pow(xi, e + 1, P)) % P
    if "cross" in g:
        for a in range(D):
            for b in range(D):
                v = (v + g["cross"][a][b] * pow(x[0], a + 1, P) * pow(x[1], b + 1, P)) % P
    return v


def opening_message(W, tau, g=None, rho=0):
    """s(X1, X2) = sum_{rest in {0,1}^{n-2}} [eq(tau, x) (A~ B~ - C~)(x) + rho g(x)] on the (D+1) x (D+1) grid."""
    A, B, C = W
    s = [[0] * (D + 1) for _ in GRID]
    for X1 in GRID:
        for X2 in GRID:
            acc = 0
            for rest in range(1 << (N_VARS - 2)):
                x = [X1, X2] + [(rest >> j) & 1 for j in range(N_VARS - 2)]
                f = eq(tau, x) * ((mle(A, x) * mle(B, x) - mle(C, x)) % P) % P
                if g is not None:
                    f = (f + rho * g_eval(g, x)) % P
                acc = (acc + f) % P
            s[X1][X2] = acc
    return s


def coeffs(s):
    """Monomial coefficients c[a][b] of the bivariate s from its values on {0..D}^2 (Lagrange, exact mod p)."""
    def lag(vals):
        n = len(vals)
        out = [0] * n
        for i, yi in enumerate(vals):
            num, den = [1], 1
            for j in range(n):
                if j != i:
                    num = [((num[k - 1] if k else 0) - j * (num[k] if k < len(num) else 0)) % P for k in range(len(num) + 1)]
                    den = den * (i - j) % P
            inv = pow(den, P - 2, P)
            for k in range(n):
                out[k] = (out[k] + yi * num[k] * inv) % P
        return out
    rows = [lag(s[X1]) for X1 in GRID]                             # coefficients in X2 for each X1 value
    return [[lag([rows[X1][b] for X1 in GRID])[a] for b in GRID] for a in GRID]


def cross(c):
    return tuple(c[a][b] for a in range(1, D + 1) for b in range(1, D + 1))


def main():
    rng = random.Random(20260923)
    tau = [rng.randrange(P) for _ in range(N_VARS)]
    W0, W1 = witness(rng), witness(rng)
    bare0 = cross(coeffs(opening_message(W0, tau)))
    trials = []
    for _ in range(4):
        rho = rng.randrange(1, P)
        trials.append(cross(coeffs(opening_message(W0, tau, libra(rng, cross=False), rho))))
    print("Libra g = a0 + sum_i g_i(x_i) (zk.SumcheckMask) on a round binding (X1, X2):")
    print(f"  9 cross coefficients X1^a X2^b (a,b>=1) identical over 4 fresh (rho, g) draws: {len(set(trials)) == 1}")
    print(f"  ... and equal to the UNMASKED witness-only coefficients:                     {trials[0] == bare0}")
    other = cross(coeffs(opening_message(W1, tau, libra(rng, cross=False), rng.randrange(1, P))))
    print(f"  a second witness (same statement: claim 0) gives different cross coefficients: {other != bare0}"
          "  -> distinguisher with advantage ~1")
    fixed = [cross(coeffs(opening_message(W0, tau, libra(rng, cross=True), rng.randrange(1, P)))) for _ in range(3)]
    print(f"with a (D x D) cross block for the packed pair: cross coefficients vary per draw: {len(set(fixed)) == 3}"
          f" and differ from the bare ones: {all(f != bare0 for f in fixed)}")


if __name__ == "__main__":
    main()
