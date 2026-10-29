"""The rank of a quadratic sketch's form in the lead's coin variables, which lower-bounds the products any ring program
needs: stdlib.

    python3 benchmarks/pouw/sketch/rank.py [--steps 4,8,16,32,64] [--seed 1] [--json OUT]

- **The words.** One row i of a 64-row tile and two adjacent output columns j, j + 1, with the data zero (data are fixed
  before the coins, so they add only affine terms and leave the form unchanged). Step tau's increment in column j is
  c_tau·W_tau(j) + a_1·L_tau1(j) + a_2·L_tau2(j) + a_3·L_tau3(j): the step's coin bit c_tau against a public weight,
  shared by the two columns, and three odd 7-bit coin lanes L = 2y + 1 per (tile, step, column), weighted
  (a_1, a_2, a_3) = (127, 1, 1) on even rows. The row tags' private G is drawn once a call, so it adds no step variable;
  leaving it out can only lower the rank. The variables are c_tau and the lanes' y.
- **The sketches.** `nh2`: S = sum_t w_t(j)·w_t(j+1) (one IMAD a pair of words, adjacent accumulators of one thread);
  `nh`: S = sum_k w_(2k-1)(j)·w_(2k)(j) (one IMAD a pair of steps); `sumsq`: S = sum_t w_t(j)^2 (one IMAD a word).
- **The bound.** A product of two linear forms has a Hessian of rank at most 2, so a program over a field that computes
  S with k multiplications has k >= rank(H)/2, H being S's Hessian in the variables (Strassen: for a quadratic form,
  products of linear forms are as good as any program). The rank is computed over GF(p) for a 61-bit prime (equal to
  the rational rank with high probability) and over GF(2), where the odd lanes' 2y vanishes and only the coin bits stay.
  It bounds one S alone: a tile's S values share their column terms, and `nh_attack.py` computes them all for less.
- **The operands.** A program with exactly rank(H)/2 products has operands spanning H's column space; a raw coin word is
  free, so the count of coordinate vectors e_i in that space (`coin_coordinates_in_colspace`; in RREF, e_i is in it iff
  i is a pivot column whose row is e_i) is how many operands could be free. Zero means every operand is a written value.
"""
from __future__ import annotations

import argparse
import json
import random

P = (1 << 61) - 1


def rref_mod(rows, p):
    """The reduced row echelon form's nonzero rows and their pivot columns."""
    m = [[v % p for v in r] for r in rows]
    rk, ncol, pivots = 0, len(m[0]) if m else 0, []
    for col in range(ncol):
        piv = next((r for r in range(rk, len(m)) if m[r][col]), None)
        if piv is None:
            continue
        m[rk], m[piv] = m[piv], m[rk]
        inv = pow(m[rk][col], p - 2, p)
        m[rk] = [(v * inv) % p for v in m[rk]]
        for r in range(len(m)):
            if r != rk and m[r][col]:
                f = m[r][col]
                m[r] = [(a - f * b) % p for a, b in zip(m[r], m[rk])]
        pivots.append(col)
        rk += 1
    return m[:rk], pivots


def rank_mod(rows, p):
    return len(rref_mod(rows, p)[1])


def coefficient_vectors(T, rng, a=(127, 1, 1)):
    """u_t, v_t: the coefficient vectors of w_t(j) and w_t(j + 1) over (c_1..c_T, y_lanes(j), y_lanes(j + 1))."""
    W = [[rng.choice([x for x in range(-127, 128) if x]) for _ in range(T)] for _ in range(2)]
    n = T + 2 * 3 * T
    def idx_lane(col, tau, ell):
        return T + col * 3 * T + 3 * tau + ell
    vecs = []
    for col in range(2):
        acc, out = [0] * n, []
        for tau in range(T):
            acc[tau] += W[col][tau]
            for ell in range(3):
                acc[idx_lane(col, tau, ell)] += 2 * a[ell]  # L = 2y + 1: the constant 1 drops out of the Hessian
            out.append(acc[:])
        vecs.append(out)
    return vecs[0], vecs[1], n


def hessian(pairs, n):
    H = [[0] * n for _ in range(n)]
    for u, v in pairs:
        for i in range(n):
            if u[i]:
                for k in range(n):
                    if v[k]:
                        H[i][k] += u[i] * v[k]
                        H[k][i] += u[i] * v[k]
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", default="4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for T in [int(s) for s in a.steps.split(",")]:
        u, v, n = coefficient_vectors(T, rng)
        cases = {
            "nh2": ([(u[t], v[t]) for t in range(T)], T, 2 * T),
            "nh": ([(u[2 * k], u[2 * k + 1]) for k in range(T // 2)], T // 2, T),
            "sumsq": ([(u[t], u[t]) for t in range(T)], T, T),
        }
        for name, (pairs, honest_products, words) in cases.items():
            H = hessian(pairs, n)
            basis, pivots = rref_mod(H, P)
            rp = len(pivots)
            free_in_span = [c for r, c in zip(basis, pivots) if sum(1 for v in r if v) == 1]
            r2 = rank_mod([[x % 2 for x in row] for row in H], 2)
            row = {"sketch": name, "T": T, "variables": n, "words": words, "honest_products": honest_products,
                   "rank_gfp": rp, "rank_gf2": r2, "products_lower_bound": (rp + 1) // 2,
                   "tight": (rp + 1) // 2 == honest_products, "coin_coordinates_in_colspace": len(free_in_span)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
