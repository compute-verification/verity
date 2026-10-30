"""Forced work of E·Rᵀ per MAC, in W1 units (one dense int8/FP8 MAC = 1), over the known exact routes, for E and R
uniform on n_E and n_R consecutive integers. Estimates on recalled instruction rates, not measurements.

H200 (sm_90a): b1 BGMMA at 8× s8 (AND-popc; XOR emulated), so P_E·P_R/8 with P = ⌈log₂ n⌉ binary planes a side
(offsets are row and column sums); 2:4 sparse int8/FP8 at 2× logical has no b1 form and costs more than a b1 plane
whenever P ≤ 4. FP8 runs at the s8 rate; no FP4, no s4.

RTX PRO 6000 (sm_120): no b1. kind::mxf4nvf4 (E2M1 × E2M1, block-scaled) at 2× FP8, and its 2:4 sparse form at 4×
logical. E2M1 times a power-of-two (or UE4M3) scale holds D = {0, ±1, ±2, ±3, ±4, ±6}: an alphabet fits one plane iff
an affine image of it lies in D, i.e. n ≤ 9 consecutive integers. Larger alphabets take a main plane plus a residual:
two dense planes (0.5 each), or a dense plane plus one 2:4-sparse pass (0.25) over a residual of density ρ, whose
overflow (groups of four with three or four nonzeros) goes to FP32 lanes at 8 units a MAC (128 lanes against 1024).

Run: python iter_price.py [--json OUT]
"""

import argparse
import json
import math

D = {0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6}
FFMA_UNITS = 8.0  # sm_120: 1024 FP8 MACs against 128 FP32 lanes an SM-clock


def planes(n):
    return max(0, math.ceil(math.log2(n))) if n > 1 else 0


def var(n):
    return (n * n - 1) / 12


def cover(n):
    """Most of n consecutive integers an affine image of D (integer scale α ≥ 1) covers."""
    best = 0
    for alpha in range(1, 8):
        img = sorted(alpha * d for d in D)
        for lo in range(img[0] - n, img[-1] + 1):
            best = max(best, sum(1 for v in img if lo <= v < lo + n))
    return min(best, n)


def overflow(rho):
    p3 = 4 * rho ** 3 * (1 - rho)
    p4 = rho ** 4
    return (p3 + 2 * p4) / 4


def side_factor(n):
    """Cost of one side's representation, in dense E2M1 planes: q planes of base-9 digits in [−4, 4] cover 9^q
    consecutive integers; the last plane may instead be a 2:4-sparse residual pass."""
    if n <= 1:
        return 0.0
    if n <= 9:
        return 1.0
    q = 1
    while 9 ** (q + 1) < n:
        q += 1
    rho = 1 - cover(n) / n if q == 1 else 0.5
    sparse = (0.25 + FFMA_UNITS * overflow(rho)) / 0.5
    return q + min(1.0, sparse)


def h200(ne, nr):
    return min(1.0, planes(ne) * planes(nr) / 8)


def sm120(ne, nr):
    return min(1.0, 0.5 * side_factor(ne) * side_factor(nr))


def mod_bits(nr):
    """b with 2^b the largest power of two dividing n_R: (A + R) mod 2^b is then uniform for every A."""
    return (nr & -nr).bit_length() - 1


def h200_mod(ne, b):
    """b1 planes of E·Lᵀ mod 2^b, L uniform mod 2^b: only plane pairs i + j < b reach the low b bits."""
    return sum(1 for i in range(planes(ne)) for j in range(b) if i + j < b) / 8


def mod_side(b):
    """E2M1 planes for a uniform residue mod 2^b: the cheater picks any representative, so one plane with scale α
    covers {α·d mod 2^b : d ∈ D}; base-9 digits cover 9^q residues with q planes."""
    m = 1 << b
    if m <= 1:
        return 0.0
    cov = max(len({(a * d) % m for d in D}) for a in range(1, m, 2))
    if cov == m:
        return 1.0
    q = 1
    while 9 ** (q + 1) < m:
        q += 1
    rho = 1 - cov / m if q == 1 else 0.5
    return q + min(1.0, (0.25 + FFMA_UNITS * overflow(rho)) / 0.5)


def sm120_mod(ne, b):
    return 0.5 * side_factor(ne) * mod_side(b)


def row(ne, nr):
    b = mod_bits(nr)
    return {"n_e": ne, "n_r": nr, "bits_e": round(math.log2(ne), 2), "bits_r": round(math.log2(nr), 2),
            "var_e": var(ne), "var_r": var(nr), "planes_b1": [planes(ne), planes(nr)],
            "h200_units": h200(ne, nr), "h200_b1_raw": planes(ne) * planes(nr) / 8,
            "sm120_fp4": [round(side_factor(ne), 3), round(side_factor(nr), 3)],
            "sm120_units": round(sm120(ne, nr), 3), "sm120_raw": round(0.5 * side_factor(ne) * side_factor(nr), 3),
            "forced": round(min(h200(ne, nr), sm120(ne, nr)), 3),
            # every-A bound from the low b bits alone (average-case premise, no smoothing)
            "mod_b": b, "h200_mod_raw": h200_mod(ne, b), "sm120_mod_raw": round(sm120_mod(ne, b), 3),
            "forced_mod": round(min(1.0, h200_mod(ne, b), sm120_mod(ne, b)), 3)}


PAIRS = [(2, 2), (2, 4), (2, 16), (3, 3), (3, 9), (3, 10), (4, 16), (3, 16), (3, 17), (3, 19), (3, 22), (3, 32),
         (3, 64), (5, 19), (5, 32), (10, 10), (19, 3), (17, 3), (2, 129), (2, 256)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [row(*p) for p in PAIRS]
    for r in rows:
        print(json.dumps(r))
    if a.json:
        open(a.json, "w").write(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
