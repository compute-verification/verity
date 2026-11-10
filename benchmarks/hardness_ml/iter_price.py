"""Forced work of E·Rᵀ per MAC, in W1 units (one dense int8/FP8 MAC = 1), over the exact plane and format routes, for
E and R uniform on n_E and n_R consecutive integers. Estimates on recalled instruction rates, not measurements. Fast
bilinear routes (exact Strassen in s8) are iter_strassen.py's; `--strassen` folds them in, and without them every
`forced` here is only a ceiling.

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

import numpy as np

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


def row(ne, nr, strassen=None):
    """`strassen`: iter_strassen.py's JSON. Its `attack` (exact Strassen in s8, both chips' s8 MMA) joins the plane
    routes in `credit_attack`, what an A = 0 cheater pays; `credit_floor` is what the average-case premise on the
    uniform mod-2^b law proves for every A, at most the attack (the same trees over Z/2^b count the same)."""
    b = mod_bits(nr)
    forced = min(h200(ne, nr), sm120(ne, nr))
    forced_mod = min(1.0, h200_mod(ne, b), sm120_mod(ne, b))
    out = {"n_e": ne, "n_r": nr, "bits_e": round(math.log2(ne), 2), "bits_r": round(math.log2(nr), 2),
           "var_e": var(ne), "var_r": var(nr), "planes_b1": [planes(ne), planes(nr)],
           "h200_units": h200(ne, nr), "h200_b1_raw": planes(ne) * planes(nr) / 8,
           "sm120_fp4": [round(side_factor(ne), 3), round(side_factor(nr), 3)],
           "sm120_units": round(sm120(ne, nr), 3), "sm120_raw": round(0.5 * side_factor(ne) * side_factor(nr), 3),
           "forced": round(forced, 3),
           # every-A bound from the low b bits alone (average-case premise, no smoothing)
           "mod_b": b, "h200_mod_raw": h200_mod(ne, b), "sm120_mod_raw": round(sm120_mod(ne, b), 3),
           "forced_mod": round(forced_mod, 3)}
    att = (strassen or {}).get(f"e{ne}r{nr}_attack")
    if att is not None:
        out["strassen"] = {k: round(v, 3) for k, v in att.items()}
        out["credit_attack"] = round(min(forced, att["fast"]), 3)
        out["credit_floor"] = round(min(forced_mod, att["fast"]) if b else 0.0, 3)
    k8 = (strassen or {}).get(f"e{ne}r{nr}_attack_k8")
    if k8 is not None:
        out["credit_attack_k8"] = {k: round(v, 3) for k, v in k8.items()}
    return out


def rounding_rho(n, grid):
    """Least ‖R − round(R)‖/‖R‖ for R uniform on n consecutive integers rounded to the nearest point of an affine image
    o + α·grid (integer α ≥ 1, o on R's integer lattice, since A + R stays integer): the post-draw move (D1's ball)
    that puts R on that grid."""
    vals = np.arange(n) - (n - 1) / 2
    best = math.inf
    span = max(grid) - min(grid)
    for alpha in range(1, 2 * n // max(1, span) + 3):
        pts = np.array(sorted(alpha * g for g in grid), dtype=float)
        for o2 in range(int(2 * (vals[0] - pts[-1])) - 2, int(2 * (vals[-1] - pts[0])) + 3):
            if o2 % 2 != (n - 1) % 2:
                continue
            p = pts + o2 / 2
            i = np.clip(np.searchsorted(p, vals), 1, len(p) - 1)
            err = np.minimum((vals - p[i - 1]) ** 2, (vals - p[i]) ** 2).mean()
            best = min(best, float(err))
    return math.sqrt(best / var(n))


def ball(ne, nr):
    """Forced work left after a post-draw move of relative size ρ: the cheapest representation R can be rounded onto
    within ρ, by route. b1 with P planes takes any progression of 2^P values; E2M1 one plane takes o + α·D."""
    out = []
    for p in range(1, planes(nr)):
        out.append({"route": f"b1 {p} planes", "rho": round(rounding_rho(nr, list(range(1 << p))), 4),
                    "h200": min(1.0, planes(ne) * p / 8)})
    if nr > 9:
        out.append({"route": "E2M1 one plane", "rho": round(rounding_rho(nr, sorted(D)), 4),
                    "sm120": min(1.0, 0.5 * side_factor(ne))})
    return out


PAIRS = [(2, 2), (2, 4), (2, 16), (3, 3), (3, 9), (3, 10), (4, 16), (3, 16), (3, 17), (3, 19), (3, 22), (3, 32),
         (3, 64), (5, 16), (5, 19), (5, 32), (10, 10), (10, 16), (19, 3), (17, 3), (2, 129), (2, 256)]
BALL_PAIRS = [(3, 19), (3, 32), (10, 16), (2, 256)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--strassen", default=None, help="iter_strassen.py's JSON")
    a = ap.parse_args()
    st = json.load(open(a.strassen)) if a.strassen else None
    rows = [row(*p, strassen=st) for p in PAIRS]
    for r in rows:
        print(json.dumps(r))
    balls = {f"{ne},{nr}": ball(ne, nr) for ne, nr in BALL_PAIRS}
    for k, v in balls.items():
        print(json.dumps({"ball": k, "routes": v}))
    if a.json:
        open(a.json, "w").write(json.dumps({"pairs": rows, "ball": balls}, indent=1))


if __name__ == "__main__":
    main()
