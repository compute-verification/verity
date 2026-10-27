#!/usr/bin/env python3
"""Two-stage steering against STEER-1 (Lemma C's premise).

The one-stage attack writes ≈ d/2 + 1 served values of the attention-output row wrong so that gate/up's C1′ operand,
round(127·γ⊙z/max|γ⊙z|) with z = h + o_out, lands on −R (gate/up's noise) and zeroes its words. Recomputing the producer
one stage back (SAMPLING's r = 1 closure) catches that. STEER-1 assumes steering through a consistent stage costs at
least what it saves. This checks the two-stage form: the cheater writes P·V's served row (the o_proj operand x, int8 with
a free dynamic scale s_x), o_proj is computed honestly (o_out = s_x·x·W_o), and asks whether some x lands z in the cone

  C = {z : round(127·γ_i z_i / max_k |γ_k z_k|) = −R_i for every i ≠ j},

with coordinate j the outlier (its product survives, as in the one-stage attack). The cone is scale-invariant, so a large s_x
makes h negligible. The construction is Babai rounding: x = round(λ·t·W⁻¹) for the cone's centre t, λ as large as the
int8 box allows. W_o is a stand-in (Gaussian, σ = 0.02, and a heavier-tailed variant): Llama's weights are not read.

  uv run python benchmarks/pouw/theory/steer2.py [--verdict PATH]
"""
from __future__ import annotations

import sys

import numpy as np
from exact_kit.checks import Checks


def quantize(z, gamma):
    u = gamma * z
    return np.rint(127.0 * u / np.abs(u).max()).astype(np.int64)


def attack(W, gamma, R, h, j, rng, polish=2000):
    t = -R * gamma[j] / (127.0 * gamma)
    t[j] = 1.0
    xs = np.linalg.solve(W.T, t)
    lam = 126.0 / np.abs(xs).max()
    x = np.rint(lam * xs)
    target = -R.copy()
    target[j] = 127

    def bad(xv):
        q = quantize(xv @ W, gamma)
        return np.flatnonzero(q != target)

    miss = bad(x)
    first = len(miss)
    # local polish: move one x entry by ±1 when it lowers the number of missed coordinates
    for _ in range(polish):
        if len(miss) == 0:
            break
        i = rng.choice(miss)
        k = int(np.argmax(np.abs(W[:, i]) * (np.abs(x) < 127)))
        best = len(miss)
        for step in (1, -1):
            x[k] += step
            m = bad(x)
            if len(m) < best:
                best, keep = len(m), step
            x[k] -= step
        if best < len(miss):
            x[k] += keep
            miss = bad(x)
    s_x = 1e6
    q_full = quantize(h + s_x * (x @ W), gamma)
    ok_full = bool(np.array_equal(np.delete(q_full, j), np.delete(target, j)))
    return first, len(miss), ok_full, float(np.abs(x).max()), float(lam)


def main() -> int:
    checks = Checks()
    rng = np.random.default_rng(20261017)
    for d in (1024, 4096):
        for wname in ("gaussian", "student-t3"):
            if wname == "gaussian":
                W = rng.normal(0, 0.02, (d, d))
            else:
                W = 0.02 * rng.standard_t(3, (d, d)) / np.sqrt(3.0)
            for gname in ("unit", "lognormal"):
                gamma = np.ones(d) if gname == "unit" else np.exp(rng.normal(-1.0, 0.5, d))
                R = rng.integers(0, 2, d).astype(np.float64)
                h = rng.normal(0, 1.0, d)
                j = int(np.argmax(gamma))
                first, last, ok_full, xmax, lam = attack(W, gamma, R, h, j, rng, polish=2000 if d <= 1024 else 300)
                checks.check(f"d={d} W {wname}, γ {gname}: Babai rounding plus polish finds no int8 attention-output row "
                             f"that steers gate/up's operand onto −R through an honest o_proj (STEER-1 not refuted by it)",
                             last > 0 and not ok_full,
                             f"misses {first} of {d - 1} coordinates, {last} after polish; with h at s_x = 1e6: "
                             f"{ok_full}; max|x| = {xmax:.0f}, λ = {lam:.3g} (the int8 box leaves λ·t·W⁻¹ this much room)")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
