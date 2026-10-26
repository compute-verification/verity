"""How wide are a protocol's tensor-core operands, for P1 (CMP-118)? Depth-1 Strassen needs the sum or difference of two
operand entries to fit the MMA's format; an entry that doesn't must be split (two limbs) or fixed by a sparse
correction. This prints, per operand distribution, the share of Strassen's pair sums and differences outside int8, and
the relative tensor-core cost of one level with one wide side,

    cost = (7 + k · (s - 1)) / 8,   k = 5 products whose factor on that side is a pair, s = 2 limbs when dense,

taking a correction as "sparse" (s = 1, its share charged as ALU work at 64 MAC-times an entry) when the share is
below 1%. Exact for the integer distributions (convolution); Gaussian and Laplace activations at σ = 25 of 127 (absmax
int8 of real activations) by their exact discretized pmf.

    uv run python benchmarks/pouw/accel_score/p1_width.py
"""
from __future__ import annotations

import math

LO, HI = -128, 127


def pmf_uniform(a: int, b: int) -> dict[int, float]:
    return {x: 1 / (b - a + 1) for x in range(a, b + 1)}


def pmf_disc(cdf, sigma: float) -> dict[int, float]:
    p = {x: cdf((x + 0.5) / sigma) - cdf((x - 0.5) / sigma) for x in range(-126, 127)}
    p[-127] = cdf(-126.5 / sigma)
    p[127] = 1 - cdf(126.5 / sigma)
    return p


def gauss_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def laplace_cdf(z: float) -> float:
    b = 1 / math.sqrt(2)
    return 0.5 * math.exp(z / b) if z < 0 else 1 - 0.5 * math.exp(-z / b)


def outside(p: dict[int, float], sign: int) -> float:
    """P(x + sign·y outside int8) for x, y independent with pmf p."""
    out = 0.0
    for x, px in p.items():
        for y, py in p.items():
            s = x + sign * y
            if s < LO or s > HI:
                out += px * py
    return out


def cost(share: float) -> float:
    if share < 0.01:
        return 7 / 8 + 5 / 8 * share * 64 / 32
    return (7 + 5) / 8


DISTS = {
    "Dither R (bits) at A = 0": {0: 0.5, 1: 0.5},
    "U in ±64 (PouwU129) at B = 0": {x: 1 / 129 for x in range(-64, 65)},
    "H1 W' = c + U at B = 0": {x: sum(1 / 129 * 0.5 for c in (0, 1) if -64 <= x - c <= 64) for x in range(-64, 66)},
    "real activations, Gaussian σ = 25": pmf_disc(gauss_cdf, 25.0),
    "real activations, Laplace σ = 25": pmf_disc(laplace_cdf, 25.0),
    "full-width uniform int8 noise": pmf_uniform(-128, 127),
    "noise uniform ±96 (data left ±31)": pmf_uniform(-96, 96),
}


def main() -> None:
    print(f"{'operand':48s} {'P(x+y out)':>11s} {'P(x-y out)':>11s} {'depth-1 cost, that side wide':>29s}")
    for name, p in DISTS.items():
        assert abs(sum(p.values()) - 1) < 1e-9, name
        plus, minus = outside(p, 1), outside(p, -1)
        print(f"{name:48s} {plus:11.2e} {minus:11.2e} {cost(max(plus, minus)):29.3f}")


if __name__ == "__main__":
    main()
