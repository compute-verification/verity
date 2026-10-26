"""How full-width are Pearl-I8's formed operands at the adversary's A = 0, and how often does a Strassen-level sum leave int8?

At A = 0 the formed operand is X̃ = clamp(rnd(E·Fᵀ / 2^s)), pure noise. 4b's sparse correction is cheap when a two-term
sum X̃_1 ± X̃_2 leaves [−128, 127] rarely. Prints, per parameter set: the noise σ in grid units, the share of entries at
|X̃| >= 64 and at the clamp, and the out-of-int8 rate of two- and four-term sums; then the same for uniform noise on
±128θ, which is what "full-width" would need.
"""

from __future__ import annotations

import numpy as np

from pearl_i8 import Params, coins, form


def sums_out(x: np.ndarray, terms: int, rng) -> float:
    flat = x.ravel()
    s = sum(rng.permutation(flat) * rng.choice([-1, 1], size=flat.size) for _ in range(terms))
    return float(np.mean((s < -128) | (s > 127)))


def main() -> None:
    rng = np.random.default_rng(7)
    m, k = 256, 1024
    for p in (Params(), Params(r=64, s=12, q_d=84), Params(r=32, s=10, q_d=6)):
        x, _, n = form(np.zeros((m, k), dtype=np.int64), coins(b"occ", "E", m, p.r), coins(b"occ", "F", k, p.r), p)
        print(f"lowrank r={p.r} s={p.s} q_d={p.q_d}: sigma_grid={float(np.std(n)) / 2 ** p.s:.1f} "
              f"P|X|>=64={np.mean(np.abs(x) >= 64):.4f} clamp={np.mean(np.abs(x) == 127):.5f} "
              f"out2={sums_out(x, 2, rng):.2e} out4={sums_out(x, 4, rng):.2e}")
    for theta in (0.5, 0.625, 0.75, 0.875, 1.0):
        x = rng.integers(-int(127 * theta), int(127 * theta) + 1, size=(m, k))
        print(f"uniform theta={theta}: out2={sums_out(x, 2, rng):.3f} out4={sums_out(x, 4, rng):.3f} "
              f"data bits={np.log2(2 * 127 * (1 - theta) + 1):.2f}")


if __name__ == "__main__":
    main()
