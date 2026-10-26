"""Property (i) at every admissible data value: the int8 overflow rate of Strassen-level sums of wrapped operands.

An operand entry is wrap8(d + e + N), with d the data (any int8, the adversary's), e a coin bit and N uniform on ±u, and
the entries of one segment independent (N's columns differ within a segment). For a sum of t entries with signs, this
prints the minimum over all data values (d_1, .., d_t) of the probability that the sum leaves [-128, 127], computed
exactly from the distributions (no sampling). t = 2 is a depth-1 Strassen form, t = 4 a depth-2 one.
"""

from __future__ import annotations

import itertools

import numpy as np


def dist(d: int, u: int) -> np.ndarray:
    """P[wrap8(d + e + N) = v] for v = -128..127, e a fair bit, N uniform on -u..u."""
    p = np.zeros(256)
    for e in (0, 1):
        for nv in range(-u, u + 1):
            p[(d + e + nv + 128) % 256] += 0.5 / (2 * u + 1)
    return p


def overflow(ps: list[np.ndarray], signs: tuple[int, ...]) -> float:
    s = np.array([1.0])
    lo = 0
    for p, sg in zip(ps, signs, strict=True):
        q = p if sg > 0 else p[::-1]
        s = np.convolve(s, q)
        lo += -128 if sg > 0 else -127
    vals = np.arange(lo, lo + len(s))
    return float(s[(vals < -128) | (vals > 127)].sum())


def main() -> None:
    for u in (64, 96, 112, 127):
        D = {d: dist(d, u) for d in range(-128, 128)}
        two = min((overflow([D[a], D[b]], sg), a, b, sg) for a in range(-128, 128, 1) for b in range(-128, 128, 4)
                  for sg in ((1, 1), (1, -1)))
        grid = range(-128, 128, 16)
        four = min((overflow([D[a] for a in ds], (1, 1, 1, 1)), ds) for ds in itertools.combinations_with_replacement(grid, 4))
        zero2 = overflow([D[0], D[0]], (1, 1))
        zero4 = overflow([D[0]] * 4, (1, 1, 1, 1))
        print(f"u = {u:3d} (theta {u / 128:.2f}): two-term overflow at d = 0 {zero2:.4f}, min over data {two[0]:.4f} "
              f"at {two[1:]}; four-term at d = 0 {zero4:.4f}, min over a 16-step grid {four[0]:.4f} at {four[1]}")


if __name__ == "__main__":
    main()
