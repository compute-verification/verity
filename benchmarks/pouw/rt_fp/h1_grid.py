"""A control: H1's U keeps its co32c and cohalf words off FP4 at split 1, whatever the data.

H1's MMA operands carry U (co32c-u16: U = −16 + Σ w_p β_p, w = (1, 2, 4, 8, 17), so U ∈ {−16..−1, 1..16}
uniform, fresh a position of its period r = 32): the T words multiply P + c + U against X, and the D and O′ words
multiply U itself (benchmarks/pouw/theory/checked_strip.py, `co32c_words`). FP4 at split 1 needs every k-block of an
operand row on one scaled E2M1 grid, s·{0, ±½, ±1, ±3/2, ±2, ±3, ±4, ±6}, for some scale s the form allows: UE4M3 on
16-blocks (nvf4), UE8M0 on 32-blocks (mxf4). For each s, an entry a + U with a = P + c chosen by the cheater (any
integer in [−200, 200]) lands on the grid with probability at most p_s = max_a |grid_s ∩ (a + U)|/32, so a block fits
some scale with probability at most Σ_s p_s^B, and one m16n8k64 A operand (16 rows of 64/B blocks) at most that to the
power 16·64/B. Exact arithmetic throughout.

Run: `python benchmarks/pouw/rt_fp/h1_grid.py OUT`; it writes OUT/bound.txt.
"""
from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction as F
from pathlib import Path

E2M1 = [F(x, 2) for x in (0, 1, 2, 3, 4, 6, 8, 12)]
GRID = set(E2M1) | {-x for x in E2M1}
W = (1, 2, 4, 8, 17)
U = sorted({-16 + sum(w for w, b in zip(W, bits) if b) for bits in itertools.product((0, 1), repeat=len(W))})
A = range(-200, 201)


def ue4m3():
    out = set()
    for e, m in itertools.product(range(16), range(8)):
        if (e, m) != (15, 7):
            v = F(m, 8) * F(2) ** -6 if e == 0 else (1 + F(m, 8)) * F(2) ** (e - 7)
            if v:
                out.add(v)
    return sorted(out)


def bound(scales, B):
    total, worst, us = F(0), (None, F(0)), set(U)
    for s in scales:
        ints = [int(x) for x in (s * g for g in GRID) if x.denominator == 1]
        p = max(F(sum(x - a in us for x in ints), len(U)) for a in A)
        total += p ** B
        worst = max(worst, (s, p), key=lambda t: t[1])
    return total, worst


def main(out: Path):
    assert len(U) == 32 and 0 not in U and (min(U), max(U)) == (-16, 16)
    lines = []
    for name, scales, B in (("nvf4 (UE4M3 scales, 16-blocks)", ue4m3(), 16),
                            ("mxf4 (UE8M0 scales, 32-blocks)", [F(2) ** e for e in range(-127, 128)], 32)):
        total, (s, p) = bound(scales, B)
        tile = total ** (16 * 64 // B)
        exp = math.log10(tile.numerator) - math.log10(tile.denominator)
        lines.append(f"{name}: an entry on one grid w.p. <= {float(p):.4f} (at s = {s}); a block on some grid "
                     f"<= {float(total):.3e}; an m16n8k64 A operand <= 10^{exp:.0f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bound.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
