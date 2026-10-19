"""Hashed seeds' grinding premium (GRIND, decision 112; `Accounting.Guarantees.TS.GrindPremiumAt`), where a row's noise is
visible once its seed is drawn: the formula `circuit.compile`'s seed-group check (decision 122) and
`benchmarks/pouw/grind_price.py`'s table share.

A unit's seed is SHA-512(tag ‖ salt ‖ position ‖ root(A) ‖ root(B)) with a live salt, so a cheater with T hash queries in a
salt's epoch draws T seeds for a seed group and keeps the best. In H1's attention forms one operand row whose noise is
matched collapses the floor of every unit reading it, and that is visible after the seed, so the cheater picks the matched
rows: over a group's rows, N of them of b noise bits each read by u units, each row's moment is 1 − 2^{-b} + 2^{-b}e^{θu}
(`LuckMoment` at κ = 0, multiplied over rows by `PrivateMoment`), and outside δ the group realizes at most

    π* = min over θ of (ln(T/δ) + Σ N·ln(1 − 2^{-b} + 2^{-b}·e^{θu})) / θ

units. The grind term is π* over the group's units; compile refuses a group whose term exceeds `FLAG` at `T_BITS`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

LN2 = math.log(2)
DELTA_BITS = 40                      # δ = 2⁻⁴⁰, the grind's own failure probability (outside ε)
T_BITS = 128                         # the cheater's hash queries a salt's epoch, at which compile checks a seed group
FLAG = 1e-3                          # a grind term over 0.1% is flagged, and compile refuses its seed group


def ln_t_over_delta(t_bits: int) -> float:
    return (t_bits + DELTA_BITS) * LN2


def visible_premium(rows: Iterable[tuple[float, float, float]], t_bits: int) -> float:
    """π* in units for visible matched rows: min over θ of (ln(T/δ) + Σ N ln(1 − 2^{-b} + 2^{-b} e^{θu}))/θ."""
    rows = list(rows)
    lt = ln_t_over_delta(t_bits)

    def obj(theta: float) -> float:
        s = lt
        for n, b, u in rows:
            x, lq = theta * u, -b * LN2
            a, d = math.log1p(-math.exp(lq)), x + lq
            m = max(a, d)
            s += n * (m + math.log(math.exp(a - m) + math.exp(d - m)))
        return s / theta

    grid = [10 ** (e / 20) for e in range(-160, 41)]
    i = min(range(len(grid)), key=lambda j: obj(grid[j]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(100):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (lo, b) if obj(a) < obj(b) else (a, hi)
    return obj((lo + hi) / 2)


__all__ = ["DELTA_BITS", "FLAG", "LN2", "T_BITS", "ln_t_over_delta", "visible_premium"]
