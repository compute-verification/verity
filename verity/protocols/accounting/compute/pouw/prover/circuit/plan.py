"""The sample size at the default report point: `record_k(ε, δ)`, the least K with (1 − ε)^K ≤ δ, and `RECORD_K`, its value
at ε = 0.1% and δ = 2⁻⁴⁰."""
from __future__ import annotations

from decimal import ROUND_CEILING, Decimal, localcontext
from fractions import Fraction


def record_k(epsilon: Fraction, delta: Fraction) -> int:
    """The smallest K with (1 − ε)^K ≤ δ, decided in exact integers: incorrect proof blocks doing a fraction ε of the work escape the
    work sampling design's sample with at most (1 − ε)^K (`Spot.Sampling.Work.work_escape_le`). At ε = 0.1% and δ = 2⁻⁴⁰ it is 27,713;
    27,712 misses (`record_sizing`, `record_sizing_one_fewer`). That ε at δ is the default point the profile reports at, not a
    policy parameter: PoUW has none."""
    eps, dlt = Fraction(epsilon), Fraction(delta)
    if not (0 < eps < 1 and 0 < dlt < 1):
        raise ValueError("0 < ε < 1 and 0 < δ < 1")
    q = 1 - eps

    def meets(k: int) -> bool:                              # q^k <= δ, cross-multiplied
        return q.numerator ** k * dlt.denominator <= dlt.numerator * q.denominator ** k

    with localcontext() as cx:                              # the start, ⌈ln δ / ln(1 − ε)⌉, in 60-digit decimals
        cx.prec = 60
        ln = lambda f: (Decimal(f.numerator).ln() - Decimal(f.denominator).ln())  # noqa: E731
        k = max(1, int((ln(dlt) / ln(q)).to_integral_value(rounding=ROUND_CEILING)) - 2)
    while not meets(k):
        k += 1
    while k > 1 and meets(k - 1):
        k -= 1
    return k


#: the tile samples at the default report point, ε = 0.1% at δ = 2⁻⁴⁰
RECORD_K = 27_713

__all__ = ["RECORD_K", "record_k"]
