"""The reporting grid every monitoring-period report states: the monitoring periods a result is presented for (`PERIODS`: a
week, a month and a year, each with its length in seconds) and the δs its bounds are stated at (`DELTAS`: 1/100 and
10⁻⁶). PoUS's wipe table, PoUB's session table and the block record each pool their figures over every period at every
δ. Stdlib only."""

from __future__ import annotations

from fractions import Fraction

#: The monitoring periods a result is presented for, each with its length in seconds.
PERIODS: tuple[tuple[str, int], ...] = (("week", 7 * 86400), ("month", 30 * 86400), ("year", 365 * 86400))
#: The δs a period's bounds are stated at.
DELTAS: tuple[Fraction, ...] = (Fraction(1, 100), Fraction(1, 10 ** 6))

__all__ = ["DELTAS", "PERIODS"]
