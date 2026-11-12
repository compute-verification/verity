"""The reporting grid every monitoring-period report states: the monitoring periods a result is presented for (`PERIODS`: a
week, a month and a year, each with its length in seconds) and the δs its bounds are stated at (`DELTAS`: 1/100 and
10⁻⁶). PoUS's wipe table, PoUB's session table and the block record each pool their figures over every period at every
δ.

And the one rule for a row's error. A row's error is the sum of the errors of everything the row covers: every check
of its period on every machine of its scope (`summed`). A sum of 1 or more bounds nothing, so the row prints `NO_BOUND`
with that sum in place of every amount (`no_bound`), its `free` and its `verified` included; the error itself is still
printed. Stdlib only."""

from __future__ import annotations

from fractions import Fraction

#: The monitoring periods a result is presented for, each with its length in seconds.
PERIODS: tuple[tuple[str, int], ...] = (("week", 7 * 86400), ("month", 30 * 86400), ("year", 365 * 86400))
#: The δs a period's bounds are stated at.
DELTAS: tuple[Fraction, ...] = (Fraction(1, 100), Fraction(1, 10 ** 6))
#: What a row prints in place of every amount where its error is 1 or more.
NO_BOUND = "no bound at this period"


def summed(error: Fraction | float | int, checks: int = 1) -> Fraction:
    """The error of a row that covers `checks` checks alike, each failing with at most `error`: their sum, exactly (a
    float counts at its exact value)."""
    return Fraction(error) * checks


def no_bound(error: Fraction | float | int) -> str | None:
    """Why a row whose error is `error` (`summed`) states no amount: None while the error is under 1, and at 1 or more
    `NO_BOUND` with the error, which the row prints in place of every amount."""
    return None if error < 1 else f"{NO_BOUND}: the row's error, {float(error):.4g}, is 1 or more"


__all__ = ["DELTAS", "NO_BOUND", "PERIODS", "no_bound", "summed"]
