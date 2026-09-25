"""verity.resource_accounting: what accounted resource claims follow from verification results.

`IntegrityAllowance` is the one result shape that crosses between protocols today: the integrity protocol produces it,
and consumers (PoUW's work claim; later PoUS's KV claim, anti-exfiltration policies) evaluate their own worst case
over it without importing the protocol that produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import comb, inf, log


@dataclass(frozen=True, slots=True)
class IntegrityAllowance:
    """Except with probability `delta`, the incorrect-unit counts m_1..m_{n_r} of an accepted transcript satisfy
    sum_i rho(m_i) < ln(1/delta), where each of the `n_r` replay units was selected with probability `p` and then
    `k` of its `n_v` verification units were checked.

    It is a constraint on what the adversary did, not an amount: different consumers are maximised by different
    strategies over the same set, so the law travels with it and each consumer supplies its own utility."""

    n_r: int
    n_v: int
    p: Fraction
    k: int
    delta: float

    def rho(self, m: int) -> float:
        """Detection risk -ln Pr[accept] of one replay unit with m incorrect verification units."""
        miss = comb(self.n_v - m, self.k) / comb(self.n_v, self.k)
        accept = 1 - self.p + self.p * miss
        return inf if accept <= 0 else -log(accept)

    def worst_case(self, utility: Callable[[int], float]) -> float:
        """An upper bound on sum_i utility(m_i) over admitted strategies, for utility(0) = 0.

        sum u(m_i) <= max_m (u(m) / rho(m)) * sum rho(m_i) < max_m (u(m) / rho(m)) * ln(1/delta), and never more than
        every replay unit at its best m."""
        ms = range(1, self.n_v + 1)
        per_risk = max(utility(m) / r if 0 < (r := self.rho(m)) < inf else (0.0 if r else inf) for m in ms)
        return min(per_risk * log(1 / self.delta), self.n_r * max(utility(m) for m in ms))


__all__ = ["IntegrityAllowance"]
