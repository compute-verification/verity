"""verity.resource_accounting: what accounted resource claims follow from verification results.

`IntegrityAllowance` is the claim sampled proofs passes to its consumers (item 27): PoUW's work claim, a KV usefulness
claim, anti-exfiltration policies.  Each consumer evaluates its own worst case over it without importing the protocol
that produced it.  It carries the two-stage law (decision 43) as data, because one number cannot serve consumers that
different strategies maximise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb, inf, log


@dataclass(frozen=True, slots=True)
class IntegrityAllowance:
    """What an accepted transcript still admits.

    Each of `n_r` replay units holds `n_v` verification units; each replay unit was selected with probability `p`, and
    then exactly `k` of its verification units were checked.  A strategy leaving m_i incorrect verification units in
    replay unit i is accepted with probability prod_i exp(-rho(m_i)), so except with probability `delta` an accepted
    transcript satisfies sum_i rho(m_i) < ln(1/delta).

    It constrains what the adversary did; it is not an amount.  A Bernoulli law over single units is n_v = k = 1."""

    n_r: int
    n_v: int
    p: Fraction
    k: int
    delta: float

    def __post_init__(self) -> None:
        if self.n_r < 0 or self.n_v < 1 or not 0 <= self.k <= self.n_v:
            raise ValueError("an allowance needs n_r >= 0, n_v >= 1 and 0 <= k <= n_v")
        if not 0 <= Fraction(self.p) <= 1 or not 0 < self.delta < 1:
            raise ValueError("an allowance needs 0 <= p <= 1 and 0 < delta < 1")

    def accept(self, m: int) -> Fraction:
        """Exact probability that one replay unit with `m` incorrect verification units passes."""
        if not 0 <= m <= self.n_v:
            raise ValueError("0 <= m <= n_v")
        return 1 - Fraction(self.p) + Fraction(self.p) * Fraction(comb(self.n_v - m, self.k), comb(self.n_v, self.k))

    def rho(self, m: int) -> float:
        """Detection risk -ln accept(m)."""
        a = self.accept(m)
        return inf if a == 0 else -log(a)

    def worst_case(self, utility: Callable[[int], float]) -> float:
        """An upper bound on sum_i utility(m_i) over the strategies the allowance admits, for utility(0) = 0.

        It is the optimum of the linear relaxation: choose x_m >= 0 replay units at each m in 1..n_v with
        sum x_m <= n_r and sum x_m rho(m) <= ln(1/delta).  Two constraints, so an optimal vertex uses at most two
        values of m, and every pair is tried."""
        budget = log(1 / self.delta)
        options = [(float(u), self.rho(m)) for m in range(1, self.n_v + 1) if (u := utility(m)) > 0]
        options = [(u, r) for u, r in options if r < inf]
        if not options:
            return 0.0
        best = max(u * (self.n_r if r == 0 else min(self.n_r, budget / r)) for u, r in options)
        for (ua, ra), (ub, rb) in combinations(sorted(options, key=lambda o: o[1]), 2):
            if rb > ra and ra * self.n_r < budget < rb * self.n_r:
                xb = (budget - self.n_r * ra) / (rb - ra)
                best = max(best, ua * (self.n_r - xb) + ub * xb)
        return best

    def incorrect_verification_units(self) -> float:
        """PoUW's consumer: incorrect verification units, u(m) = m.  Peaks at fully incorrect replay units."""
        return self.worst_case(lambda m: m)

    def forgeable_tiles(self, tiles_per_vu: int) -> float:
        """PoUW's forgeable tiles when each verification unit holds `tiles_per_vu` tiles, u(m) = m * tiles_per_vu.

        `tiles_per_vu` must be the partition sampled proofs used: a smaller value under-counts silently."""
        if tiles_per_vu < 1:
            raise ValueError("tiles_per_vu >= 1")
        return self.worst_case(lambda m: m * tiles_per_vu)

    def touched_replay_units(self) -> float:
        """The KV consumer: replay units with any incorrect verification unit, u(m) = 1.  Peaks at m = 1."""
        return self.worst_case(lambda m: 1)

    def to_json(self) -> dict:
        return {"n_r": self.n_r, "n_v": self.n_v, "p": str(Fraction(self.p)), "k": self.k, "delta": self.delta}


__all__ = ["IntegrityAllowance"]
