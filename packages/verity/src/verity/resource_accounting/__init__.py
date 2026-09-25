"""verity.resource_accounting: what accounted resource claims follow from verification results.

`IntegrityProfile` summarises what has been verified about a circuit's evaluation: the families established on every
audit, the two-stage sampling law (decision 43) and the failure probability.  Sampled proofs produces it, and each
consumer (PoUW's work claim, a KV usefulness claim, anti-exfiltration policies) reads what it needs off it without
importing the protocol that produced it.  The worst-case readers below are consumer-side: each takes the profile and
the consumer's own utility, because different consumers are maximised by different strategies over the same law.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import comb, inf, log


@dataclass(frozen=True, slots=True)
class IntegrityProfile:
    """What an accepted transcript establishes about the evaluation.

    `covered`: the families proved on every audit, outside both draws; their faults survive with probability 0.
    The law: each of `n_r` replay units holds `n_v` verification units, was selected with probability `p`, and then
    had exactly `k` of its verification units checked.  A strategy leaving m_i incorrect uncovered verification units in
    replay unit i is accepted with probability prod_i exp(-rho(m_i)), so except with probability `delta` an accepted
    transcript satisfies sum_i rho(m_i) < ln(1/delta).  A Bernoulli law over single units is n_v = k = 1."""

    n_r: int
    n_v: int
    p: Fraction
    k: int
    delta: float
    covered: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.n_r < 0 or self.n_v < 1 or not 0 <= self.k <= self.n_v:
            raise ValueError("a profile needs n_r >= 0, n_v >= 1 and 0 <= k <= n_v")
        if not 0 <= Fraction(self.p) <= 1 or not 0 < self.delta < 1:
            raise ValueError("a profile needs 0 <= p <= 1 and 0 < delta < 1")
        object.__setattr__(self, "covered", frozenset(self.covered))

    def accept(self, m: int) -> Fraction:
        """Exact probability that one replay unit with `m` incorrect verification units passes."""
        if not 0 <= m <= self.n_v:
            raise ValueError("0 <= m <= n_v")
        return 1 - Fraction(self.p) + Fraction(self.p) * Fraction(comb(self.n_v - m, self.k), comb(self.n_v, self.k))

    def rho(self, m: int) -> float:
        """Detection risk -ln accept(m)."""
        a = self.accept(m)
        return inf if a == 0 else -log(a)

    def to_json(self) -> dict:
        return {"covered": sorted(self.covered), "n_r": self.n_r, "n_v": self.n_v, "p": str(Fraction(self.p)), "k": self.k,
                "delta": self.delta}


def worst_case(profile: IntegrityProfile, utility: Callable[[int], float]) -> float:
    """An upper bound on sum_i utility(m_i) over the strategies `profile` admits, for utility(0) = 0.

    It is the optimum of the linear relaxation: choose x_m >= 0 replay units at each m in 1..n_v with sum x_m <= n_r
    and sum x_m rho(m) <= ln(1/delta).  Two constraints, so an optimal vertex uses at most two values of m, and every
    pair is tried."""
    budget = log(1 / profile.delta)
    options = [(float(u), profile.rho(m)) for m in range(1, profile.n_v + 1) if (u := utility(m)) > 0]
    options = [(u, r) for u, r in options if r < inf]
    if not options:
        return 0.0
    n_r = profile.n_r
    best = max(u * (n_r if r == 0 else min(n_r, budget / r)) for u, r in options)
    for (ua, ra), (ub, rb) in combinations(sorted(options, key=lambda o: o[1]), 2):
        if rb > ra and ra * n_r < budget < rb * n_r:
            xb = (budget - n_r * ra) / (rb - ra)
            best = max(best, ua * (n_r - xb) + ub * xb)
    return best


def incorrect_verification_units(profile: IntegrityProfile) -> float:
    """PoUW's reader: incorrect verification units, u(m) = m.  Peaks at fully incorrect replay units."""
    return worst_case(profile, lambda m: m)


def forgeable_tiles(profile: IntegrityProfile, tiles_per_vu: int) -> float:
    """PoUW's reader when each verification unit holds `tiles_per_vu` tiles, u(m) = m * tiles_per_vu.

    `tiles_per_vu` must be the partition sampled proofs used: a smaller value under-counts silently."""
    if tiles_per_vu < 1:
        raise ValueError("tiles_per_vu >= 1")
    return worst_case(profile, lambda m: m * tiles_per_vu)


def touched_replay_units(profile: IntegrityProfile) -> float:
    """The KV reader: replay units with any incorrect verification unit, u(m) = 1.  Peaks at m = 1."""
    return worst_case(profile, lambda m: 1)


__all__ = ["IntegrityProfile", "forgeable_tiles", "incorrect_verification_units", "touched_replay_units", "worst_case"]
