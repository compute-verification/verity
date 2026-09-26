"""`IntegrityProfile`: what verification established about the evaluation of a transcript of Programs.

An integrity protocol (sampled proofs) produces it.  Resource accounting, policies and PoUW read what they need off it
without importing the protocol that produced it: the facts checked exhaustively, and the two-stage sampling law
(decision 43) with its failure probability.  Readers for a particular consumer (PoUW's incorrect-unit bound, the KV
touched-unit bound) live with the consumer, as a utility passed to `worst_case`.  One `delta` serves every consumer;
the application's failure union counts it once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import comb, inf, log


@dataclass(frozen=True, slots=True)
class IntegrityProfile:
    """Checked by sampling: each of `n_r` replay units holds `n_v` verification units, was selected with probability
    `p`, and then had exactly `k` of its verification units replayed; `replayed` = (RUs, VUs) actually checked, all
    passed.  A strategy leaving m_i incorrect verification units in replay unit i is accepted with probability
    prod_i exp(-rho(m_i)), so except with probability `delta` an accepted transcript satisfies
    sum_i rho(m_i) < ln(1/delta).  A Bernoulli law over single units is n_v = k = 1.

    Checked exhaustively:
        `programs`    digests of the evaluated Programs, in commit order;
        `prescribed`  families of `Input` values checked against external sources (e.g. "prompt", "weights", "noise");
        `linked`      every other `Input` equals a committed output of an earlier Program in the transcript;
        `covered`     gate families proved on every audit, outside both draws (their faults survive with probability 0).
    """

    n_r: int
    n_v: int
    p: Fraction
    k: int
    delta: float
    programs: tuple[str, ...] = field(default=(), kw_only=True)
    prescribed: frozenset[str] = field(default=frozenset(), kw_only=True)
    linked: bool = field(default=False, kw_only=True)
    covered: frozenset[str] = field(default=frozenset(), kw_only=True)
    replayed: tuple[int, int] = field(default=(0, 0), kw_only=True)

    def __post_init__(self) -> None:
        if self.n_r < 0 or self.n_v < 1 or not 0 <= self.k <= self.n_v:
            raise ValueError("a profile needs n_r >= 0, n_v >= 1 and 0 <= k <= n_v")
        if not 0 <= Fraction(self.p) <= 1 or not 0 < self.delta < 1:
            raise ValueError("a profile needs 0 <= p <= 1 and 0 < delta < 1")
        object.__setattr__(self, "programs", tuple(self.programs))
        object.__setattr__(self, "prescribed", frozenset(self.prescribed))
        object.__setattr__(self, "covered", frozenset(self.covered))
        object.__setattr__(self, "replayed", tuple(self.replayed))

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
        """An upper bound on sum_i utility(m_i) over the strategies the profile admits, for utility(0) = 0.

        It is the optimum of the linear relaxation: choose x_m >= 0 replay units at each m in 1..n_v with
        sum x_m <= n_r and sum x_m rho(m) <= ln(1/delta).  Two constraints, so an optimal vertex uses at most two
        values of m, and every pair is tried."""
        budget = log(1 / self.delta)
        options = [(float(u), self.rho(m)) for m in range(1, self.n_v + 1) if (u := utility(m)) > 0]
        options = [(u, r) for u, r in options if r < inf]
        if not options:
            return 0.0
        n_r = self.n_r
        best = max(u * (n_r if r == 0 else min(n_r, budget / r)) for u, r in options)
        for (ua, ra), (ub, rb) in combinations(sorted(options, key=lambda o: o[1]), 2):
            if rb > ra and ra * n_r < budget < rb * n_r:
                xb = (budget - n_r * ra) / (rb - ra)
                best = max(best, ua * (n_r - xb) + ub * xb)
        return best

    def to_json(self) -> dict:
        return {"programs": list(self.programs), "prescribed": sorted(self.prescribed), "linked": self.linked,
                "covered": sorted(self.covered), "n_r": self.n_r, "n_v": self.n_v, "p": str(Fraction(self.p)),
                "k": self.k, "delta": self.delta, "replayed": list(self.replayed)}


__all__ = ["IntegrityProfile"]
