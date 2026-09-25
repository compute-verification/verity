"""verity.proofs: what verification established.

`IntegrityProfile` summarises what an integrity protocol verified about the evaluation of a transcript of Programs.
Consumers (PoUW's work claim, PoUS's KV claim, output and training policies) read what they need off it without
importing the protocol that produced it: facts that were checked exhaustively, and a bound on what sampling may
have missed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import comb, inf, log


@dataclass(frozen=True, slots=True)
class IntegrityProfile:
    """Checked exhaustively:
        `programs`    digests of the evaluated Programs, in commit order;
        `prescribed`  families of `Input` values checked against external sources (e.g. "prompt", "weights", "noise");
        `linked`      every other `Input` equals a committed output of an earlier Program in the transcript.
    Checked by sampling: each of `n_r` replay units was selected with probability `p`, then `k` of its `n_v`
    verification units were replayed (`replayed` = (RUs, VUs) actually checked, all passed).  Except with probability
    `delta`, the incorrect-unit counts m_1..m_{n_r} satisfy sum_i rho(m_i) < ln(1/delta)."""

    programs: tuple[str, ...]
    prescribed: frozenset[str]
    linked: bool
    n_r: int
    n_v: int
    p: Fraction
    k: int
    delta: float
    replayed: tuple[int, int]

    def rho(self, m: int) -> float:
        """Detection risk -ln Pr[accept] of one replay unit with m incorrect verification units."""
        miss = comb(self.n_v - m, self.k) / comb(self.n_v, self.k)
        accept = 1 - self.p + self.p * miss
        return inf if accept <= 0 else -log(accept)

    def worst_case(self, utility: Callable[[int], float]) -> float:
        """An upper bound on sum_i utility(m_i) over admitted strategies (utility(0) = 0): at most
        max_m utility(m) / rho(m) per unit of risk, times ln(1/delta), and never more than every unit at its best m."""
        ms = range(1, self.n_v + 1)
        per_risk = max(utility(m) / r if 0 < (r := self.rho(m)) < inf else (0.0 if r else inf) for m in ms)
        return min(per_risk * log(1 / self.delta), self.n_r * max(utility(m) for m in ms))

    def incorrect_units(self) -> float:
        """Verification units that may be wrong: what free compute or forged work can buy."""
        return self.worst_case(lambda m: m)

    def touched_units(self) -> float:
        """Replay units with any wrong verification unit: what laundered state or wrong outputs can reach."""
        return self.worst_case(lambda m: 1)


__all__ = ["IntegrityProfile"]
