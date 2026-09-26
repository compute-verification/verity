"""`IntegrityProfile`: what verification established about the evaluation of a transcript of Programs.

An integrity protocol (sampled proofs) produces it.  Resource accounting, policies and PoUW read what they need off it
without importing the protocol that produced it: the facts checked exhaustively, and the verifier's sampling law
(decision 43) with its failure probability.  Readers for a particular consumer (PoUW's incorrect-unit bound, the KV
touched-unit bound) live with the consumer, as a utility passed to `worst_case`.  One `delta` serves every consumer;
the application's failure union counts it once.

The law is stated over named levels of a partition hierarchy (`verity.ir.partition.Level`): each level's members
partition the members of the level above, and a level is either drawn (`Bernoulli`, `Subset`) or only groups.  The
names are the producer's labels (sampled proofs draws "replay" with `Bernoulli(p)` and "check" with `Subset(k)`); the
profile carries draws, never roles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import comb, inf, log, prod
from types import MappingProxyType

from verity.ir.partition import Hierarchy, Level


@dataclass(frozen=True, slots=True)
class Bernoulli:
    """Every member of the level is drawn independently with probability `p`."""

    p: Fraction

    def __post_init__(self) -> None:
        if not 0 <= Fraction(self.p) <= 1:
            raise ValueError("a Bernoulli draw needs 0 <= p <= 1")
        object.__setattr__(self, "p", Fraction(self.p))

    def to_json(self) -> dict:
        return {"bernoulli": str(self.p)}


@dataclass(frozen=True, slots=True)
class Subset:
    """Exactly `k` members of the level are drawn, uniformly, inside each drawn member of the nearest drawn level
    above (the whole transcript when none is)."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("a Subset draw needs k >= 0")

    def to_json(self) -> dict:
        return {"subset": self.k}


Draw = Bernoulli | Subset


@dataclass(frozen=True, slots=True)
class IntegrityProfile:
    """Checked by sampling over `levels`, coarse to fine: `sizes[0]` members at the top level, then `sizes[i]`
    members of level i inside each member of level i - 1.  The finest level is the checked one; each member of it
    is one statement.  `draws` names the drawn levels: at most one `Bernoulli`, and a `Subset` only at the checked
    level; undrawn levels only group.  `drawn` counts the members actually drawn at each level, all passed.

    Let n be the checked members inside one Bernoulli member (the whole transcript when no level is Bernoulli-drawn,
    with p = 1).  The k-subset is uniform over all n of them, so undrawn levels in between don't change the law: a
    strategy leaving m_i incorrect checked members in Bernoulli member i is accepted with probability
    prod_i exp(-rho(m_i)), and except with probability `delta` an accepted transcript satisfies
    sum_i rho(m_i) < ln(1/delta).  A Bernoulli law over single units is one level drawn `Bernoulli(p)`.

    Checked exhaustively:
        `programs`    digests of the evaluated Programs, in commit order;
        `prescribed`  families of `Input` values checked against external sources (e.g. "prompt", "weights", "noise");
        `linked`      every other `Input` equals a committed output of an earlier Program in the transcript;
        `covered`     gate families proved on every audit, outside the draws (their faults survive with probability 0).
    """

    levels: Hierarchy
    sizes: tuple[int, ...]
    draws: Mapping[str, Draw] = field(hash=False)
    delta: float
    programs: tuple[str, ...] = field(default=(), kw_only=True)
    prescribed: frozenset[str] = field(default=frozenset(), kw_only=True)
    linked: bool = field(default=False, kw_only=True)
    covered: frozenset[str] = field(default=frozenset(), kw_only=True)
    drawn: Mapping[str, int] = field(default_factory=dict, kw_only=True, hash=False)

    def __post_init__(self) -> None:
        levels = tuple(Level(x) if isinstance(x, str) else x for x in self.levels)
        names = [lv.name for lv in levels]
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "sizes", tuple(self.sizes))
        if not levels or len(set(names)) != len(names) or len(self.sizes) != len(levels):
            raise ValueError("a profile needs distinct level names and one size per level")
        if self.sizes[0] < 0 or any(s < 1 for s in self.sizes[1:]):
            raise ValueError("a profile needs sizes[0] >= 0 and every finer size >= 1")
        order = {name: i for i, name in enumerate(names)}
        if not set(self.draws) <= set(order) or not set(self.drawn) <= set(order):
            raise ValueError("draws and drawn counts are keyed by level names")
        object.__setattr__(self, "draws", MappingProxyType({n: self.draws[n] for n in names if n in self.draws}))
        object.__setattr__(self, "drawn", MappingProxyType({n: int(self.drawn[n]) for n in names if n in self.drawn}))
        bernoulli = [n for n, d in self.draws.items() if isinstance(d, Bernoulli)]
        subsets = [n for n, d in self.draws.items() if isinstance(d, Subset)]
        if len(bernoulli) + len(subsets) != len(self.draws) or len(bernoulli) > 1:
            raise ValueError("a profile's draws are Bernoulli or Subset, with at most one Bernoulli level")
        if names[-1] not in self.draws or subsets not in ([], [names[-1]]):
            raise ValueError("the checked (finest) level is drawn, and only it may be drawn by Subset")
        if any(v < 0 for v in self.drawn.values()):
            raise ValueError("drawn counts are >= 0")
        _, n, _, k = self._law()
        if k > n:
            raise ValueError(f"a Subset draw of {k} exceeds the {n} checked members it draws from")
        if not 0 < self.delta < 1:
            raise ValueError("a profile needs 0 < delta < 1")
        object.__setattr__(self, "programs", tuple(self.programs))
        object.__setattr__(self, "prescribed", frozenset(self.prescribed))
        object.__setattr__(self, "covered", frozenset(self.covered))

    def _index(self, name: str) -> int:
        for i, lv in enumerate(self.levels):
            if lv.name == name:
                return i
        raise KeyError(f"the profile has no level {name!r}")

    def count(self, name: str, within: str | None = None) -> int:
        """Members of level `name` inside one member of level `within` (None: the whole transcript)."""
        i, j = self._index(name), (-1 if within is None else self._index(within))
        if j > i:
            raise ValueError(f"level {within!r} is finer than {name!r}")
        return prod(self.sizes[j + 1:i + 1])

    def _law(self) -> tuple[int, int, Fraction, int]:
        """(Bernoulli members, checked members n inside each, p, k)."""
        checked = self.levels[-1].name
        at = next((n for n, d in self.draws.items() if isinstance(d, Bernoulli)), None)
        p = self.draws[at].p if at is not None else Fraction(1)
        sub = self.draws.get(checked)
        k = sub.k if isinstance(sub, Subset) else 1
        members = self.count(at) if at is not None else 1
        return members, self.count(checked, at), p, k

    def accept(self, m: int) -> Fraction:
        """Exact probability that one Bernoulli member with `m` incorrect checked members passes:
        1 - p + p * C(n - m, k) / C(n, k)."""
        _, n, p, k = self._law()
        if not 0 <= m <= n:
            raise ValueError(f"0 <= m <= {n}")
        return 1 - p + p * Fraction(comb(n - m, k), comb(n, k))

    def rho(self, m: int) -> float:
        """Detection risk -ln accept(m)."""
        a = self.accept(m)
        return inf if a == 0 else -log(a)

    def worst_case(self, utility: Callable[[int], float], cap: float | None = None) -> float:
        """An upper bound on sum_i utility(m_i) over the Bernoulli members i the profile admits, for utility(0) = 0.

        It is the optimum of the linear relaxation: choose x_m >= 0 Bernoulli members at each m in 1..n with
        sum x_m <= members and sum x_m rho(m) <= ln(1/delta).  Two constraints, so an optimal vertex uses at most two
        values of m, and every pair is tried.  A consumer reading at a level above the Bernoulli level (touched
        requests, say) passes `cap`, that level's member count (`count`), which the bound never exceeds."""
        if cap is not None and cap < 0:
            raise ValueError("cap >= 0")
        members, n, _, _ = self._law()
        budget = log(1 / self.delta)
        options = [(float(u), self.rho(m)) for m in range(1, n + 1) if (u := utility(m)) > 0]
        options = [(u, r) for u, r in options if r < inf]
        best = 0.0
        if options:
            best = max(u * (members if r == 0 else min(members, budget / r)) for u, r in options)
            for (ua, ra), (ub, rb) in combinations(sorted(options, key=lambda o: o[1]), 2):
                if rb > ra and ra * members < budget < rb * members:
                    xb = (budget - members * ra) / (rb - ra)
                    best = max(best, ua * (members - xb) + ub * xb)
        return best if cap is None else min(best, float(cap))

    def to_json(self) -> dict:
        return {"programs": list(self.programs), "prescribed": sorted(self.prescribed), "linked": self.linked,
                "covered": sorted(self.covered),
                "levels": [{"name": lv.name, "query": lv.query, "size": size,
                            "draw": self.draws[lv.name].to_json() if lv.name in self.draws else None}
                           for lv, size in zip(self.levels, self.sizes)],
                "delta": self.delta, "drawn": dict(self.drawn)}


__all__ = ["Bernoulli", "Draw", "IntegrityProfile", "Subset"]
