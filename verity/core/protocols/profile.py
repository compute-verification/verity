"""`IntegrityProfile`: the audit's output, bounds on the faulty proof blocks of a partition, per stratum, at `delta`.

An audit protocol (one-stage, sampled proofs) produces it: the facts checked exhaustively, and the verifier's sampling
law (decision 43) with its failure probability, from which the bounds on wrong units follow (`bound`, `stratum_bounds`).
It counts wrong units and weights nothing.  Applications consume it without importing the protocol that produced it:
each weights wrong units by what one costs it, through the generic weighted bounds (`harm_bound`, `worst_case`), and
its reader lives with the application, not in core.  One `delta` serves every consumer; the application's failure
union counts it once.

The law is stated over named levels of a partition hierarchy (`verity.ml.circuits.partition.Level`): each level's members
partition the members of the level above, and a level is either drawn (`Bernoulli`, `Subset`, `Stratified`) or only
groups.  The names are the producer's labels (sampled proofs' one-stage profile is one level "unit"); the profile
carries draws, never roles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, expm1, floor, inf, lgamma, log, log1p, nextafter, prod
from types import MappingProxyType

from verity.ml.circuits.partition import Hierarchy, Level


#: `worst_case`'s margin: risks are shrunk and the bound raised by this relative amount, far above the arithmetic's error
#: (a few units in the last place), so the bound is never below the relaxation's exact optimum
MARGIN = 2.0 ** -30


def neg_log_ratio(a: int, d: int) -> float:
    """-ln(a / d) for integers 0 < a <= d, to a few units in the last place: log1p of the correctly rounded 1 - a/d while
    a/d >= 1/2 (no cancellation), the log of the correctly rounded d/a while it fits a float, and the difference of the
    two big-integer logs only beyond that, where -ln(a/d) > 693 and the difference is relatively tiny."""
    if 2 * a >= d:
        return -log1p(-((d - a) / d))
    if a << 1000 >= d:
        return log(d / a)
    return log(d) - log(a)


@dataclass(frozen=True, slots=True)
class Bernoulli:
    """Every member of the level is sampled independently with probability `p`."""

    p: Fraction

    def __post_init__(self) -> None:
        if not 0 <= Fraction(self.p) <= 1:
            raise ValueError("a Bernoulli draw needs 0 <= p <= 1")
        object.__setattr__(self, "p", Fraction(self.p))

    def to_json(self) -> dict:
        return {"bernoulli": str(self.p)}

    def count_bound(self, population: int, delta: float) -> int:
        """The largest M <= population with (1 - p)^M >= delta (Lean `Bound.bernoulliBound`): each member is sampled
        independently, so a set of M wrong members escapes with probability exactly (1 - p)^M, and except with
        probability delta an accepted transcript has at most this many.  All of them when 1 - population * p >= delta
        (Bernoulli's inequality), else the search doubles M while it escapes, then halves, in exact arithmetic."""
        if not 0 < delta < 1:
            raise ValueError("0 < delta < 1")
        d, q = Fraction(delta), 1 - self.p
        if 1 - population * self.p >= d:
            return population
        lo, hi = 0, 1
        while hi <= population and q ** hi >= d:
            lo, hi = hi, 2 * hi
        hi = min(hi, population + 1)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if q ** mid >= d else (lo, mid)
        return lo


@dataclass(frozen=True, slots=True)
class Subset:
    """Exactly `k` members of the level are sampled, uniformly, inside each drawn member of the nearest drawn level
    above (the whole transcript when none is)."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("a Subset draw needs k >= 0")

    def to_json(self) -> dict:
        return {"subset": self.k}


@dataclass(frozen=True, slots=True)
class IntegrityProfile:
    """Checked by sampling over `levels`, coarse to fine: `sizes[0]` members at the top level, then `sizes[i]`
    members of level i inside each member of level i - 1.  The finest level is the checked one; each member of it
    is one statement.  `draws` names the drawn levels: at most one `Bernoulli`, and a `Subset` only at the checked
    level; undrawn levels only group.  A `Stratified` draw is the only draw, of the checked level, with its strata
    tiling that level's members; `accept`, `rho` and `worst_case` are the other laws', and it reads through `strata`.
    `drawn` counts the members actually drawn at each level, all passed.

    A profile of more than one level is valid only when its checked level is committed before the first draw
    (as serving commits every value, at serve time).  When the checked level is committed after
    a sample it follows, the prover picks how many checked members of a drawn member are wrong after seeing that sample,
    and `accept(m)` over m wrong checked members no longer bounds it; such a lifecycle's profile is over the level
    committed first.

    Let n be the checked members inside one Bernoulli member (the whole transcript when no level is Bernoulli-drawn,
    with p = 1).  The k-subset is uniform over all n of them, so undrawn levels in between don't change the law: a
    strategy leaving m_i incorrect checked members in Bernoulli member i is accepted with probability
    prod_i exp(-rho(m_i)), and except with probability `delta` an accepted transcript satisfies
    sum_i rho(m_i) < ln(1/delta).  A Bernoulli law over single statements is one level drawn `Bernoulli(p)`.

    Checked exhaustively:
        `programs`    digests of the evaluated Programs, in commit order;
        `prescribed`  families of `Input` values checked against external sources (e.g. "prompt", "weights", "noise");
        `linked`      every other `Input` equals a committed output of an earlier Program in the transcript;
        `covered`     gate families proved on every audit, outside the samples (their faults survive with probability 0).
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
        stratified = [n for n, d in self.draws.items() if isinstance(d, Stratified)]
        if len(bernoulli) + len(subsets) + len(stratified) != len(self.draws) or len(bernoulli) > 1:
            raise ValueError("a profile's draws are Bernoulli, Subset or Stratified, with at most one Bernoulli level")
        if names[-1] not in self.draws or subsets not in ([], [names[-1]]):
            raise ValueError("the checked (finest) level is drawn, and only it may be drawn by Subset")
        if stratified and list(self.draws) != [names[-1]]:
            raise ValueError("a Stratified draw is the profile's only draw, of the checked level")
        if any(v < 0 for v in self.drawn.values()):
            raise ValueError("drawn counts are >= 0")
        if stratified:
            if self.draws[names[-1]].population != self.count(names[-1]):
                raise ValueError(f"the strata hold {self.draws[names[-1]].population} units, the checked level "
                                 f"{self.count(names[-1])}")
        else:
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
        if isinstance(self.draws[checked], Stratified):
            raise ValueError("a stratified profile has no single k: read it through `strata`")
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

    def _odds(self) -> tuple[int, int, int, int, int]:
        """(n, k, p's numerator, p's denominator, C(n, k)): accept(m) = ((den - num) C(n, k) + num C(n - m, k)) /
        (den C(n, k)), in integers."""
        _, n, p, k = self._law()
        return n, k, p.numerator, p.denominator, comb(n, k)

    def rho(self, m: int) -> float:
        """Detection risk -ln accept(m), from exact integers to a few units in the last place (`neg_log_ratio`): no
        underflow where `accept` falls below the smallest float, and no cancellation where it is close to 1."""
        n, k, num, den, top = self._odds()
        if not 0 <= m <= n:
            raise ValueError(f"0 <= m <= {n}")
        a = (den - num) * top + num * comb(n - m, k)
        return inf if a == 0 else neg_log_ratio(a, den * top)

    def _rhos(self) -> list[float]:
        """rho(0), ..., rho(n), with C(n - m, k) updated exactly from m to m + 1: one pass, the same values as `rho`."""
        n, k, num, den, top = self._odds()
        whole, c, out = den * top, top, []
        for m in range(n + 1):
            a = (den - num) * top + num * c
            out.append(inf if a == 0 else neg_log_ratio(a, whole))
            c = c * (n - m - k) // (n - m) if n - m > k else 0
        return out

    def worst_case(self, utility: Callable[[int], float], cap: float | None = None) -> float:
        """An upper bound on sum_i utility(m_i) over the Bernoulli members i the profile admits, for utility(0) = 0.

        It is the optimum of the linear relaxation: choose x_m >= 0 Bernoulli members at each m in 1..n with
        sum x_m <= members and sum x_m rho(m) <= ln(1/delta).  Scaled by `members`, that is the largest value, at
        mean risk at most ln(1/delta)/members, of the upper concave envelope of the points (rho(m), utility(m)) and
        (0, 0).  rho is nondecreasing in m (accept is), so the points arrive sorted and one pass builds the envelope:
        linear in n, where trying every pair of m was quadratic.  The rounding is directed: every risk is shrunk and the
        result raised by `MARGIN`, capped at members * max utility, which the optimum never exceeds, so the bound is
        never below the exact optimum.  A consumer reading at a level above the Bernoulli level (touched requests, say)
        passes `cap`, that level's member count (`count`), which the bound never exceeds."""
        if cap is not None and cap < 0:
            raise ValueError("cap >= 0")
        members, n, _, _ = self._law()
        rhos = self._rhos()
        points = [(r * (1 - MARGIN), float(u)) for m in range(1, n + 1) if (u := utility(m)) > 0 and (r := rhos[m]) < inf]
        best = 0.0
        if points and members > 0:
            hull: list[tuple[float, float]] = []
            for pt in sorted([(0.0, 0.0), *points]):
                while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (pt[1] - hull[-2][1])
                                          >= (hull[-1][1] - hull[-2][1]) * (pt[0] - hull[-2][0])):
                    hull.pop()
                hull.append(pt)
            t = log(1 / self.delta) / members
            for (ra, ua), (rb, ub) in zip(hull, hull[1:] + [(inf, 0.0)]):
                if ra <= t:
                    best = max(best, ua)
                    if t < rb < inf:
                        best = max(best, ua + (ub - ua) * (t - ra) / (rb - ra))
            top = max(u for _r, u in points)
            full = members * top
            if Fraction(full) < members * Fraction(top):                 # the cap, rounded up
                full = nextafter(full, inf)
            best = full if best * members >= full else min(full, nextafter(best * members * (1 + MARGIN), inf))
        return best if cap is None else min(best, float(cap))

    def strata(self) -> Stratified | None:
        """The law as independent uniform draws inside strata of the checked level: its `Stratified` draw, or, when a
        `Subset(k)` of the checked level is the only draw, one stratum of all its members, named after the level.  None
        when a level is drawn `Bernoulli`."""
        checked = self.levels[-1].name
        law = self.draws[checked]
        if isinstance(law, Stratified):
            return law
        if isinstance(law, Subset) and len(self.draws) == 1:
            return Stratified.uniform(self.count(checked), law.k, checked)
        return None

    def bound(self) -> int | float:
        """Except with probability delta, an accepted transcript has at most this many wrong checked members: exact
        when the law has strata (`Stratified.count_bound`) or is one `Bernoulli` draw of the checked level
        (`Bernoulli.count_bound`), else `worst_case` with utility m (the linear relaxation, from above)."""
        law, checked = self.strata(), self.levels[-1].name
        if law is not None:
            return law.count_bound(self.delta)
        if isinstance(self.draws[checked], Bernoulli) and len(self.draws) == 1:
            return self.draws[checked].count_bound(self.count(checked), self.delta)
        return self.worst_case(lambda m: float(m))

    def stratum_bounds(self) -> dict[str, int]:
        """Each stratum's own bound at delta (`Stratified.stratum_bound`); empty when a level is sampled `Bernoulli`.  They
        hold jointly with `bound`, not each at its own delta: the one-stage profile bounds a family of wrong-unit sets by
        its likeliest set to escape, and every set breaking one of these bounds escapes with less than delta."""
        law = self.strata()
        return {} if law is None else {s.name: law.stratum_bound(s.name, self.delta) for s in law.strata}

    def harm_bound(self, harm: float | Mapping[str, float | Sequence[tuple[float, int]]]) -> float:
        """An upper bound, at delta, on the total harm of the wrong checked members: the generic weighted bound an
        application reads with its own harm per member.  `harm` is one number for every member, or, when the law has
        strata, per stratum a number or (harm, members) pairs (`Stratified.harm_bound`).  Under a `Bernoulli` level
        every member weighs the same: `worst_case` with utility harm * m."""
        law = self.strata()
        if law is not None:
            per = harm if isinstance(harm, Mapping) else {s.name: float(harm) for s in law.strata}
            return law.harm_bound(per, self.delta)
        if isinstance(harm, Mapping):
            raise ValueError("under a Bernoulli level every checked member weighs the same: pass one number")
        return self.worst_case(lambda m: harm * m)

    def to_json(self) -> dict:
        return {"programs": list(self.programs), "prescribed": sorted(self.prescribed), "linked": self.linked,
                "covered": sorted(self.covered),
                "levels": [{"name": lv.name, "query": lv.query, "size": size,
                            "draw": self.draws[lv.name].to_json() if lv.name in self.draws else None}
                           for lv, size in zip(self.levels, self.sizes)],
                "delta": self.delta, "drawn": dict(self.drawn)}


#: `Stratified.harm_bound`'s margin: the risk budget and the result are raised by this relative amount, above the float
#: risks' error (lgamma differences at a billion units are within 2^-26 relative), so the bound is never below the
#: relaxation's optimum
HARM_MARGIN = 2.0 ** -20


@dataclass(frozen=True, slots=True)
class Stratum:
    """`size` units of the partition, `k` of them sampled uniformly, independently of every other stratum."""

    name: str
    size: int
    k: int

    def __post_init__(self) -> None:
        if not self.name or self.size < 0 or not 0 <= self.k <= self.size:
            raise ValueError("a stratum needs a name, size >= 0 and 0 <= k <= size")

    def escape(self, m: int) -> Fraction:
        """C(size - m, k) / C(size, k): the sample misses m given units of the stratum."""
        return Fraction(comb(self.size - m, self.k), comb(self.size, self.k))

    def _finite(self) -> int:
        """The most wrong units the stratum can hold at a positive escape."""
        return self.size - self.k if self.k else self.size

    def _risk(self, m: int) -> float:
        """-ln escape(m) in floats, for searching only: the log1p sum to a few units in the last place per term while k
        is small, the lgamma difference beyond."""
        if self.k == 0 or m == 0:
            return 0.0
        if m > self._finite():
            return inf
        if self.k <= 1 << 14:
            return -sum(log1p(-m / (self.size - i)) for i in range(self.k))
        n, k = self.size, self.k
        return (lgamma(n + 1) - lgamma(n - k + 1)) - (lgamma(n - m + 1) - lgamma(n - m - k + 1))


def _key_lt(a: Stratum, ma: int, b: Stratum, mb: int) -> bool:
    """The increment of stratum a from ma wrong units is smaller than b's from mb: k_a / (n_a - ma) < k_b / (n_b - mb),
    in integers (the risk increment -ln(1 - k / (n - m)) is increasing in that ratio)."""
    return a.k * (b.size - mb) < b.k * (a.size - ma)


@dataclass(frozen=True, slots=True)
class Stratified:
    """The verifier's law as independent uniform draws inside disjoint strata of the partition's units (the `stratified`
    law): the uniform k-subset is one stratum, and a template floor is a stratum with
    k >= 1.  The strata are fixed by the agreement before registration, like every law.

    A wrong-unit set with m_s units in stratum s escapes the sample with prod_s C(n_s - m_s, k_s) / C(n_s, k_s), which
    depends only on the counts.  Its risk rho = -ln(escape) is convex in each m_s, the increment from m to m + 1 being
    -ln(1 - k_s / (n_s - m)), so the worst case of any additive harm is a greedy fill in the order of increment per unit
    of harm: `miss` and `count_bound` exactly (increments ordered by k_s / (n_s - m), compared in integers, and the
    escape in rationals), `harm_bound` from above (the linear relaxation, which may take a fraction of its last
    increment).  By the one-stage profile (`Audit/OneStage.lean` `audit_profile`, which holds for every law),
    Pr[accept and at least K wrong] <= miss(K) + epsilon."""

    strata: tuple[Stratum, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strata", tuple(self.strata))
        if not self.strata or not all(isinstance(s, Stratum) for s in self.strata) \
                or len({s.name for s in self.strata}) != len(self.strata):
            raise ValueError("a stratified law needs strata with distinct names")

    @classmethod
    def uniform(cls, n: int, k: int, name: str = "unit") -> Stratified:
        """The uniform k-subset of n units: one stratum."""
        return cls((Stratum(name, n, k),))

    @property
    def population(self) -> int:
        return sum(s.size for s in self.strata)

    @property
    def k(self) -> int:
        """Units drawn."""
        return sum(s.k for s in self.strata)

    def stratum(self, name: str) -> Stratum:
        for s in self.strata:
            if s.name == name:
                return s
        raise KeyError(f"the law has no stratum {name!r}")

    def inclusion(self, name: str) -> Fraction:
        """The probability that a given unit of the stratum is sampled."""
        s = self.stratum(name)
        return Fraction(s.k, s.size) if s.size else Fraction(0)

    def _counts(self, counts: Mapping[str, int]) -> list[int]:
        if not set(counts) <= {s.name for s in self.strata}:
            raise KeyError(f"unknown strata {sorted(set(counts) - {s.name for s in self.strata})}")
        ms = [int(counts.get(s.name, 0)) for s in self.strata]
        if any(not 0 <= m <= s.size for m, s in zip(ms, self.strata)):
            raise ValueError("0 <= wrong units <= size in every stratum")
        return ms

    def escape(self, counts: Mapping[str, int]) -> Fraction:
        """The exact probability that the sample misses a wrong-unit set with these counts per stratum."""
        return prod((s.escape(m) for s, m in zip(self.strata, self._counts(counts))), start=Fraction(1))

    def rho(self, counts: Mapping[str, int]) -> float:
        """-ln escape, from exact integers to a few units in the last place per stratum (`neg_log_ratio`)."""
        out = 0.0
        for s, m in zip(self.strata, self._counts(counts)):
            a = comb(s.size - m, s.k)
            if a == 0:
                return inf
            out += neg_log_ratio(a, comb(s.size, s.k))
        return out

    def _worst(self, K: int) -> list[int] | None:
        """The counts of the K wrong units that escape most likely (None when every set of K is caught): the K smallest
        increments over all strata.  A float threshold places the counts, and exact exchanges settle the margin."""
        strata, finite = self.strata, [s._finite() for s in self.strata]
        if K > sum(finite):
            return None

        def at(q: float) -> list[int]:
            return [f if s.k == 0 else 0 if q <= 0 else max(0, min(f, floor(s.size - s.k / q) + 1))
                    for s, f in zip(strata, finite)]

        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if sum(at(mid)) >= K else (mid, hi)
        ms = at(hi)

        def smallest_next() -> int | None:
            best = None
            for i in range(len(strata)):
                if ms[i] < finite[i] and (best is None or _key_lt(strata[i], ms[i], strata[best], ms[best])):
                    best = i
            return best

        def largest_last() -> int | None:
            best = None
            for i in range(len(strata)):
                if ms[i] > 0 and (best is None or _key_lt(strata[best], ms[best] - 1, strata[i], ms[i] - 1)):
                    best = i
            return best

        while sum(ms) < K:
            ms[smallest_next()] += 1
        while sum(ms) > K:
            ms[largest_last()] -= 1
        while (i := largest_last()) is not None and (j := smallest_next()) is not None and i != j \
                and _key_lt(strata[j], ms[j], strata[i], ms[i] - 1):
            ms[i] -= 1
            ms[j] += 1
        return ms

    def worst_counts(self, K: int) -> dict[str, int] | None:
        """Where K wrong units escape most likely: their counts per stratum, or None when every set of K is caught."""
        ms = self._worst(K)
        return None if ms is None else {s.name: m for s, m in zip(self.strata, ms)}

    def miss(self, K: int) -> Fraction:
        """The count curve's sampling term: the largest escape of a set of K wrong units (0 <= K <= population), exact."""
        if not 0 <= K <= self.population:
            raise ValueError(f"0 <= K <= {self.population}")
        ms = self._worst(K)
        return Fraction(0) if ms is None else prod((s.escape(m) for s, m in zip(self.strata, ms)), start=Fraction(1))

    def count_bound(self, delta: float) -> int:
        """The largest K with miss(K) >= delta: except with probability delta (plus epsilon), an accepted transcript has
        at most this many wrong units.  miss is nonincreasing in K, so a binary search in exact arithmetic."""
        if not 0 < delta < 1:
            raise ValueError("0 < delta < 1")
        d = Fraction(delta)
        lo, hi = 0, self.population + 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if self.miss(mid) >= d else (lo, mid)
        return lo

    def stratum_bound(self, name: str, delta: float) -> int:
        """The largest m with escape(m wrong units, all in this stratum) >= delta: the stratum's own count curve, what a
        consumer of that stratum alone reads."""
        if not 0 < delta < 1:
            raise ValueError("0 < delta < 1")
        s, d = self.stratum(name), Fraction(delta)
        lo, hi = 0, s.size + 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if s.escape(mid) >= d else (lo, mid)
        return lo

    def harm_bound(self, harm: Mapping[str, float | Sequence[tuple[float, int]]], delta: float) -> float:
        """An upper bound on sum_{u in B} harm(u) over the wrong-unit sets B with escape >= delta.

        `harm[name]` is one number (every unit of the stratum) or (harm, units) pairs, the stratum's other units harming
        nothing; the harm is the consumer's, what a wrong unit costs it.  Inside a stratum the adversary
        fills the most harmful units first, so the value of m wrong units is concave in m while the risk is convex, and
        the linear relaxation (units taken in the order of risk increment per unit of harm, the last one fractionally)
        is the optimum from above.  The float risks are searched, then the budget and the result are raised by
        `HARM_MARGIN`."""
        if not 0 < delta < 1:
            raise ValueError("0 < delta < 1")
        if not set(harm) <= {s.name for s in self.strata}:
            raise KeyError(f"unknown strata {sorted(set(harm) - {s.name for s in self.strata})}")
        segs: list[list[tuple[float, int]]] = []
        for s in self.strata:
            h = harm.get(s.name, 0.0)
            pairs = [(float(h), s.size)] if isinstance(h, (int, float)) else [(float(v), int(c)) for v, c in h]
            if any(v < 0 or c < 0 for v, c in pairs) or sum(c for _, c in pairs) > s.size:
                raise ValueError(f"stratum {s.name!r}: harms >= 0 over at most its {s.size} units")
            segs.append(sorted(((v, c) for v, c in pairs if v > 0 and c > 0), reverse=True))
        budget = log(1 / delta) * (1 + HARM_MARGIN)

        def unit(i: int, m: int) -> float:
            """The harm of stratum i's (m + 1)-th wrong unit (0 past its harmful units)."""
            for v, c in segs[i]:
                if m < c:
                    return v
                m -= c
            return 0.0

        def fill(tau: float) -> tuple[list[int], float, float]:
            ms, value, risk = [], 0.0, 0.0
            for s, sg in zip(self.strata, segs):
                m, f = 0, s._finite()
                for v, c in sg:
                    top = min(m + c, f)
                    x = top if s.k == 0 else 0 if tau * v <= 0 else \
                        max(0, min(top, floor(s.size - s.k / -expm1(-tau * v)) + 1))
                    whole = x == m + c
                    if x > m:
                        value, m = value + (x - m) * v, x
                    if not whole:
                        break
                ms.append(m)
                risk += s._risk(m)
            return ms, value, risk

        lo, hi = 0.0, 1.0
        while fill(hi)[2] <= budget and hi < 2.0 ** 60:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fill(mid)[2] <= budget else (lo, mid)
        ms, value, risk = fill(lo)
        while True:
            options = [(s._risk(m + 1) - s._risk(m)) / unit(i, m) for i, (s, m) in enumerate(zip(self.strata, ms))
                       if m < s._finite() and unit(i, m) > 0]
            if not options:
                break
            ratio = min(options)
            i = next(i for i, (s, m) in enumerate(zip(self.strata, ms))
                     if m < s._finite() and unit(i, m) > 0 and (s._risk(m + 1) - s._risk(m)) / unit(i, m) == ratio)
            step = self.strata[i]._risk(ms[i] + 1) - self.strata[i]._risk(ms[i])
            if risk + step > budget:
                value += unit(i, ms[i]) * (budget - risk) / step
                break
            risk, value, ms[i] = risk + step, value + unit(i, ms[i]), ms[i] + 1
        return nextafter(value * (1 + HARM_MARGIN), inf)

    def to_json(self) -> dict:
        return {"law": "stratified", "strata": [{"name": s.name, "size": s.size, "k": s.k} for s in self.strata]}


Draw = Bernoulli | Subset | Stratified


__all__ = ["HARM_MARGIN", "MARGIN", "Bernoulli", "Draw", "IntegrityProfile", "Stratified", "Stratum", "Subset",
           "neg_log_ratio"]
