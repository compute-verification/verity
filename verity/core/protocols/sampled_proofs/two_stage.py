"""The two-stage law (decision 43) and its draws.

Stage 1 draws replay units (RUs); the prover then commits the drawn RUs' interiors; stage 2 checks `k` of each drawn RU's
proof units, all of them when it holds fewer or `k` is None.  Each RU class has a `Stage`: its first stage `first` and its
`k`.  The first stage is either

* a probability p (decision 43's form): each RU of the class independently with probability p.  With every unit its own
  RU and k = 1 it is a Bernoulli law over single units (`TwoStageLaw.bernoulli`), which verification plans use; or
* any one-stage law over the class's RUs: a `one_stage.draw` law object (`subset`, `bernoulli`, `stratified` or `work`,
  the last two expanded by `one_stage.draw.derive` over the RUs' program and query), drawn as the one-stage audit draws
  its units (`draw_first`, the Lean verifier's `Flock.Draw.drawWith`), a work law with the verifier's closure.  Unless
  its law is `bernoulli`, such a stage checks every proof unit of a drawn RU (k = None): a second draw thinning a subset,
  stratified or work draw leaves an escape that no profile law states.

The verifier draws every coin from its own randomness (`OwnRandomness`), with no derivation and no beacon: the RU draw
after the boundaries are registered, the proof-unit draw after the interiors are.  A stage that checks every proof unit
draws nothing in stage 2.  A drawn RU's outcome is final: nothing redraws.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from verity.ml.randomness import uniform_from_words
from verity.core.protocols.profile import Bernoulli, IntegrityProfile, Subset

from .one_stage.audit import core_law
from .one_stage.draw import strata_problem

REPLAY = "replay"
"""The profile's only level: the replay units."""

_LAW_KEYS = {"subset": {"law", "k"}, "bernoulli": {"law", "p"}, "stratified": {"law", "k", "strata"},
             "work": {"law", "k", "strata"}}


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _ratio(p: Any) -> tuple[int, int] | None:
    num, _, den = str(p).partition("/")
    if num.isdigit() and den.isdigit() and 0 < int(den) and int(num) <= int(den):
        return int(num), int(den)
    return None


def _law_problem(law: Mapping[str, Any]) -> str | None:
    kind = law.get("law")
    if kind not in _LAW_KEYS:
        return "a one-stage law is subset, bernoulli, stratified or work"
    if set(law) != _LAW_KEYS[kind]:
        return f"a {kind} law is exactly {sorted(_LAW_KEYS[kind])}"
    if kind == "bernoulli":
        return None if _ratio(law["p"]) else "a bernoulli law's p is NUM/DEN with 0 <= NUM <= DEN and DEN > 0"
    if not _int(law["k"]) or law["k"] < 0:
        return f"a {kind} law's k is a natural number"
    if kind != "subset" and not isinstance(law["strata"], list):
        return f"a {kind} law is expanded: its strata are a list (one_stage.draw.derive)"
    return None


@dataclass(frozen=True)
class Stage:
    """One RU class's law: its RUs drawn by `first`, a probability or a one-stage law object (the module docstring), then
    `k` proof units of each drawn RU checked (`None`: all of them)."""

    first: Fraction | Mapping[str, Any]
    k: int | None = None

    def __post_init__(self) -> None:
        if self.k is not None and (not _int(self.k) or self.k < 0):
            raise ValueError("a stage needs k >= 0")
        if isinstance(self.first, Mapping):
            problem = _law_problem(self.first)
            if problem:
                raise ValueError(problem)
            if self.first["law"] != "bernoulli" and self.k is not None:
                raise ValueError(f"a {self.first['law']} first stage checks every proof unit of a drawn RU (k = None): "
                                 "a second draw would thin it into an escape no profile law states")
            object.__setattr__(self, "first", dict(self.first))
        else:
            if not 0 <= Fraction(self.first) <= 1:
                raise ValueError("a stage needs 0 <= p <= 1")
            object.__setattr__(self, "first", Fraction(self.first))

    @property
    def p(self) -> Fraction | None:
        """The probability each RU is drawn when the first stage is Bernoulli (a probability or a `bernoulli` law);
        None for a subset, stratified or work law."""
        if isinstance(self.first, Fraction):
            return self.first
        if self.first["law"] == "bernoulli":
            num, den = _ratio(self.first["p"])
            return Fraction(num, den)
        return None

    def checked(self, n_v: int) -> int:
        return n_v if self.k is None else min(self.k, n_v)

    def to_json(self) -> dict:
        if isinstance(self.first, Fraction):
            return {"p": str(self.first), "k": self.k}
        return {"law": dict(self.first), "k": self.k}


@dataclass(frozen=True)
class TwoStageLaw:
    stages: Mapping[str, Stage]

    @classmethod
    def bernoulli(cls, rates: Mapping[str, Fraction]) -> TwoStageLaw:
        """Independent per-unit rates, as the law with every unit its own RU."""
        return cls({name: Stage(Fraction(rate), 1) for name, rate in rates.items()})

    def stage(self, cls: str) -> Stage:
        try:
            return self.stages[cls]
        except KeyError:
            raise KeyError(f"the law has no stage for replay-unit class {cls!r}") from None

    def profile(self, cls: str, units: Iterable[ReplayUnit] | None = None, *, n_r: int | None = None,
                n_v: int | None = None, delta: float, **facts: Any) -> IntegrityProfile:
        """The profile an accepted transcript leaves over the RUs of class `cls`.

        It is over the RUs, the only level committed before the first draw.  The interiors are committed after the RU
        draw, so the prover picks how many proof units of a drawn RU are wrong after seeing it, and one is always
        enough; a count of wrong proof units per RU is not a claim this lifecycle supports (`docs/audit-protocols.md`
        §3.3, pitfall 6).

        Under a Bernoulli first stage, a wrong RU of `n_v` proof units is caught only when it is drawn and a wrong proof
        unit is among the `k` checked, so it escapes with probability at most 1 - p * k / n_v, independently across RUs.
        A larger RU escapes more often, so the class's rate is its largest RU's: one level `REPLAY` drawn
        `Bernoulli(p * k / n_v)` with `n_v` the largest RU of the class (`k = n_v` when every proof unit is checked).  It
        counts wrong RUs, each of at most `n_v` proof units.

        Under a subset, stratified or work first stage every proof unit of a drawn RU is checked, so a wrong RU escapes
        exactly when the first stage misses it, and the profile is the one-stage law's over the class's RUs: `Subset(k)`,
        or the stratified law's `Stratified` (a work law is the stratified law at its sizes), as `one_stage.audit.profile`
        gives it over units.  It counts the drawn RUs only: a work draw's closure proves more, and only helps.  `n_v` is
        not read.

        Pass the class's RUs as `units` (RUs of other classes are ignored): `n_r` is their count and `n_v` their largest
        proof-unit count.  A caller's `n_r` must equal the count and its `n_v` can't be smaller than the largest; without
        `units`, `n_r` is required, and `n_v` (the largest RU's proof-unit count) under a Bernoulli first stage.

        The Bernoulli rate rests on the proof-unit draw being unknown to the prover when it commits the interiors: the
        verifier draws them from its own randomness after the interiors are committed (pitfall 4; there is no beacon).
        Nothing here checks that; a proof-unit draw the prover could predict before committing voids the rate.

        `facts` are the exhaustively checked fields (`programs`, `prescribed`, `linked`, `covered`) and `drawn`
        ({`REPLAY`: RUs drawn})."""
        stage = self.stage(cls)
        if units is not None:
            mine = [u for u in _distinct(units) if u.cls == cls]
            if not mine:
                raise ValueError(f"no replay unit of class {cls!r}")
            if n_r is not None and n_r != len(mine):
                raise ValueError(f"n_r = {n_r}, but the class has {len(mine)} RUs")
            largest = max(u.n_v for u in mine)
            if n_v is not None and n_v < largest:
                raise ValueError(f"n_v = {n_v} is smaller than the class's largest RU ({largest} proof units): the profile "
                                 "would overstate detection")
            n_r, n_v = len(mine), max(largest, n_v or 0)
        p = stage.p
        if p is None:
            if n_r is None:
                raise ValueError("pass the class's RUs, or n_r")
            law = stage.first
            if law["law"] == "subset":
                if law["k"] > n_r:
                    raise ValueError(f"a subset first stage of {law['k']} exceeds the class's {n_r} RUs")
                return IntegrityProfile((REPLAY,), (n_r,), {REPLAY: Subset(law["k"])}, delta, **facts)
            problem = strata_problem(law, n_r)
            if problem:
                raise ValueError(f"the class's {law['law']} first stage over its {n_r} RUs: {problem}")
            return IntegrityProfile((REPLAY,), (n_r,), {REPLAY: core_law(law, n_r)}, delta, **facts)
        if n_r is None or n_v is None:
            raise ValueError("pass the class's RUs, or n_r and n_v (its largest RU's proof-unit count)")
        if n_v < 1:
            raise ValueError("an RU holds at least one proof unit")
        rate = p * Fraction(stage.checked(n_v), n_v)
        return IntegrityProfile((REPLAY,), (n_r,), {REPLAY: Bernoulli(rate)}, delta, **facts)

    def to_json(self) -> dict:
        return {name: stage.to_json() for name, stage in sorted(self.stages.items())}


@dataclass(frozen=True, slots=True)
class ReplayUnit:
    """An RU as the sampler sees it: its index in the committed RU order, class and proof-unit count."""

    index: int
    cls: str
    n_v: int


class _RanOut(Exception):
    pass


def _width(n: int) -> int:
    return (n.bit_length() + 64 + 7) // 8


def _uniform(n: int, read: Callable[[int], bytes]) -> int:
    w = _width(n)
    return uniform_from_words(n, (int.from_bytes(read(w), "big") for _ in iter(int, 1)), 8 * w)


def _subset(k: int, n: int, read: Callable[[int], bytes]) -> list[int]:
    swapped: dict[int, int] = {}
    for i in range(k):
        j = i + _uniform(n - i, read)
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return sorted(swapped.get(i, i) for i in range(k))


def _bernoulli(num: int, den: int, n: int, read: Callable[[int], bytes]) -> list[int]:
    if num == 0:
        return []
    if num == den:
        return list(range(n))
    return [u for u in range(n) if _uniform(den, read) < num]


def _on(src: bytes) -> Callable[[int], bytes]:
    at = 0

    def read(w: int) -> bytes:
        nonlocal at
        if at + w > len(src):
            raise _RanOut
        at += w
        return src[at - w:at]
    return read


class OwnRandomness:
    """The verifier's own randomness, read directly: exact samplers over raw bytes from `source` (default the operating
    system's generator), consumed in order, with no hash and no derivation.  They are the Lean verifier's `Flock.Draw` and
    M0's `unit_draw` samplers: `uniform(n)` reads w = ceil((bitlen(n) + 64) / 8)-byte big-endian words and returns the
    first below the largest multiple of n, mod n; `bernoulli(a/b)`, a/b in lowest terms, is `uniform(b) < a` (p = 0 and
    p = 1 read nothing); `subset` is a sparse partial Fisher-Yates."""

    def __init__(self, source: Callable[[int], bytes] = os.urandom) -> None:
        self._source = source

    def read(self, n: int) -> bytes:
        b = self._source(n)
        if len(b) != n:
            raise ValueError("the randomness source ran out")
        return b

    def uniform(self, n: int) -> int:
        if n <= 0:
            raise ValueError("uniform(n) needs n > 0")
        return _uniform(n, self.read)

    def bernoulli(self, p: Fraction) -> bool:
        p = Fraction(p)
        if not 0 <= p <= 1:
            raise ValueError("bernoulli(p) needs 0 <= p <= 1")
        if p in (0, 1):
            return p == 1
        return self.uniform(p.denominator) < p.numerator

    def subset(self, k: int, n: int) -> tuple[int, ...]:
        if not 0 <= k <= n:
            raise ValueError("subset needs 0 <= k <= n")
        return tuple(_subset(k, n, self.read))

    def stream(self, need: int, run: Callable[[Callable[[int], bytes]], list[int]]) -> list[int]:
        """Lean's `Flock.Draw.onWith`: `run` on a stream of `need` bytes, extended by `need` more and rerun from its
        start whenever it runs out, so the draw stays exact."""
        src = self.read(need)
        while True:
            try:
                return run(_on(src))
            except _RanOut:
                src += self.read(need)


def closure_of(closure: Mapping[int, Sequence[int]], units: Sequence[int]) -> list[int]:
    """A draw's closure under the verifier's closure map (Lean `Flock.Draw.closureOf`): every unit in a drawn unit's
    closure that is not itself drawn, ascending."""
    drawn = set(units)
    return sorted({c for u in units for c in closure.get(u, ()) if c not in drawn})


def draw_first(randomness: OwnRandomness, law: Mapping[str, Any], population: int,
               closure: Mapping[int, Sequence[int]] | None = None) -> dict[str, Any]:
    """Stage 1 under a one-stage law: the one-stage draw object over the RU positions [0, population) (`one_stage.draw`;
    `check_draw` holds it to the registration and the law), as the Lean verifier's `Flock.Draw.drawWith` draws it from
    the same bytes: `subset` and `bernoulli` on one stream (`bernoulli` at the law's NUM/DEN as written), `stratified`
    and `work` one stream per stratum, in order, each stratum's `subset k_s n_s` mapped to the stratum's units.  With the
    verifier's closure map (a work law only, as `flock-verify draw --closure`), the draw carries its `closure`
    (`closure_of`) when that is not empty."""
    problem = _law_problem(law)
    if problem:
        raise ValueError(problem)
    kind = law["law"]
    if closure is not None and kind != "work":
        raise ValueError("a closure widens a work draw only")
    if kind == "subset":
        k = law["k"]
        if k > population:
            raise ValueError(f"a subset draw of {k} exceeds the population of {population}")
        units = randomness.stream(2 * k * _width(population) + 64, lambda read: _subset(k, population, read))
    elif kind == "bernoulli":
        num, den = _ratio(law["p"])
        units = randomness.stream(2 * population * _width(den) + 64, lambda read: _bernoulli(num, den, population, read))
    else:
        problem = strata_problem(law, population)
        if problem:
            raise ValueError(problem)
        units = []
        for s in law["strata"]:
            ranges, k_s = s["units"], s["k"]
            n_s = sum(hi - lo for lo, hi in ranges)
            local = randomness.stream(2 * k_s * _width(n_s) + 64, lambda read: _subset(k_s, n_s, read))
            units += [_nth(ranges, j) for j in local]
        units.sort()
    out = {**dict(law), "population": population, "units": units}
    if closure is not None:
        extra = closure_of(closure, units)
        if extra:
            out["closure"] = extra
    return out


def _nth(ranges: Sequence[Sequence[int]], j: int) -> int:
    for lo, hi in ranges:
        if j < hi - lo:
            return lo + j
        j -= hi - lo
    raise IndexError(j)


def _distinct(units: Iterable[ReplayUnit]) -> list[ReplayUnit]:
    units = list(units)
    if len({u.index for u in units}) != len(units):
        raise ValueError("replay-unit indices must be distinct: each is a draw's address")
    return units


def select_replay_units(randomness: OwnRandomness, law: TwoStageLaw, units: Iterable[ReplayUnit]) -> list[ReplayUnit]:
    """The first stage, in the given order: an RU of a class with a probability is drawn with it, independently; the RUs
    of a class with a one-stage law are drawn together by `draw_first` over their positions in index order, when the
    class's first RU comes up.  A work law's closure is `draw_first`'s, with the class's closure map."""
    units = _distinct(units)
    drawn: dict[str, set[int]] = {}
    out = []
    for u in units:
        stage = law.stage(u.cls)
        if isinstance(stage.first, Fraction):
            if randomness.bernoulli(stage.first):
                out.append(u)
            continue
        if u.cls not in drawn:
            mine = sorted((x for x in units if x.cls == u.cls), key=lambda x: x.index)
            drawn[u.cls] = {mine[j].index for j in draw_first(randomness, stage.first, len(mine))["units"]}
        if u.index in drawn[u.cls]:
            out.append(u)
    return out


def select_proof_units(randomness: OwnRandomness, law: TwoStageLaw, selected: Iterable[ReplayUnit]) -> dict[int, tuple[int, ...]]:
    """The second stage, after the interiors are registered: RU index -> the sorted proof-unit positions (0 .. n_v - 1)
    checked in it, a uniform `k`-subset when the RU holds more than `k`, and otherwise every position, drawing nothing."""
    out = {}
    for u in _distinct(selected):
        k = law.stage(u.cls).checked(u.n_v)
        out[u.index] = randomness.subset(k, u.n_v) if k < u.n_v else tuple(range(u.n_v))
    return out


def check_probability(law: TwoStageLaw, cls: str, n_v: int) -> Fraction:
    """The marginal probability that one given proof unit of an RU of class `cls` holding `n_v` proof units is checked,
    under a Bernoulli first stage (a subset, stratified or work law's depends on the population and the stratum)."""
    stage = law.stage(cls)
    if stage.p is None:
        raise ValueError(f"class {cls!r}: a {stage.first['law']} first stage's marginal depends on the population")
    return stage.p * Fraction(stage.checked(n_v), n_v) if n_v else Fraction(0)


__all__ = ["REPLAY", "OwnRandomness", "ReplayUnit", "Stage", "TwoStageLaw", "check_probability", "closure_of",
           "draw_first", "select_proof_units", "select_replay_units"]
