"""Sensitivity: how the overhead moves with every parameter, and where it falls below a target.

Every function takes a registry and a scenario name and varies registry keys, so any parameter of any layer (a GPU's
FLOP/s, a wipe's `k_wipe`, a premise's bits, a family's efficiency, a choice of regime) is swept the same way. The
metric is a function of a `Result`: by default the overhead's lower bound `LO` (the least over the adversary's
families); `ATTACK` is the best known attack's overhead, and `upto(status)` is `LO` from only the bounds and cuts at
least as strong as `status`. Results are cached per registry and override, so the functions below share their
evaluations.

* `sweep`: the overhead along one key.
* `tornado`: the overhead at each ranged parameter's `low` and `high` (the registry's plausible range), the rest held,
  sorted by swing.
* `threshold`: the value of one key at which the overhead crosses a target, by bisection (`crossing`).
* `dangers`: the ranged parameters whose plausible range alone takes the overhead below the target, and where;
  `rescues`, those whose range alone lifts it above.
* `corner`: every ranged parameter at its adversarial end at once.
* `margins`: for every ranged parameter, the overhead along a window around its plausible range and where it crosses
  each target.
* `phase`: the overhead over a grid of two keys; `frontier`, the curve in that plane where it crosses a target.
* `attribution`: what sets the overhead: the bound's weight per gate split into the dark leaves and g's terms.
"""

from __future__ import annotations

import json
import math
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

from rho import lean
from rho.envelopes import Context, Delta, Regime
from rho.params import Param, Registry
from rho.scenario import Result, Scenario, evaluate, evaluate_named, reads
from rho.theorem import Partition

Metric = Callable[[Result], float]

#: from strongest to weakest, as `lean.weakest` names them
STATUSES = ("proved", "conditional", "draft", "absent")


def LO(r: Result) -> float:
    return r.lo


def ATTACK(r: Result) -> float:
    return r.attack


def _rank(cites: Iterable[str]) -> int:
    return STATUSES.index(lean.weakest(cites))


@dataclass(frozen=True)
class _Restricted:
    """A regime whose cuts are only those at least as strong as `status` (a serving block left with none is
    unconstrained)."""

    regime: Regime
    status: str

    @property
    def name(self) -> str:
        return self.regime.name

    def partition(self, ctx: Context) -> Partition:
        p = self.regime.partition(ctx)
        top = STATUSES.index(self.status)
        return replace(p, serving=tuple(replace(b, cuts=tuple(c for c in b.cuts if _rank(c.cites) <= top))
                                        for b in p.serving))

    def delta(self, ctx: Context) -> Delta:
        return self.regime.delta(ctx)


def restrict(s: Scenario, status: str) -> Scenario:
    """`s` with only the bounds and cuts whose status is `status` or stronger, and no attacks; the uncertified and dark
    gates at the elementary cap where `s_×`'s straddling premise is weaker than `status`."""
    top = STATUSES.index(status)
    st = s.structure if _rank(s.structure.cites()) <= top else s.structure.elementary()
    return replace(s, regime=_Restricted(s.regime, status), bounds=tuple(b for b in s.bounds if _rank(b.cites) <= top),
                   attacks=(), structure=st)


@cache
def upto(status: str) -> Metric:
    """`LO` from only the bounds and cuts at least as strong as `status`: `upto("conditional")` drops the draft
    statements (the straddling premise among them, so the uncertified and dark gates count at the elementary cap),
    `upto("proved")` keeps only what needs no named premise."""
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {STATUSES}")

    def metric(r: Result) -> float:
        return evaluate(restrict(r.scenario, status)).lo

    metric.__name__ = metric.__qualname__ = f"upto({status!r})"
    return metric


_results: weakref.WeakKeyDictionary[Registry, dict[str, Result]] = weakref.WeakKeyDictionary()
CACHE = 50_000


def result(registry: Registry, scenario: str, values: Mapping[str, Any] | None = None) -> Result:
    """The scenario evaluated with `values` in place of the registry's."""
    key = json.dumps([scenario, sorted((values or {}).items())], default=str)
    memo = _results.setdefault(registry, {})
    if key not in memo:
        if len(memo) >= CACHE:
            memo.clear()
        memo[key] = evaluate_named(registry.override(values or {}), scenario)
    return memo[key]


def at(registry: Registry, scenario: str, values: Mapping[str, Any] | None = None, metric: Metric = LO) -> float:
    return metric(result(registry, scenario, values))


def sweep(registry: Registry, scenario: str, key: str, values: Iterable[Any], metric: Metric = LO,
          given: Mapping[str, Any] | None = None) -> list[tuple[Any, float]]:
    return [(v, at(registry, scenario, {**(given or {}), key: v}, metric)) for v in values]


@dataclass(frozen=True)
class Swing:
    key: str
    low: float
    high: float
    at_low: float
    at_high: float
    base: float

    @property
    def size(self) -> float:
        """The swing in decades of the overhead."""
        vals = [v for v in (self.at_low, self.at_high, self.base) if v > 0 and math.isfinite(v)]
        return math.log10(max(vals) / min(vals)) if len(vals) > 1 else (math.inf if vals else 0.0)

    @property
    def worse(self) -> tuple[float, float]:
        """The end of the range with the lower overhead, and the overhead there."""
        return (self.low, self.at_low) if self.at_low <= self.at_high else (self.high, self.at_high)

    @property
    def better(self) -> tuple[float, float]:
        return (self.high, self.at_high) if self.at_low <= self.at_high else (self.low, self.at_low)


def ranged(registry: Registry, keys: Iterable[str] | None = None) -> list[str]:
    keys = list(keys) if keys is not None else [p.key for p in registry.params() if p.ranged]
    return [k for k in keys if registry.param(k).ranged]


def tornado(registry: Registry, scenario: str, keys: Iterable[str] | None = None, metric: Metric = LO) -> list[Swing]:
    """Keys default to the ranged ones the scenario reads (`scenario.reads`)."""
    base = at(registry, scenario, None, metric)
    out = []
    for k in ranged(registry, sorted(reads(registry, scenario)) if keys is None else keys):
        p = registry.param(k)
        out.append(Swing(k, p.low, p.high, at(registry, scenario, {k: p.low}, metric),
                         at(registry, scenario, {k: p.high}, metric), base))
    return sorted((s for s in out if s.size > 0), key=lambda s: -s.size)


def crossing(f: Callable[[float], float], target: float, a: float, b: float, rtol: float = 1e-4,
             steps: int = 80) -> float | None:
    """Where `f` crosses `target` between `a` and `b` (geometric bisection when both ends are positive), or None when
    `f − target` has the same sign at both ends."""
    fa, fb = f(a) - target, f(b) - target
    if math.isnan(fa) or math.isnan(fb):
        return None
    if fa == 0:
        return a
    if fb == 0:
        return b
    if (fa > 0) == (fb > 0):
        return None
    geometric = a > 0 and b > 0
    for _ in range(steps):
        if (abs(math.log(b / a)) if geometric else abs(b - a)) <= rtol * (1 if geometric else max(abs(a), abs(b))):
            break
        m = math.sqrt(a * b) if geometric else (a + b) / 2
        fm = f(m) - target
        if fm == 0:
            return m
        if (fa > 0) != (fm > 0):
            b, fb = m, fm
        else:
            a, fa = m, fm
    return math.sqrt(a * b) if geometric else (a + b) / 2


def threshold(registry: Registry, scenario: str, key: str, target: float, lo: float, hi: float,
              metric: Metric = LO, steps: int = 80, given: Mapping[str, Any] | None = None,
              rtol: float = 1e-4) -> float | None:
    """The value of `key` in `[lo, hi]` at which the metric crosses `target`, with the keys in `given` held at their
    values, or None when it doesn't cross there."""
    return crossing(lambda v: at(registry, scenario, {**(given or {}), key: v}, metric), target, lo, hi, rtol, steps)


@dataclass(frozen=True)
class Danger:
    key: str
    #: the value at which the overhead crosses the target, and the worse end of the plausible range
    crossing: float
    worst: float
    rho_worst: float


def dangers(registry: Registry, scenario: str, target: float, keys: Iterable[str] | None = None,
            metric: Metric = LO, swings: Sequence[Swing] | None = None) -> list[Danger]:
    """The ranged parameters whose plausible range alone takes the overhead below `target`, each with where it
    crosses."""
    out = []
    for s in tornado(registry, scenario, keys, metric) if swings is None else swings:
        worst, rho_worst = s.worse
        if rho_worst < target <= s.base:
            x = threshold(registry, scenario, s.key, target, registry[s.key], worst, metric)
            if x is not None:
                out.append(Danger(s.key, x, worst, rho_worst))
    return out


def rescues(registry: Registry, scenario: str, target: float, keys: Iterable[str] | None = None,
            metric: Metric = LO, swings: Sequence[Swing] | None = None) -> list[Danger]:
    """The ranged parameters whose plausible range alone lifts the overhead from below `target` to it, each with where
    it crosses (a `Danger` read the other way: `worst` is the better end, and `rho_worst` the overhead there)."""
    out = []
    for s in tornado(registry, scenario, keys, metric) if swings is None else swings:
        best, rho_best = s.better
        if s.base < target <= rho_best:
            x = threshold(registry, scenario, s.key, target, registry[s.key], best, metric)
            if x is not None:
                out.append(Danger(s.key, x, best, rho_best))
    return out


@dataclass(frozen=True)
class Corner:
    #: each ranged parameter that lowers the overhead, at the end of its range that lowers it more
    values: Mapping[str, float]
    rho: float
    base: float
    #: the overhead with every such parameter at its worse end at once (the corner, unless an interaction makes it
    #: higher)
    joint: float
    #: the overhead as the parameters move to their worse ends one at a time, the worst alone first; a move that would
    #: raise it is
    #: left out
    path: tuple[tuple[str, float], ...]


def corner(registry: Registry, scenario: str, keys: Iterable[str] | None = None, metric: Metric = LO,
           swings: Sequence[Swing] | None = None) -> Corner:
    """Every ranged parameter at the end of its plausible range where the overhead alone is lower, all at once. Never
    above the base: when an interaction makes the joint corner higher than the path that adds one parameter at a time,
    the path's end is the corner."""
    swings = tornado(registry, scenario, keys, metric) if swings is None else swings
    base = at(registry, scenario, None, metric)
    worse = sorted((s for s in swings if s.worse[1] < base), key=lambda s: s.worse[1])
    joint_values = {s.key: s.worse[0] for s in worse}
    joint = at(registry, scenario, joint_values, metric)
    values: dict[str, float] = {}
    rho, path = base, []
    for s in worse:
        v = at(registry, scenario, {**values, s.key: s.worse[0]}, metric)
        if v <= rho:
            values[s.key] = s.worse[0]
            rho = v
            path.append((s.key, v))
    if joint < rho:
        values, rho = joint_values, joint
    return Corner(values, rho, base, joint, tuple(path))


def window(p: Param, widen: float = 1.0) -> tuple[float, float]:
    """A window around `p`'s plausible range: positive ranges grow by their own ratio to the power `widen` (at least a
    decade) at each end, others by ten widths; a range inside `[0, 1]` (a share, a utilization) stays inside it."""
    lo, hi = sorted((float(p.low), float(p.high)))
    cap = 1.0 if 0 <= lo and hi <= 1 else math.inf
    if lo > 0:
        f = max((hi / lo) ** widen, 10.0)
        return lo / f, min(hi * f, cap)
    w = 10 * (hi - lo) * widen
    return (max(lo - w, 0.0) if lo >= 0 else lo - w), min(hi + w, cap)


def position(p: Param, v: float) -> float:
    """Where `v` lies relative to `p`'s plausible range: 0 at its low end, 1 at its high end (logarithmically when the
    range is positive)."""
    lo, hi = float(p.low), float(p.high)
    if lo > 0 and hi > 0 and v > 0:
        return math.log(v / lo) / math.log(hi / lo) if hi != lo else 0.0
    return (v - lo) / (hi - lo) if hi != lo else 0.0


def unposition(p: Param, u: float) -> float:
    lo, hi = float(p.low), float(p.high)
    if lo > 0 and hi > 0:
        return lo * (hi / lo) ** u
    return lo + u * (hi - lo)


@dataclass(frozen=True)
class Margin:
    key: str
    value: float
    low: float
    high: float
    window: tuple[float, float]
    #: the overhead at points across the window
    samples: tuple[tuple[float, float], ...]
    #: target → the crossings nearest the current value, below and above it, within the window (None: none there)
    crossings: Mapping[float, tuple[float | None, float | None]]


def margins(registry: Registry, scenario: str, targets: Sequence[float], keys: Iterable[str] | None = None,
            metric: Metric = LO, widen: float = 1.0, points: int = 25, swings: Sequence[Swing] | None = None,
            rtol: float = 1e-3) -> list[Margin]:
    """For each parameter that moves the overhead (in tornado order): the overhead across a `window` around its
    plausible range, and where it crosses each target between the current value and either end of the window."""
    out = []
    for s in tornado(registry, scenario, keys, metric) if swings is None else swings:
        p = registry.param(s.key)
        a, b = window(p, widen)
        ua, ub = position(p, a), position(p, b)
        xs = [unposition(p, ua + (ub - ua) * i / (points - 1)) for i in range(points)]
        samples = tuple(sweep(registry, scenario, s.key, xs, metric))
        v = float(p.value)
        crossings = {t: (threshold(registry, scenario, s.key, t, v, a, metric, rtol=rtol) if a < v else None,
                         threshold(registry, scenario, s.key, t, v, b, metric, rtol=rtol) if v < b else None)
                     for t in targets}
        out.append(Margin(s.key, v, p.low, p.high, (a, b), samples, crossings))
    return out


def phase(registry: Registry, scenario: str, x: str, xs: Sequence[Any], y: str, ys: Sequence[Any],
          metric: Metric = LO) -> list[list[float]]:
    """The overhead at each `(x, y)`: rows over `ys`, columns over `xs`."""
    return [[at(registry, scenario, {x: xv, y: yv}, metric) for xv in xs] for yv in ys]


def frontier(registry: Registry, scenario: str, x: str, xs: Sequence[float], y: str, ylo: float, yhi: float,
             target: float, metric: Metric = LO, rtol: float = 1e-3) -> list[tuple[float, float | None]]:
    """The curve in the `(x, y)` plane where the overhead crosses `target`: for each `x`, the `y` in `[ylo, yhi]` at
    which it does (None where it doesn't cross there)."""
    return [(xv, threshold(registry, scenario, y, target, ylo, yhi, metric, given={x: xv}, rtol=rtol)) for xv in xs]


@dataclass(frozen=True)
class Term:
    #: `dark` (the dark leaves), or one of g's terms: a block kind or an uncertified route (`Overhead.terms`)
    name: str
    #: its part of the overhead's rate, `1 / overhead` before its floor at 1 (`Overhead.rates`), and its share of it
    part: float
    share: float
    #: `1 / part`: the overhead can't exceed it while this term remains
    ceiling: float


def attribution(r: Result, side: str = "bound") -> list[Term]:
    """What sets the overhead, largest first: its rate (`Overhead.rates`) split into the dark leaves and g's terms, for
    the family the adversary picks (the least `lo`; with `side="attack"`, its best known attack's)."""
    if side == "bound":
        fam = min(r.families, key=lambda f: (f.lo, f.combined.padded), default=None)
        s = fam.combined.overhead if fam else None
    else:
        fam = min(r.families, key=lambda f: f.attack, default=None)
        best = fam.best_attack if fam else None
        s = best.overhead if best else None
    if s is None:
        return []
    parts = s.rates()
    rate = math.fsum(parts.values())
    return sorted((Term(t, v, v / rate, 1 / v) for t, v in parts.items() if v > 0),
                  key=lambda t: -t.part)
