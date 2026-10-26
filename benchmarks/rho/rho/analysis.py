"""Sensitivity: how ρ moves with every parameter, and where it falls below a target.

Every function takes a registry and a scenario name and varies registry keys, so any parameter of any layer (a GPU's
FLOP/s, a wipe's residual, a premise's bits, a family's efficiency, a choice of regime) is swept the same way. The
metric is a function of a `Result`, by default the guaranteed slowdown `lo` (the least lower bound over the
adversary's families); `HI` is the best attack's.

* `sweep`: ρ along one key.
* `tornado`: ρ at each ranged parameter's `low` and `high` (the registry's plausible range), the rest held, sorted by
  swing.
* `threshold`: the value of one key at which ρ crosses a target, by bisection on its range.
* `dangers`: for every ranged parameter, whether its plausible range reaches below the target, and where.
* `phase`: ρ over a grid of two keys.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rho.params import Registry
from rho.scenario import Result, evaluate_named

Metric = Callable[[Result], float]


def LO(r: Result) -> float:
    return r.lo


def HI(r: Result) -> float:
    return r.hi


def at(registry: Registry, scenario: str, values: Mapping[str, Any] | None = None, metric: Metric = LO) -> float:
    return metric(evaluate_named(registry.override(values or {}), scenario))


def sweep(registry: Registry, scenario: str, key: str, values: Iterable[Any], metric: Metric = LO) -> list[tuple[Any, float]]:
    return [(v, at(registry, scenario, {key: v}, metric)) for v in values]


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
        """The swing in decades of ρ."""
        vals = [v for v in (self.at_low, self.at_high, self.base) if v > 0 and math.isfinite(v)]
        return math.log10(max(vals) / min(vals)) if len(vals) > 1 else (math.inf if vals else 0.0)


def ranged(registry: Registry, keys: Iterable[str] | None = None) -> list[str]:
    keys = list(keys) if keys is not None else [p.key for p in registry.params() if p.ranged]
    return [k for k in keys if registry.param(k).ranged]


def tornado(registry: Registry, scenario: str, keys: Iterable[str] | None = None, metric: Metric = LO) -> list[Swing]:
    base = at(registry, scenario, None, metric)
    out = []
    for k in ranged(registry, keys):
        p = registry.param(k)
        out.append(Swing(k, p.low, p.high, at(registry, scenario, {k: p.low}, metric),
                         at(registry, scenario, {k: p.high}, metric), base))
    return sorted((s for s in out if s.size > 0), key=lambda s: -s.size)


def threshold(registry: Registry, scenario: str, key: str, target: float, lo: float, hi: float,
              metric: Metric = LO, steps: int = 60) -> float | None:
    """The value of `key` in `[lo, hi]` at which the metric crosses `target` (geometric bisection when both ends are
    positive), or None when it doesn't cross there."""
    def f(v: float) -> float:
        return at(registry, scenario, {key: v}, metric) - target

    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fa * fb > 0 or math.isnan(fa) or math.isnan(fb):
        return None
    geometric = a > 0 and b > 0
    for _ in range(steps):
        m = math.sqrt(a * b) if geometric else (a + b) / 2
        fm = f(m)
        if fm == 0:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return math.sqrt(a * b) if geometric else (a + b) / 2


@dataclass(frozen=True)
class Danger:
    key: str
    #: the value at which ρ crosses the target, and the worse end of the plausible range
    crossing: float
    worst: float
    rho_worst: float


def dangers(registry: Registry, scenario: str, target: float, keys: Iterable[str] | None = None,
            metric: Metric = LO) -> list[Danger]:
    """The ranged parameters whose plausible range alone takes ρ below `target`, each with where it crosses."""
    out = []
    for s in tornado(registry, scenario, keys, metric):
        worst, rho_worst = (s.low, s.at_low) if s.at_low <= s.at_high else (s.high, s.at_high)
        if rho_worst < target <= s.base:
            x = threshold(registry, scenario, s.key, target, min(s.low, s.high), max(s.low, s.high), metric)
            if x is not None:
                out.append(Danger(s.key, x, worst, rho_worst))
    return out


def phase(registry: Registry, scenario: str, x: str, xs: Sequence[Any], y: str, ys: Sequence[Any],
          metric: Metric = LO) -> list[list[float]]:
    """ρ at each `(x, y)`: rows over `ys`, columns over `xs`."""
    return [[at(registry, scenario, {x: xv, y: yv}, metric) for xv in xs] for yv in ys]
