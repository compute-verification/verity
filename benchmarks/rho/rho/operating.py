"""Operating points: what ρ means in time, at the work and exfiltration references (`work.*`, `exfil.*` in the registry).

* **Work** (`work.<name>.flops`, reference-algorithm FLOPs): the adversary's whole compute does it in `ρ · flops / rate`
  seconds (`Scenario.rate` is its unencumbered FLOP/s), at the guaranteed ρ and at the best attack's; its dark pool alone
  in `flops / (d · rate)`, `d` its share of progress (`Fleet.dark_effective`).
* **Exfiltration** (`exfil.<name>.bits`): onsite, what monitored locations produce reaches the dark pool only through
  the channels no certifier reads (the regime's `timing` and `side` bit/s per host, times the location's hosts;
  certified egress is the declared circuit's output, fixed by its registered inputs), from one location or from every
  location at once. Remotely nothing bounds it (0 s).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from rho.params import Registry
from rho.scenario import Result, Scenario


@dataclass(frozen=True)
class Point:
    name: str
    kind: str
    #: seconds, by who does it
    seconds: Mapping[str, float]


def work(result: Result, flops: float) -> dict[str, float]:
    s = result.scenario
    d = s.context.fleet.dark_effective
    return {"unencumbered": flops / s.rate, "guaranteed": result.lo * flops / s.rate,
            "best attack": result.hi * flops / s.rate, "dark pool": flops / (d * s.rate) if d > 0 else math.inf}


def exfil(scenario: Scenario, bits: float) -> dict[str, float]:
    if not hasattr(scenario.regime, "side"):
        return {"one location": 0.0, "every location": 0.0}
    side = (getattr(scenario.regime, "timing", 0.0) + scenario.regime.side) * scenario.context.location.hosts
    return {"one location": bits / side if side > 0 else math.inf,
            "every location": bits / (side * scenario.context.fleet.count) if side > 0 else math.inf}


def points(registry: Registry, result: Result) -> list[Point]:
    out = [Point(n, "work", work(result, registry[f"work.{n}.flops"])) for n in registry.names("work")]
    out += [Point(n, "exfil", exfil(result.scenario, registry[f"exfil.{n}.bits"])) for n in registry.names("exfil")]
    return out


def human(seconds: float) -> str:
    """`seconds` in the largest unit that keeps it at least 1 (s, min, h, days, years)."""
    if math.isinf(seconds):
        return "never"
    for unit, size in (("years", 365 * 86400.0), ("days", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} s"
