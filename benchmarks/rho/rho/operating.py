"""Operating points: what the overhead means in time, at the work and exfiltration references (`work.*`, `exfil.*` in
the registry).

* **Work** (`work.<name>.flops`, reference-algorithm FLOPs, `T_0` gates through `hardware.Units`): the inventory does it
  in at least the overhead times `T_0 / Φ_total` seconds at the overhead's lower bound, and in that at the best known
  attack's (`Scenario.seconds`, the idealized normalization); the reference itself in `T_0 / (MFU₀ Φ_total)`
  (`unencumbered`, at the training MFU), and the dark leaves alone in `T_0 / (mfu_dark ε_dark Φ_total)`.
* **Exfiltration** (`exfil.<name>.bits`): onsite, what monitored locations produce reaches the dark leaves only through
  the channels no certifier reads (the certifier's `timing` and `side` bit/s per host, times the location's hosts;
  certified egress is the declared circuit's output, fixed by its registered inputs), from one location or from every
  location at once. Remotely nothing bounds it (0 s).
* **Faulty outputs** (`faulty.*`): a block the sampled proofs let go unchecked returns outputs the adversary chose, in
  certified egress, in either regime. Each faulty step carries `step_bits` and is drawn with chance `q`, and a drawn one
  ends its identity, so time doesn't bound the channel; detection does. Its row gives the bits it carries before the
  first detection, except with chance `δ`, and the chance of detection moving each reference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rho.params import Registry
from rho.scenario import Result, Scenario


@dataclass(frozen=True)
class Point:
    name: str
    kind: str
    #: seconds, by who does it
    seconds: Mapping[str, float]


def _inner(regime: Any) -> Any:
    """The regime a wrapper (`analysis.restrict`'s) runs inside."""
    while hasattr(regime, "regime"):
        regime = regime.regime
    return regime


def work(result: Result, flops: float) -> dict[str, float]:
    s = result.scenario
    f, phi = s.context.fleet, s.total_phi
    gates = s.context.units.gates(flops)
    dark = f.eps_dark * f.dark_mfu * phi
    return {"unencumbered": gates / (f.mfu_train * phi), "lower bound": result.lo * gates / phi,
            "best attack": result.attack * gates / phi, "dark pool": gates / dark if dark > 0 else math.inf}


def exfil(scenario: Scenario, bits: float) -> dict[str, float]:
    certifier = getattr(_inner(scenario.regime), "certifier", None)
    if certifier is None:
        return {"one location": 0.0, "every location": 0.0}
    side = (certifier.timing + certifier.side) * scenario.context.location.hosts
    return {"one location": bits / side if side > 0 else math.inf,
            "every location": bits / (side * scenario.context.fleet.count) if side > 0 else math.inf}


@dataclass(frozen=True)
class Faulty:
    """The faulty-outputs channel: `identities` identities, each ended by its first drawn faulty step."""

    #: the sampled proofs' draw rate per step (1 where every unit is proved)
    q: float
    delta: float
    #: the output bits a faulty step chooses
    step_bits: float
    #: steps in a class drawn at a floor of one
    floor_steps: float
    identities: float = 1.0

    @property
    def capacity(self) -> float:
        """Bits it carries before the first detection, except with chance `δ`: `J ln(1/δ) (1/q + floor_steps)` steps."""
        return self.identities * math.log(1 / self.delta) * (1 / self.q + self.floor_steps) * self.step_bits

    def detect(self, bits: float) -> float:
        """The chance of detection moving `bits`: `1 − exp(−q n)` over its `n = bits / step_bits` steps, at any rate."""
        return -math.expm1(-self.q * bits / self.step_bits)


@dataclass(frozen=True)
class Row:
    name: str
    #: bits, before the first detection at δ
    capacity: float
    #: the chance of detection, by exfiltration reference
    detect: Mapping[str, float]
    #: the draw rate per step and the reporting confidence it was taken at
    q: float = 0.0
    delta: float = math.nan


def faulty(registry: Registry, scenario: Scenario) -> Row:
    """The faulty-outputs row at the regime's identities, at the reporting confidence `faulty.delta`. A sampled regime
    draws at its own rate (`Detection.q`, with its step bits and floor); an unsampled one proves every unit (`q = 1`);
    unmonitored, nothing detects it."""
    refs = {n: registry[f"exfil.{n}.bits"] for n in registry.names("exfil")}
    d = getattr(_inner(scenario.regime), "detection", None)
    delta = registry["faulty.delta"]
    if d is None:
        return Row("faulty outputs", math.inf, dict.fromkeys(refs, 0.0), 0.0, delta)
    if d.sampled:
        f = Faulty(d.q(), delta, d.step_bits, d.floor_steps, d.identities)
    else:
        f = Faulty(1.0, delta, registry["faulty.step_bits"], registry["faulty.floor_steps"], d.identities)
    return Row("faulty outputs", f.capacity, {n: f.detect(b) for n, b in refs.items()}, f.q, delta)


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
