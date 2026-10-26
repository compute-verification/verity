"""A scenario: one choice from every layer, and what ρ it gives.

A scenario names a fleet (with its location and dark share), a regime, the NCI policy, a horizon, a run (the target and
its model), the algorithm families the adversary may use, and the bounds and attacks to apply (all of them unless it
lists some). Every choice is a registry key (`scenario.<name>.<choice>`), so a sweep can vary a choice like any number.

`evaluate` gives, for each family, ρ under each bound alone, under every bound it meets at once (`lo`: each block gets
the least of their caps, under the union of their premises) and under each attack (`hi`: the best construction). The
adversary picks its family, so the scenario's ρ lies in `[min_A lo_A, min_A hi_A]`. A bound above an attack
(`lo > hi`) is a contradiction: one of the bound's premises is false for that algorithm, and `contradictions` names it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rho import hardware, instances, lean, regimes
from rho.attacks import ATTACKS
from rho.classes import CLASSES, Argument, Outside
from rho.families import Family, family
from rho.instances import Model, Run
from rho.params import Registry
from rho.profile import Profile
from rho.regimes import Context, Regime
from rho.theorem import Allowance, Budget, Partition, Slowdown, least, slowdown

YEAR = 365 * 86400.0


@dataclass(frozen=True)
class Scenario:
    name: str
    regime: Regime
    context: Context
    model: Model
    run: Run
    families: tuple[Family, ...]
    bounds: tuple[Argument, ...]
    attacks: tuple[Argument, ...]

    @property
    def rate(self) -> float:
        """The adversary's whole compute, unencumbered, in FLOP/s: the monitored fleet and its dark share."""
        f = self.context.fleet
        return f.count * self.context.location.phi * f.mfu_train / (1 - f.dark)

    def years(self, rho: float) -> float:
        """Years to reach the run's target at slowdown `rho`."""
        return rho * self.run.target(self.model) / self.rate / YEAR


def build(registry: Registry, name: str) -> Scenario:
    t = f"scenario.{name}"
    fl = hardware.fleet(registry, registry[f"{t}.fleet"])
    ctx = Context(fl, hardware.location(registry, fl.location), regimes.policy(registry, registry[f"{t}.policy"]),
                  hardware.horizon(registry, registry[f"{t}.horizon"]))
    run = instances.run(registry, registry[f"{t}.run"])
    return Scenario(
        name, regimes.regime(registry, registry[f"{t}.regime"]), ctx, instances.model(registry, run.model), run,
        tuple(family(registry, n) for n in registry[f"{t}.families"]),
        tuple(CLASSES[n] for n in registry.get(f"{t}.bounds", list(CLASSES))),
        tuple(ATTACKS[n] for n in registry.get(f"{t}.attacks", list(ATTACKS))))


@dataclass(frozen=True)
class Estimate:
    argument: str
    side: str
    #: the weakest Lean status among its citations (`lean.weakest`), or `outside`
    status: str
    slowdown: Slowdown | None
    outside: str = ""
    #: ρ inside one declared block `(Γ, k)` alone: `κ Γ / (2 a(Γ, k))`, the declared circuit's slowdown
    declared: float = math.nan

    @property
    def rho(self) -> float:
        return self.slowdown.rho if self.slowdown else math.nan


@dataclass(frozen=True)
class FamilyResult:
    family: str
    profile: Profile
    bounds: tuple[Estimate, ...]
    combined: Estimate
    attacks: tuple[Estimate, ...]

    @property
    def lo(self) -> float:
        return self.combined.rho

    @property
    def hi(self) -> float:
        return min((a.rho for a in self.attacks if a.slowdown), default=math.inf)

    @property
    def contradictions(self) -> tuple[str, ...]:
        return tuple(b.argument for b in self.bounds if b.slowdown and b.rho > self.hi * (1 + 1e-9))


@dataclass(frozen=True)
class Result:
    scenario: Scenario
    families: tuple[FamilyResult, ...]

    @property
    def lo(self) -> float:
        return min((f.lo for f in self.families), default=math.inf)

    @property
    def hi(self) -> float:
        return min((f.hi for f in self.families), default=math.inf)


def _estimate(arg: Argument, p: Profile, partition: Partition, block: Budget) -> tuple[Estimate, Allowance | None]:
    try:
        a = arg.allowance(p)
    except Outside as e:
        return Estimate(arg.name, arg.side, "outside", None, str(e)), None
    return Estimate(arg.name, arg.side, lean.weakest(arg.cites), slowdown(partition, a, p.kappa, p.efficiency, arg.side),
                    declared=_declared(a, p, block)), a


def _declared(a: Allowance, p: Profile, block: Budget) -> float:
    n = a(block)
    return max(1.0, p.kappa * block.compute / (2 * n)) if n > 0 else math.inf


def evaluate(s: Scenario) -> Result:
    partition = s.regime.partition(s.context)
    block = Budget(s.context.policy.gamma, s.context.policy.k)
    out = []
    for fam in s.families:
        p = fam.profile(s.model, s.run)
        bounds = [_estimate(b, p, partition, block) for b in s.bounds]
        met = [a for _, a in bounds if a is not None]
        cites = [c for b, (_, a) in zip(s.bounds, bounds, strict=True) if a is not None for c in b.cites]
        combined = Estimate("+".join(e.argument for e, a in bounds if a is not None) or "none", "bound",
                            lean.weakest(cites), slowdown(partition, least(*met), p.kappa, p.efficiency),
                            declared=_declared(least(*met), p, block))
        out.append(FamilyResult(fam.name, p, tuple(e for e, _ in bounds), combined,
                                tuple(_estimate(a, p, partition, block)[0] for a in s.attacks)))
    return Result(s, tuple(out))


def evaluate_named(registry: Registry, name: str) -> Result:
    return evaluate(build(registry, name))
