"""A scenario: one choice from every layer, and what ρ it gives.

A scenario names a fleet (with its location and dark share), a regime, the NCI policy, a horizon, a run (the target and
its model), the algorithm families the adversary may use, and the bounds and attacks to apply (all of them unless it
lists some). With `pofc = "<rules>"` its declared work also runs under the proof of forward computation's rules
(`rho.pofc`), and `pofc_physical = true` adds the forward-only links to the server-time block. Every choice is a registry key (`scenario.<name>.<choice>`), so a sweep can vary a choice like any number.

`evaluate` gives, for each family, ρ under each bound alone, under every bound it meets at once (`lo`: each block gets
the least of their caps, under the union of their premises) and under each attack (`hi`: the best construction). The
adversary picks its family, so the scenario's ρ lies in `[min_A lo_A, min_A hi_A]`. A bound above an attack
(`lo > hi`) is a contradiction: one of the bound's premises is false for that algorithm, and `contradictions` names it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from rho import hardware, instances, lean, regimes
from rho.attacks import ATTACKS
from rho.classes import CLASSES, Argument, Outside, premise_status, premises
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
    #: `data/premises.toml` by cited name (`classes.premises`), read from the scenario's registry
    premises: Mapping[str, Mapping] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        """The adversary's whole unencumbered training rate, in reference FLOP/s: the monitored fleet's, and the dark
        pool's share of progress (`Fleet.eps_dark_rate`) beside it."""
        f = self.context.fleet
        return f.count * self.context.location.phi * f.mfu_train / (1 - f.eps_dark_rate)

    def years(self, rho: float) -> float:
        """Years to reach the run's target at slowdown `rho`."""
        return rho * self.run.target(self.model) / self.rate / YEAR


def build(registry: Registry, name: str) -> Scenario:
    t = f"scenario.{name}"
    fl = hardware.fleet(registry, registry[f"{t}.fleet"])
    ctx = Context(fl, hardware.location(registry, fl.location), regimes.policy(registry, registry[f"{t}.policy"]),
                  hardware.horizon(registry, registry[f"{t}.horizon"]))
    run = instances.run(registry, registry[f"{t}.run"])
    s = Scenario(
        name, regimes.regime(registry, registry[f"{t}.regime"]), ctx, instances.model(registry, run.model), run,
        tuple(family(registry, n) for n in registry[f"{t}.families"]),
        tuple(CLASSES[n] for n in registry.get(f"{t}.bounds", list(CLASSES))),
        tuple(ATTACKS[n] for n in registry.get(f"{t}.attacks", list(ATTACKS))), premises(registry))
    if f"{t}.pofc" in registry:
        from rho import pofc  # pofc builds on this module's Scenario

        link = pofc.link(registry, fl.location) if registry.get(f"{t}.pofc_physical", False) else None
        s = pofc.apply(s, pofc.rules(registry, registry[f"{t}.pofc"]), link=link)
    return s


@dataclass(frozen=True)
class Estimate:
    argument: str
    side: str
    #: the weakest status (`lean.weakest`) among `cites`, or `outside`
    status: str
    slowdown: Slowdown | None
    outside: str = ""
    #: what it rests on: the argument's citations and those of the cuts that bound it
    cites: tuple[str, ...] = ()
    #: where its premises stand at the largest block that bounds it (`classes.premise_status`: supported, assumed,
    #: contested, refuted); empty for an attack
    premise: str = ""
    #: ρ inside one inference block `(Γ, k)` alone: `κ Γ / (2 a(Γ, k))`, the declared circuit's slowdown
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


def _cut_cites(partition: Partition, s: Slowdown) -> tuple[str, ...]:
    """The citations of the cut that set each region's share (none where the region's compute did)."""
    cuts = {(r.name, c.name): c for r in partition.regions for c in (*r.cuts, *((r.layout,) if r.layout else ()))}
    return tuple(dict.fromkeys(c for name, share in s.regions.items() if (name, share.cut) in cuts
                               for c in cuts[name, share.cut].cites))


def _binding_k(partition: Partition, s: Slowdown) -> float:
    """The most novel bits of any block in the cuts that set the regions' shares: where premises must hold."""
    cuts = {(r.name, c.name): c for r in partition.regions for c in (*r.cuts, *((r.layout,) if r.layout else ()))}
    return max((b.budget.inputs for name, share in s.regions.items() if (name, share.cut) in cuts
                for b in cuts[name, share.cut].blocks), default=0.0)


def _rated(name: str, side: str, cites: tuple[str, ...], s: Slowdown, partition: Partition, declared: float,
           table: Mapping[str, Mapping]) -> Estimate:
    cites = tuple(dict.fromkeys(cites + _cut_cites(partition, s)))
    premise = premise_status(cites, _binding_k(partition, s), table=table) if side == "bound" else ""
    return Estimate(name, side, lean.weakest(cites), s, cites=cites, declared=declared, premise=premise)


def _estimate(arg: Argument, p: Profile, partition: Partition, block: Budget,
              table: Mapping[str, Mapping]) -> tuple[Estimate, Allowance | None]:
    try:
        a = arg.allowance(p)
    except Outside as e:
        return Estimate(arg.name, arg.side, "outside", None, str(e)), None
    s = slowdown(partition, a, p.kappa, p.efficiency, arg.side)
    return _rated(arg.name, arg.side, arg.cites, s, partition, _declared(a, p, block), table), a


def _declared(a: Allowance, p: Profile, block: Budget) -> float:
    n = a(block)
    return max(1.0, p.kappa * block.compute / (2 * n)) if n > 0 else math.inf


def _combine(args: tuple[Argument, ...], bounds: list[tuple[Estimate, Allowance | None]], p: Profile,
             partition: Partition, block: Budget, table: Mapping[str, Mapping]) -> Estimate:
    """Every bound met at once: each block gets the least of their caps, under the union of their citations."""
    met = [a for _, a in bounds if a is not None]
    cites = tuple(c for b, (_, a) in zip(args, bounds, strict=True) if a is not None for c in b.cites)
    return _rated("+".join(e.argument for e, a in bounds if a is not None) or "none", "bound", cites,
                  slowdown(partition, least(*met), p.kappa, p.efficiency), partition, _declared(least(*met), p, block),
                  table)


def evaluate(s: Scenario) -> Result:
    partition = s.regime.partition(s.context)
    block = Budget(s.context.policy.gamma, s.context.policy.k)
    out = []
    for fam in s.families:
        p = fam.profile(s.model, s.run)
        bounds = [_estimate(b, p, partition, block, s.premises) for b in s.bounds]
        combined = _combine(s.bounds, bounds, p, partition, block, s.premises)
        out.append(FamilyResult(fam.name, p, tuple(e for e, _ in bounds), combined,
                                tuple(_estimate(a, p, partition, block, s.premises)[0] for a in s.attacks)))
    return Result(s, tuple(out))


def reads(registry: Registry, name: str) -> frozenset[str]:
    """The registry keys scenario `name` reads: the parameters its ρ can depend on."""
    tracked = registry.tracked()
    build(tracked, name)
    return frozenset(tracked.seen or ())


def evaluate_named(registry: Registry, name: str) -> Result:
    return evaluate(build(registry, name))
