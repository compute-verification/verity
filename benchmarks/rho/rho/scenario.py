"""A scenario: one choice from every layer, and what overhead it gives.

A scenario names a fleet (with its location and dark share), a regime, the NCI policy, a horizon, a run (the target and
its model), the algorithm families the adversary may use, and the bounds and attacks to apply (all of them unless it
lists some). Every choice is a registry key (`scenario.<name>.<choice>`), so a sweep can vary a choice like any number.

`evaluate` gives, for each family, the overhead under each bound alone and under every bound it meets at once (`lo`,
the lower bound: each block gets the least of the caps whose premises aren't refuted at its bits, under the union of
those bounds' premises), and under each attack (`best_attack`: the best known construction, reported beside the bound,
not as an end of it). The adversary picks its family, so the scenario's overhead is at least `min_A lo_A`. A bound
above an attack is a contradiction: one of the bound's premises is false for that algorithm, and `contradictions` names
it.

The overhead is the hardware multiple over the reference training algorithm at its MFU, `MFU₀` (the scenario's `mfu`),
at least 1: Lean's `hardwareMultiple`, and for a remote regime `Nci.Guarantees.DampenerRemote`'s `ρ`
(`theorem.Overhead`). The padded size ratio `|A| / T_0`, each serving block padded to its capacity, is the intermediate
the bound gives (`Overhead.padded`), and a contradiction compares it. The time to the target, wall-clock, is reported
beside the overhead: at least the overhead times `T_0 / (MFU₀ Φ_total)`, the reference's own time (`Scenario.seconds`).

A scenario's `structure` (else `structure.default`, `reference`; `data/exclusion.toml`) is the elementary uncertified
cap's `s_×`, from the same source as `units.gates_per_flop` (`Structure.check`); a bound that counts uncertified or dark
gates at more than one gate an operation cites its straddling premise, and the runs that measured `s_×`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from rho import envelopes, hardware, instances, lean
from rho.attacks import ATTACKS
from rho.classes import (
    CLASSES,
    STANDINGS,
    Argument,
    Outside,
    Structure,
    per_block,
    premise_status,
    premises,
    structure,
    weight_per_gate,
)
from rho.envelopes import ROUTES, Context, Regime
from rho.families import Family, family
from rho.instances import Model, Run
from rho.params import Registry
from rho.profile import Profile
from rho.theorem import Budget, Cap, Overhead, Partition, least, overhead

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
    #: the elementary uncertified cap a bound counts uncertified and dark gates at
    structure: Structure
    #: `data/premises.toml` by cited name (`classes.premises`), read from the scenario's registry
    premises: Mapping[str, Mapping] = field(default_factory=dict)
    #: reference efficiency (`task.reference_efficiency`): a bound counts free compute at `MFU₀`, not at peak
    reference_efficiency: bool = True

    def __post_init__(self) -> None:
        self.structure.check(self.context.units)

    @property
    def mfu(self) -> float:
        """The inventory's share of capacity a construction trains with, the monitored fleet's and the dark leaves':
        `MFU₀`, the reference training algorithm's, which the overhead is a multiple against."""
        f = self.context.fleet
        return f.eps_dark * f.dark_mfu + (1 - f.eps_dark) * f.mfu_train

    @property
    def total_phi(self) -> float:
        """`Φ_total`: the inventory's gates per second, the monitored fleet's and the dark leaves' (`ε_dark` of it)."""
        f = self.context.fleet
        return f.count * self.context.location.phi / (1 - f.eps_dark)

    @property
    def reference(self) -> float:
        """`T_0`: the reference circuit's gates to the run's target (its FLOPs through `hardware.Units`)."""
        return self.context.units.gates(self.run.target(self.model))

    def seconds(self, overhead: float) -> float:
        """`overhead T_0 / (MFU₀ Φ_total)`: the time to the run's target at `overhead`, a multiple of the reference's."""
        return overhead * self.reference / (self.mfu * self.total_phi)

    def years(self, overhead: float) -> float:
        return self.seconds(overhead) / YEAR


def build(registry: Registry, name: str) -> Scenario:
    t = f"scenario.{name}"
    fl = hardware.fleet(registry, registry[f"{t}.fleet"])
    ctx = Context(fl, hardware.location(registry, fl.location), envelopes.policy(registry, registry[f"{t}.policy"]),
                  hardware.horizon(registry, registry[f"{t}.horizon"]), hardware.units(registry))
    run = instances.run(registry, registry[f"{t}.run"])
    return Scenario(
        name, envelopes.regime(registry, registry[f"{t}.regime"]), ctx, instances.model(registry, run.model), run,
        tuple(family(registry, n) for n in registry[f"{t}.families"]),
        tuple(CLASSES[n] for n in registry.get(f"{t}.bounds", list(CLASSES))),
        tuple(ATTACKS[n] for n in registry.get(f"{t}.attacks", list(ATTACKS))),
        structure(registry, registry.get(f"{t}.structure", registry["structure.default"])), premises(registry),
        bool(registry["task.reference_efficiency"]))


@dataclass(frozen=True)
class Estimate:
    argument: str
    side: str
    #: the weakest status (`lean.weakest`) among `cites`, or `outside`
    status: str
    overhead: Overhead | None
    outside: str = ""
    #: what it rests on: the argument's citations and those of the cuts that bound it
    cites: tuple[str, ...] = ()
    #: where its premises stand at the largest block that bounds it (`classes.premise_status`: supported, assumed,
    #: contested, refuted); empty for an attack
    premise: str = ""
    #: the overhead inside one inference block `(c, k)` alone: `w c / q_T(c, k)`, the registered circuit's overhead
    declared: float = math.nan
    #: every bound at once only: the bounds left out at a block that binds, their premises refuted there
    excluded: tuple[str, ...] = ()

    @property
    def rho(self) -> float:
        """The overhead: the hardware multiple over the reference at its MFU."""
        return self.overhead.rho if self.overhead else math.nan

    @property
    def padded(self) -> float:
        """The padded size ratio `|A| / T_0`."""
        return self.overhead.padded if self.overhead else math.nan


@dataclass(frozen=True)
class FamilyResult:
    family: str
    profile: Profile
    bounds: tuple[Estimate, ...]
    combined: Estimate
    attacks: tuple[Estimate, ...]

    @property
    def lo(self) -> float:
        """The overhead's lower bound: every bound met at once."""
        return self.combined.rho

    @property
    def best_attack(self) -> Estimate | None:
        """The best known attack: the construction with the least overhead (None: none applies)."""
        return min((a for a in self.attacks if a.overhead), key=lambda a: (a.rho, a.padded), default=None)

    @property
    def attack(self) -> float:
        """The best known attack's overhead (∞: none applies), beside the lower bound and not an end of it."""
        best = self.best_attack
        return best.rho if best else math.inf

    @property
    def contradictions(self) -> tuple[str, ...]:
        """The bounds above the best known attack, in the padded size ratio the bound states: one of each one's premises
        is false for this family."""
        best = self.best_attack
        return tuple(b.argument for b in self.bounds if b.overhead and best and b.padded > best.padded * (1 + 1e-9))


@dataclass(frozen=True)
class Result:
    scenario: Scenario
    families: tuple[FamilyResult, ...]

    @property
    def lo(self) -> float:
        """The overhead's lower bound, at the family the adversary picks."""
        return min((f.lo for f in self.families), default=math.inf)

    @property
    def attack(self) -> float:
        """The best known attack's overhead over every family."""
        return min((f.attack for f in self.families), default=math.inf)


def _cuts(partition: Partition) -> dict:
    return {(b.name, c.name): c for b in partition.serving for c in (*b.cuts, *((b.layout,) if b.layout else ()))}


def _cut_cites(partition: Partition, s: Overhead) -> tuple[str, ...]:
    """The citations of the cut that set each serving block's share (none where its capacity did)."""
    cuts = _cuts(partition)
    return tuple(dict.fromkeys(c for name, share in s.serving.items() if (name, share.cut) in cuts
                               for c in cuts[name, share.cut].cites))


def _binding_ks(partition: Partition, s: Overhead) -> tuple[float, ...]:
    """The novel bits of each block in the cuts that set the serving blocks' shares, least first: where premises must
    hold."""
    cuts = _cuts(partition)
    return tuple(sorted({b.budget.inputs for name, share in s.serving.items() if (name, share.cut) in cuts
                         for b in cuts[name, share.cut].blocks}))


def _binding_k(partition: Partition, s: Overhead) -> float:
    """The most novel bits of any block in the cuts that set the serving blocks' shares."""
    return max(_binding_ks(partition, s), default=0.0)


def _uncertified(s: Overhead) -> bool:
    """Whether the bound counts any gate at `q_U`: the dark leaves, or a cut's uncertified routes."""
    return s.dark > 0 or any(t.split(".")[0] in ROUTES and v > 0 for t, v in s.terms.items())


def _rated(name: str, side: str, cites: tuple[str, ...], s: Overhead, partition: Partition, declared: float,
           table: Mapping[str, Mapping], st: Structure) -> Estimate:
    cites = tuple(dict.fromkeys(cites + _cut_cites(partition, s)
                                + (st.cites() if side == "bound" and _uncertified(s) else ())))
    premise = premise_status(cites, _binding_k(partition, s), table=table) if side == "bound" else ""
    return Estimate(name, side, lean.weakest(cites), s, cites=cites, declared=declared, premise=premise)


def _estimate(arg: Argument, p: Profile, ctx: Context, partition: Partition, block: Budget,
              table: Mapping[str, Mapping], st: Structure, mfu: float, ref: bool) -> tuple[Estimate, Cap | None]:
    try:
        cap = per_block(arg.allowance(p), ctx.units)
    except Outside as e:
        return Estimate(arg.name, arg.side, "outside", None, str(e)), None
    w = weight_per_gate(p, ctx.units)
    s = overhead(partition, cap, w, p.efficiency, arg.side, u=st.weight(w), mfu=mfu, reference_efficiency=ref)
    return _rated(arg.name, arg.side, arg.cites, s, partition, _declared(cap, w, block), table, st), cap


def _declared(cap: Cap, w: float, block: Budget) -> float:
    n = cap(block)
    return max(1.0, w * block.gates / n) if n > 0 else math.inf


def _refuted(cites: tuple[str, ...], k: float, table: Mapping[str, Mapping]) -> bool:
    return premise_status(cites, k, table=table) == STANDINGS[-1]


def _admitted(cap: Cap, cites: tuple[str, ...], table: Mapping[str, Mapping]) -> Cap:
    """`cap` at a block where the premises among `cites` aren't refuted at its bits, and no cap where they are."""
    return lambda b: math.inf if _refuted(cites, b.inputs, table) else cap(b)


def _combine(args: tuple[Argument, ...], bounds: list[tuple[Estimate, Cap | None]], p: Profile, ctx: Context,
             partition: Partition, block: Budget, table: Mapping[str, Mapping], st: Structure, mfu: float,
             ref: bool) -> Estimate:
    """Every bound met at once, the lower bound: each block gets the least of the caps whose premises aren't refuted at
    its bits (a bound refuted there doesn't bound it, and is still reported alone, tagged refuted), under the union of
    the citations of the bounds admitted at a block that binds. `excluded` names the bounds refuted at one."""
    met = [(a, cap) for a, (_, cap) in zip(args, bounds, strict=True) if cap is not None]
    cap = least(*(_admitted(c, a.cites, table) for a, c in met))
    w = weight_per_gate(p, ctx.units)
    s = overhead(partition, cap, w, p.efficiency, u=st.weight(w), mfu=mfu, reference_efficiency=ref)
    ks = _binding_ks(partition, s) or (0.0,)
    refuted = {a.name: [_refuted(a.cites, k, table) for k in ks] for a, _ in met}
    used = [a for a, _ in met if not all(refuted[a.name])]
    e = _rated("+".join(a.name for a in used) or "none", "bound", tuple(c for a in used for c in a.cites), s,
               partition, _declared(cap, w, block), table, st)
    return replace(e, excluded=tuple(n for n, r in refuted.items() if any(r)))


def evaluate(s: Scenario) -> Result:
    ctx, st = s.context, s.structure
    partition = s.regime.partition(ctx)
    block = Budget(ctx.c, ctx.policy.k)
    out = []
    for fam in s.families:
        p = fam.profile(s.model, s.run)
        ref = s.reference_efficiency
        bounds = [_estimate(b, p, ctx, partition, block, s.premises, st, s.mfu, ref) for b in s.bounds]
        combined = _combine(s.bounds, bounds, p, ctx, partition, block, s.premises, st, s.mfu, ref)
        out.append(FamilyResult(fam.name, p, tuple(e for e, _ in bounds), combined,
                                tuple(_estimate(a, p, ctx, partition, block, s.premises, st, s.mfu, ref)[0]
                                      for a in s.attacks)))
    return Result(s, tuple(out))


def reads(registry: Registry, name: str) -> frozenset[str]:
    """The registry keys scenario `name` reads: the parameters its overhead can depend on."""
    tracked = registry.tracked()
    build(tracked, name)
    return frozenset(tracked.seen or ())


def evaluate_named(registry: Registry, name: str) -> Result:
    return evaluate(build(registry, name))
