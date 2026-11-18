"""A resource scheme as the resource accounting layer takes it, and the one registry a configuration names it from.

The proof layer gives every scheme the same four things; a scheme reads them and owns none of them:

- the registered circuit, with its blocks and wires (`verity.commitments.periods.interface.Circuit`);
- the commitments to its trace, logged as they arrive (`verity.commitments.periods.interface.Window`'s records);
- the verifier's coins, each logged after the commitments it must follow (the challenger's ledger entries,
  `Core.Record`, whose Python is `verity.channel.Record`);
- the audit: one sampling design over all blocks, and what its sample established
  (`verity.protocols.proofs.spot.verifier.profile.IntegrityProfile`, `Core.Integrity` in Lean).

A scheme supplies a `Scheme`: its blocks (`Block`), the cost of each in its resource, its assumptions, its conditions on
the log (`Order`), its other charges (`Charge`) and its settings. It gets back a `Bound`. Where the proof layer does not
yet provide one of its four things uniformly, a scheme states what it requires (`REQUIREMENTS`) and the stand-in it runs
on today (`Scheme.requires`).

The definition is the Lean structure `Verity.Scheme` and its `Verity.Scheme.GetsBack` (`security/Definitions/Core/
Scheme.lean`); `Scheme`, `Block`, `Order`, `Charge` and `Bound` mirror the fields a bound reads under its names (``δ``
spelt ``delta``), and a change to either names the other's line in its message. `hooks`, `settings`, `option`,
`ports` and `requires` are read by the adapter and the runner, by no guarantee, and have no Lean field.

**Declaring a scheme.** One module, ``<protocol>/scheme.py``, calls `register(Scheme(...))`, and each name it registers
is a line of `verity.protocols.compose.schemes.DECLARATIONS`, from which every reader gets it (`get`, `load`). Its fields, with the Lean they mirror
(`Verity.Scheme`):

- ``blocks``: `Block(definition, cost)`, `Block` and `cost`; ``cost`` reads public values logged before the sample's coin.
- ``charges``: `Charge(name, bound)`, summed in `charges`; a known channel with no bound yet is ``bounded=False``.
- ``log``: `Order`s, `logOK`; a deadline (``within_ns``) names one of ``assumptions``, the names `breaks` ranges over.
- ``over``, ``under``: `over` and `under`, each side said in words with its natural unit; a scheme whose term is said
  says both. A scheme bounding what it credits by what was spent or held puts the credit ``over``; a factor in its
  guarantee is cleared into the two sides, never carried beside them.
- ``delta_scheme``: `δscheme`, said by name: a number, zero included, or a `Term`, the Lean constant of a family of
  schemes (one for each value of its parameters) and its code, which `bound` evaluates at parameters its caller gives;
  unsaid, `bound` refuses.
- code only: ``name`` (the configuration's), ``protocol`` (`Core.Protocol.id`), ``resource`` (`Core.Account.Part`),
  ``settings`` (typed defaults, the application's ``[params]`` over them through `settings`), ``hooks``, ``ports``,
  ``option``, ``requires``, and what it does on a served run: ``requests(ctx)``, the requests it makes of the run
  (`verity.hooks`), ``attach(exchange, ctx)``, which subscribes to the run's exchange and installs its kernels, returning what the adapter
  reads of it (a `verity.hooks.Front`, a `verity.hooks.Gate`), and
  ``processes``, what it starts beside the workload by name (`verity.hooks.StartBeside`, entered in `PROCESSES`).
  ``ctx`` is a mapping: ``settings``, the scheme's own (`settings`, the application's over its defaults); ``build``,
  the key of the build it attaches to (one per exchange); and what the launcher passes, which the network attester
  reads as ``admit``, ``source``, ``submit``, ``abort``, ``take``, ``pull``, ``log``, ``clock`` and ``schedule``.
  The adapter's exchange attaches it (``Exchange.attach``), files those of ``requests(ctx)`` that ``attach`` did not,
  and declares it with its settings under its name (`verity.hooks.Declared`).

What it gets back: `bound` gives `Bound`, `GetsBack`'s numbers; `credited` gives `Placed.credited`, the first-level blocks
the integrity profile's ``credited`` takes.

A served run offers every scheme the same five hooks (`verity.hooks.HOOKS`), and a scheme names those it attaches
through (`Scheme.hooks`).

A scheme's `name` is the name a configuration lists (`protocols = [...]` in `applications/NAME.toml`, which
`verity.config.Component.name` declares), so changing schemes is changing that name. Its `protocol` is the id of the
Lean check that decides it (`Core.Protocol.id`), and `resource` is the part of the registered hardware its bound is
about, as `Core.Account.Part` names it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal

from verity.hooks import HOOKS


def _no_requests(ctx: Any) -> tuple[Any, ...]:
    return ()


def _attaches_nothing(exchange: Any, ctx: Any) -> None:
    return None


#: `Core.Account.Part`'s names, each with the unit its rows are in (`Account.Part.unit`)
RESOURCES: Mapping[str, str] = MappingProxyType({"compute": "MACs", "memory": "bits", "read": "bits",
                                                 "egress": "bits"})
#: what a scheme needs of the proof layer that the layer doesn't provide every scheme the same way yet
REQUIREMENTS = ("one-labelling", "log-order", "log-times", "coins-at-rate", "one-design")
_NAME = re.compile(r"[a-z][a-z0-9-]*")


@dataclass(frozen=True)
class Block:
    """A kind of block the scheme adds to the registered circuit: its Definition, an exact function of its inputs, by the
    id `verity.commitments.periods.interface.resolve` reads, and ``cost``, the resource credited for one correct block.

    ``cost`` takes the block's public values (its Definition's public shape and the record's values for it, such as a
    kept width), every one of which is logged before the sample's coin: a value the prover could set after seeing the
    sample would let it raise the cost of the sampled blocks. `Verity.Scheme`'s block is one sampled unit; a `Block`
    here is a kind whose every unit of one kind and public values costs the same, which is how the two agree. No
    declared cost may be below what the verdict credits for the block: a larger cost only weakens the bound."""

    definition: str
    cost: Callable[[Mapping[str, int]], int]


@dataclass(frozen=True)
class Order:
    """A condition on the verifier's log (`Verity.Scheme.logOK`): ``committed`` is logged ``direction`` (``before`` or
    ``after``) ``coin``, a coin drawn or another logged event, and, where ``within_ns`` is set, within that many
    nanoseconds of it: a deadline, which no guarantee of the proof layer reads, so it names the scheme's ``assumption``
    that does (a bound on the prover's speed, say)."""

    committed: str
    coin: str
    direction: Literal["before", "after"] = "before"
    within_ns: int | None = None
    assumption: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("before", "after"):
            raise ValueError(f"{self.committed}: an order is before or after {self.coin}, not {self.direction!r}")
        if (self.within_ns is None) != (self.assumption is None):
            raise ValueError(f"{self.committed} {self.direction} {self.coin}: a time is read only by an assumption, "
                             "and names it")
        if self.within_ns is not None and self.within_ns <= 0:
            raise ValueError(f"{self.committed} {self.direction} {self.coin}: a deadline is a positive time")


@dataclass(frozen=True)
class Charge:
    """A known way to a correct block without spending the resource: its name, and where its bound is stated (the key
    of the scheme's published answer that carries it, or the Lean declaration that bounds it). A known channel is
    declared even before anything bounds it, with ``bounded`` False and ``bound`` naming the statement to come: then
    `bound` refuses the scheme, since a bound without it would read as complete (`Verity.Scheme.charges` is a number)."""

    name: str
    bound: str
    bounded: bool = True


@dataclass(frozen=True)
class Port:
    """An external port the scheme reads (``input``) or writes (``output``), laid out as the field ``field`` of the
    public signature of its block's Definition ``definition``: the scan's state row, the released bytes' positions. The
    adapter's lowering reads the layout from here; no guarantee reads it."""

    name: str
    direction: Literal["input", "output"]
    definition: str
    field: str


@dataclass(frozen=True)
class Term:
    """A scheme's own term that depends on public parameters: in the Lean a family of schemes, one for each value, whose
    `δscheme` is the constant ``lean`` at them. ``parameters`` are its parameters' names (``q``, ``N``), and ``value``
    evaluates it at them by name; no parameter's value is written into a declaration (`bound` takes them)."""

    lean: str
    parameters: tuple[str, ...]
    value: Callable[..., Fraction]

    def __post_init__(self) -> None:
        if not self.lean or not self.parameters or len(set(self.parameters)) != len(self.parameters):
            raise ValueError(f"a term names its Lean constant and its parameters once each, not {self.lean!r} "
                             f"{self.parameters}")


@dataclass(frozen=True)
class Scheme:
    """What a resource scheme supplies. ``settings`` are its settings with their defaults, which a configuration's
    ``[params."<protocol>"]`` overrides (`settings`); ``option`` is the adapter's entry that attaches it to a served run
    (``NAME:scheme``, an adapter's ``protocols`` spelling), empty when it attaches through none; ``hooks`` the served
    run's hooks it attaches through (`HOOKS`); ``ports`` the external ports it reads or writes; ``requires`` maps each
    requirement it has of the proof layer to today's stand-in. ``delta_scheme`` is `Verity.Scheme.δscheme`, the bound on
    the probability that one of its ``assumptions`` fails (a key guessed, a collision), said by name in every
    declaration: a number, zero where none can fail, or a `Term` where it depends on public parameters; None, not said,
    lets a scheme be registered and read but not bounded (`bound`). ``over`` and ``under`` are `Verity.Scheme.over` and
    `under` in words, each with its natural unit: the bound is that ``over`` exceeds ``under`` by at most the incorrect
    blocks' cost and the charges; a scheme whose term is said says both."""

    name: str
    protocol: str
    resource: str
    blocks: tuple[Block, ...]
    assumptions: tuple[str, ...]
    log: tuple[Order, ...]
    charges: tuple[Charge, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    option: str = ""
    hooks: tuple[str, ...] = ()
    requires: Mapping[str, str] = field(default_factory=dict)
    ports: tuple[Port, ...] = ()
    delta_scheme: Fraction | Term | None = None
    over: str = ""
    under: str = ""
    requests: Callable[[Any], Sequence[Any]] = _no_requests
    attach: Callable[[Any, Any], Any] = _attaches_nothing
    processes: Mapping[str, Callable[..., Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"a scheme's name is a configuration's name ({_NAME.pattern}), not {self.name!r}")
        if unknown := sorted(set(self.hooks) - set(HOOKS)):
            raise ValueError(f"{self.name}: {unknown} are not hooks of a served run {HOOKS}")
        if self.resource not in RESOURCES:
            raise ValueError(f"{self.name}: its resource is one of {sorted(RESOURCES)}, not {self.resource!r}")
        if unknown := sorted(set(self.requires) - set(REQUIREMENTS)):
            raise ValueError(f"{self.name}: {unknown} are not requirements of the proof layer {REQUIREMENTS}")
        names = [c.name for c in self.charges]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name} names a charge twice: {names}")
        if unknown := sorted({o.assumption for o in self.log if o.assumption} - set(self.assumptions)):
            raise ValueError(f"{self.name}: its deadlines name {unknown}, which are not among its assumptions")
        if unknown := sorted({p.definition for p in self.ports} - {b.definition for b in self.blocks}):
            raise ValueError(f"{self.name}: its ports lie in {unknown}, which are not its blocks' Definitions")
        if self.delta_scheme is not None and not isinstance(self.delta_scheme, Term) and not 0 <= self.delta_scheme < 1:
            raise ValueError(f"{self.name}: its own term δscheme lies in [0, 1), not {self.delta_scheme}")
        if self.delta_scheme is not None and not (self.over and self.under):
            raise ValueError(f"{self.name}: a scheme whose term is said says its two sides (Verity.Scheme.over and under) "
                             "in words, each with its unit")
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
        object.__setattr__(self, "requires", MappingProxyType(dict(self.requires)))

    @property
    def unit(self) -> str:
        return RESOURCES[self.resource]


def settings(scheme: Scheme, given: Mapping[str, Any]) -> dict[str, Any]:
    """``scheme``'s settings with ``given`` over its defaults; refuses a key it doesn't have and a value of another type
    than its default's."""
    if unknown := sorted(set(given) - set(scheme.settings)):
        raise ValueError(f"{scheme.name} has no setting {unknown}: its settings are {sorted(scheme.settings)}")
    for k, v in given.items():
        if type(v) is not type(scheme.settings[k]):
            raise ValueError(f"{scheme.name}.{k} is a {type(scheme.settings[k]).__name__}, not {v!r}")
    return {**scheme.settings, **given}


@dataclass(frozen=True)
class Bound:
    """What the audit's guarantee gives a scheme (`Verity.Scheme.GetsBack`): at cutoff ``delta``, its ``over`` exceeds
    its ``under`` by more than ``missable``, the largest cost of a set of its blocks that the sample misses with
    probability above ``delta``, plus each of its charges' bounds, with probability at most ``delta`` plus
    ``delta_scheme``, the scheme's own term: where that is a `Term`, ``term`` names its Lean constant and
    ``parameters`` the values it was evaluated at."""

    scheme: str
    delta: Fraction
    missable: int
    charges: Mapping[str, int]
    delta_scheme: Fraction
    term: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def excess(self) -> int:
        """The most ``over`` exceeds ``under`` by, but with `probability`."""
        return self.missable + sum(self.charges.values())

    @property
    def probability(self) -> Fraction:
        """The probability that ``over`` exceeds ``under`` by more than `excess`: ``delta + delta_scheme``."""
        return self.delta + self.delta_scheme


def bound(scheme: Scheme, delta: Fraction, missable: int, charges: Mapping[str, int],
          parameters: Mapping[str, Any] | None = None) -> Bound:
    """``scheme``'s `Bound`, carrying its own term, evaluated at ``parameters`` where it is a `Term` (``N`` read from
    the registered circuit, ``q`` and the rest the caller's); refuses charges other than exactly the ones it states, a
    scheme with a known charge nothing bounds yet or whose own term is not said, and parameters other than exactly the
    term's."""
    if sorted(charges) != sorted(c.name for c in scheme.charges):
        raise ValueError(f"{scheme.name}'s charges are {sorted(c.name for c in scheme.charges)}, not {sorted(charges)}")
    if unbounded := [f"{c.name} ({c.bound})" for c in scheme.charges if not c.bounded]:
        raise ValueError(f"{scheme.name} has known charges nothing bounds yet: {', '.join(unbounded)}; a bound "
                         "without them would read as complete")
    if scheme.delta_scheme is None:
        raise ValueError(f"{scheme.name}: its delta_scheme (Verity.Scheme.δscheme) is not said: a declaration says its "
                         "own term by name, zero where none of its assumptions can fail")
    if not 0 < delta < 1 or missable < 0 or any(v < 0 for v in charges.values()):
        raise ValueError(f"{scheme.name}: δ lies in (0, 1) and every amount is at least 0")
    given = dict(parameters or {})
    term = scheme.delta_scheme
    if not isinstance(term, Term):
        if given:
            raise ValueError(f"{scheme.name}: its own term is the number {term}, which takes no parameters {sorted(given)}")
        return Bound(scheme.name, Fraction(delta), missable, MappingProxyType(dict(charges)), Fraction(term))
    if missing := [p for p in term.parameters if p not in given]:
        raise ValueError(f"{scheme.name}: its own term {term.lean} is evaluated at {list(term.parameters)}, and "
                         f"{missing} were not given")
    if extra := sorted(set(given) - set(term.parameters)):
        raise ValueError(f"{scheme.name}: its own term {term.lean} has no parameters {extra}")
    value = Fraction(term.value(**given))
    if not 0 <= value < 1:
        raise ValueError(f"{scheme.name}: its own term {term.lean} at {given} is {value}, outside [0, 1): no bound")
    return Bound(scheme.name, Fraction(delta), missable, MappingProxyType(dict(charges)), value, term.lean,
                 MappingProxyType(given))


#: the registered schemes, by name: a scheme's module registers it when imported, and a reader gets it through
#: `verity.protocols.compose.schemes.get`
SCHEMES: dict[str, Scheme] = {}


def register(scheme: Scheme) -> Scheme:
    if scheme.name in SCHEMES:
        raise KeyError(f"a scheme named {scheme.name!r} is registered already")
    SCHEMES[scheme.name] = scheme
    for name, start in scheme.processes.items():
        register_process(name, start)
    return scheme


#: the processes schemes start beside a workload, by name (`verity.hooks.StartBeside`): each called with the readers
#: its request names
PROCESSES: dict[str, Callable[..., Any]] = {}


def register_process(name: str, start: Callable[..., Any]) -> Callable[..., Any]:
    if name in PROCESSES:
        raise KeyError(f"a process named {name!r} is registered already")
    PROCESSES[name] = start
    return start


@dataclass(frozen=True)
class Circuits:
    """What a scheme's executed computation is traced as: ``schemes``, the schemes with a circuit, and ``traced_as(scheme,
    m, k, n)``, one executed call's circuit (None without one)."""

    schemes: Mapping[str, Any]
    traced_as: Callable[[str, int, int, int], Any]


#: the circuits of each scheme's executed computation, by the name a configuration lists it under
CIRCUITS: dict[str, Circuits] = {}


def register_circuits(name: str, circuits: Circuits) -> Circuits:
    if name in CIRCUITS:
        raise KeyError(f"circuits named {name!r} are registered already")
    CIRCUITS[name] = circuits
    return circuits


def credited(schemes: Sequence[Scheme], strata: Iterable[tuple[str, Sequence[Sequence[int]]]]) -> frozenset[int]:
    """`Verity.Scheme.Placed.credited`: the numbers of the first-level proof blocks that ``schemes`` credit, in the
    numbering the integrity profile's ``credited`` takes. ``strata`` are the registered circuit's first-level proof blocks
    by template, ``(descriptor id, ascending [lo, hi) ranges)``, as each party derives them from the public Program and
    query (`verity.protocols.proofs.spot.verifier.one_stage.partition.strata`). A declared `Block` places on the
    templates whose descriptor id is its Definition id, bare or followed by its statics (``Name_vN{...}``). Refuses a
    declared Definition id that resolves to no block, and a template two blocks would place on, of one scheme or two
    (`Placed.inj`, and one credited set for all)."""
    strata = [(tid, rs) for tid, rs in strata]
    placed: dict[str, str] = {}
    out: set[int] = set()
    for s in schemes:
        for b in s.blocks:
            hits = [(tid, rs) for tid, rs in strata if tid == b.definition or tid.startswith(b.definition + "{")]
            if not hits:
                raise ValueError(f"{s.name}: its block {b.definition} resolves to no block of the circuit, whose "
                                 f"templates are {[tid for tid, _ in strata]}")
            for tid, rs in hits:
                if tid in placed:
                    raise ValueError(f"{tid} would be credited twice: by {placed[tid]} and by {s.name}'s {b.definition}")
                placed[tid] = f"{s.name}'s {b.definition}"
                out.update(u for lo, hi in rs for u in range(lo, hi))
    return frozenset(out)


__all__ = ["CIRCUITS", "PROCESSES", "REQUIREMENTS", "RESOURCES", "SCHEMES", "Block", "Bound", "Charge", "Circuits", "Order",
           "Port", "Scheme", "Term", "bound", "credited", "register", "register_circuits", "register_process", "settings"]
