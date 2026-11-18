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

**Declaring a scheme.** One module, ``<protocol>/scheme.py``, calls `register(Scheme(...))` and is listed in
`verity.protocols.compose.schemes.DECLARATIONS`, which every reader loads (`load`). Its fields, with the Lean they mirror
(`Verity.Scheme`):

- ``blocks``: `Block(definition, cost)`, `Block` and `cost`; ``cost`` reads public values logged before the sample's coin.
- ``charges``: `Charge(name, bound)`, summed in `charges`; a known channel with no bound yet is ``bounded=False``.
- ``log``: `Order`s, `logOK`; a deadline (``within_ns``) names one of ``assumptions``, the names `breaks` ranges over.
- ``delta_scheme``: `δscheme`, said by name, zero included; unsaid, `bound` refuses.
- code only: ``name`` (the configuration's), ``protocol`` (`Core.Protocol.id`), ``resource`` (`Core.Account.Part`),
  ``settings`` (typed defaults, the application's ``[params]`` over them through `settings`), ``hooks``, ``ports``,
  ``option``, ``requires``.

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
class Scheme:
    """What a resource scheme supplies. ``settings`` are its settings with their defaults, which a configuration's
    ``[params."<protocol>"]`` overrides (`settings`); ``option`` is the adapter's entry that attaches it to a served run
    (``NAME:scheme``, an adapter's ``protocols`` spelling), empty when it attaches through none; ``hooks`` the served
    run's hooks it attaches through (`HOOKS`); ``ports`` the external ports it reads or writes; ``requires`` maps each
    requirement it has of the proof layer to today's stand-in. ``delta_scheme`` is `Verity.Scheme.δscheme`, the bound on
    the probability that one of its ``assumptions`` fails (a key guessed, a collision), said by name in every
    declaration, zero where none can; None, not said, lets a scheme be registered and read but not bounded (`bound`)."""

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
    delta_scheme: Fraction | None = None

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
        if self.delta_scheme is not None and not 0 <= self.delta_scheme < 1:
            raise ValueError(f"{self.name}: its own term δscheme lies in [0, 1), not {self.delta_scheme}")
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
    """What the audit's guarantee gives a scheme (`Verity.Scheme.GetsBack`): at cutoff ``delta``, the resource it
    leaves uncredited is more than ``missable``, the largest cost of a set of its blocks that the sample misses with
    probability above ``delta``, plus each of its charges' bounds, with probability at most ``delta`` plus
    ``delta_scheme``, the scheme's own term."""

    scheme: str
    delta: Fraction
    missable: int
    charges: Mapping[str, int]
    delta_scheme: Fraction

    @property
    def uncredited(self) -> int:
        return self.missable + sum(self.charges.values())

    @property
    def probability(self) -> Fraction:
        """The probability that more than `uncredited` is left uncredited: ``delta + delta_scheme``."""
        return self.delta + self.delta_scheme


def bound(scheme: Scheme, delta: Fraction, missable: int, charges: Mapping[str, int]) -> Bound:
    """``scheme``'s `Bound`, carrying its own term; refuses charges other than exactly the ones it states, and a scheme
    with a known charge nothing bounds yet or whose own term is not said."""
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
    return Bound(scheme.name, Fraction(delta), missable, MappingProxyType(dict(charges)), scheme.delta_scheme)


#: the registered schemes, by name: a scheme's package registers it when imported
SCHEMES: dict[str, Scheme] = {}


def register(scheme: Scheme) -> Scheme:
    if scheme.name in SCHEMES:
        raise KeyError(f"a scheme named {scheme.name!r} is registered already")
    SCHEMES[scheme.name] = scheme
    return scheme


#: the processes schemes start beside a workload, by name (`verity.hooks.StartBeside`): each called with the readers
#: its request names
PROCESSES: dict[str, Callable[..., Any]] = {}


def register_process(name: str, start: Callable[..., Any]) -> Callable[..., Any]:
    if name in PROCESSES:
        raise KeyError(f"a process named {name!r} is registered already")
    PROCESSES[name] = start
    return start


def get(name: str) -> Scheme:
    """The scheme registered under ``name``; `KeyError` names the registered ones."""
    try:
        return SCHEMES[name]
    except KeyError:
        raise KeyError(f"no scheme {name!r}; registered: {', '.join(sorted(SCHEMES)) or 'none'}") from None


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


__all__ = ["PROCESSES", "REQUIREMENTS", "RESOURCES", "SCHEMES", "Block", "Bound", "Charge", "Order", "Port", "Scheme",
           "bound", "credited", "get", "register", "register_process", "settings"]
