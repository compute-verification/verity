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

A served run offers every scheme the same five hooks (`verity.hooks.HOOKS`), and a scheme names those it attaches
through (`Scheme.hooks`).

A scheme's `name` is the name a configuration lists (`protocols = [...]` in `applications/NAME.toml`, which
`verity.config.Component.name` declares), so changing schemes is changing that name. Its `protocol` is the id of the
Lean check that decides it (`Core.Protocol.id`), and `resource` is the part of the registered hardware its bound is
about, as `Core.Account.Part` names it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

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
    id `verity.commitments.periods.interface.resolve` reads, and the cost of one correct block in the scheme's resource
    from the block's public shape."""

    definition: str
    cost: Callable[[Mapping[str, int]], int]


@dataclass(frozen=True)
class Order:
    """A condition on the verifier's log: ``committed`` is logged before the coin ``coin`` is drawn and, where
    ``within_ns`` is set, within that many nanoseconds of it, a time only the assumption ``assumption`` reads."""

    committed: str
    coin: str
    within_ns: int | None = None
    assumption: str | None = None

    def __post_init__(self) -> None:
        if (self.within_ns is None) != (self.assumption is None):
            raise ValueError(f"{self.committed} before {self.coin}: a time is read only by an assumption, and "
                             "names it")


@dataclass(frozen=True)
class Charge:
    """A known way to a correct block without spending the resource: its name, and where its bound is stated (the key
    of the scheme's published answer that carries it, or the Lean declaration that bounds it)."""

    name: str
    bound: str


@dataclass(frozen=True)
class Scheme:
    """What a resource scheme supplies. ``settings`` are its settings with their defaults, which a configuration's
    ``[params."<protocol>"]`` overrides (`settings`); ``option`` is the adapter's entry that attaches it to a served run
    (``NAME:scheme``, an adapter's ``protocols`` spelling), empty when it attaches through none; ``hooks`` the served
    run's hooks it attaches through (`HOOKS`); ``requires`` maps each requirement it has of the proof layer to today's
    stand-in."""

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
    """What the audit's guarantee gives a scheme at cutoff ``delta``: the resource it leaves uncredited is at most
    ``missable``, the largest cost of a set of its blocks that the sample misses with probability above ``delta``, plus
    each of its charges' bounds."""

    scheme: str
    delta: Fraction
    missable: int
    charges: Mapping[str, int]

    @property
    def uncredited(self) -> int:
        return self.missable + sum(self.charges.values())


def bound(scheme: Scheme, delta: Fraction, missable: int, charges: Mapping[str, int]) -> Bound:
    """``scheme``'s `Bound`; refuses charges other than exactly the ones it states."""
    if sorted(charges) != sorted(c.name for c in scheme.charges):
        raise ValueError(f"{scheme.name}'s charges are {sorted(c.name for c in scheme.charges)}, not {sorted(charges)}")
    if not 0 < delta < 1 or missable < 0 or any(v < 0 for v in charges.values()):
        raise ValueError(f"{scheme.name}: δ lies in (0, 1) and every amount is at least 0")
    return Bound(scheme.name, Fraction(delta), missable, MappingProxyType(dict(charges)))


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


__all__ = ["PROCESSES", "REQUIREMENTS", "RESOURCES", "SCHEMES", "Block", "Bound", "Charge", "Order", "Scheme", "bound",
           "get", "register", "register_process", "settings"]
