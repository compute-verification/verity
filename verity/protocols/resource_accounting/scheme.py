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

A served run offers every scheme the same five hooks (`HOOKS`), and a scheme names those it attaches through
(`Scheme.hooks`): values at block boundaries (`Value`), a wrap of a linear's call (`LinearCall`, `LinearWrap`), KV block
events (`Loaded`, `KvPool`, `KvBlocks`, `Step`), admission and release events (`Admission`, `Gate`), and the engine's
build options (`Build`, `Builds`). The adapter hands them over and names no scheme.

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
from typing import Any, Literal, Protocol

#: `Core.Account.Part`'s names, each with the unit its rows are in (`Account.Part.unit`)
RESOURCES: Mapping[str, str] = MappingProxyType({"compute": "MACs", "memory": "bits", "read": "bits",
                                                 "egress": "bits"})
#: what a scheme needs of the proof layer that the layer doesn't provide every scheme the same way yet
REQUIREMENTS = ("one-labelling", "log-order", "log-times", "coins-at-rate", "one-design")
#: the hooks a served run offers every scheme alike: values at block boundaries, a wrap of a linear's call, KV block
#: events, admission and release events, and the engine's build options
HOOKS = ("values", "linear", "kv", "admission", "build")
_NAME = re.compile(r"[a-z][a-z0-9-]*")

type Handle = object
"""A device buffer as the serving system holds it: opaque here, read only by the scheme's own prover code."""


# --- the hooks ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Value:
    """``values``: one value as the workload produces it at a block boundary, at the site ``(module, kind, slot)``."""

    module: str
    kind: str
    slot: int | str
    value: Handle


@dataclass(frozen=True)
class LinearCall:
    """``linear``: one call of a linear layer on one rank: the layer (``weight_id`` stable across ranks, ``n`` and ``k``
    this rank's), its input, weight and bias."""

    weight_id: int
    qualname: str
    n: int
    k: int
    weight_format: str
    x: Handle
    weight: Handle
    bias: Handle | None = None


class LinearWrap(Protocol):
    """A scheme's wrap of a linear's call: it returns the call's output, computing it itself or through ``original``,
    the serving system's own call, given a weight in place of the layer's or None for the layer's."""

    def __call__(self, call: LinearCall, original: Callable[[Handle | None], Handle]) -> Handle: ...


@dataclass(frozen=True)
class Loaded:
    """``kv``: the model's weights once loaded, by name."""

    weights: tuple[tuple[str, Handle], ...]


@dataclass(frozen=True)
class KvPool:
    """``kv``: the KV pool once allocated: ``pages`` pages of ``slots`` slots, each slot ``slot_bytes`` bytes in each of
    ``planes`` planes, one storage a layer."""

    pages: int
    slots: int
    slot_bytes: int
    planes: int
    layers: tuple[Handle, ...]


@dataclass(frozen=True)
class KvBlocks:
    """``kv``: pages the pool handed out (``allocated``), returned with no reader left (``freed``), or handed out again
    to a new reader (``touched``)."""

    event: Literal["allocated", "freed", "touched"]
    blocks: tuple[int, ...]


@dataclass(frozen=True)
class KvWrite:
    """``count`` slots from ``offset`` of page ``block``, written in one step."""

    block: int
    offset: int
    count: int


@dataclass(frozen=True)
class Step:
    """``kv``: scheduler step ``index`` begins (``writes``, the slots it writes) or ends (``ended``)."""

    index: int
    writes: tuple[KvWrite, ...] = ()
    ended: bool = False


@dataclass(frozen=True)
class Admission:
    """``admission``: requests by their release key, ``added`` to the engine, ``scheduled`` at a step, or ``finished``."""

    event: Literal["added", "scheduled", "finished"]
    keys: tuple[str, ...]
    step: int | None = None


class Gate(Protocol):
    """A scheme's hold on admission: ``add`` admits a request when the scheme calls ``admit``, and ``pending`` says
    whether it holds any not yet admitted."""

    def add(self, key: str, admit: Callable[[], None]) -> None: ...

    def pending(self) -> bool: ...


@dataclass(frozen=True)
class Build:
    """``build``: what a scheme's refusals read before the engine is built: the stage (``serve``, ``capture`` or
    ``commit``), the schemes enabled with it as ``name:scheme``, the ranks, whether the engine runs eagerly, and each
    scheme's own settings."""

    stage: Literal["serve", "capture", "commit"]
    enabled: tuple[str, ...]
    world: int = 1
    enforce_eager: bool = True
    settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class Builds(Protocol):
    """A scheme's say on the engine's build: why this run cannot enable it (empty: it can), and whether its computation
    fixes the build's kernels so serving may not override them."""

    def refusals(self, build: Build) -> tuple[str, ...]: ...

    def requires_build(self, build: Build) -> bool: ...


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


def get(name: str) -> Scheme:
    """The scheme registered under ``name``; `KeyError` names the registered ones."""
    try:
        return SCHEMES[name]
    except KeyError:
        raise KeyError(f"no scheme {name!r}; registered: {', '.join(sorted(SCHEMES)) or 'none'}") from None


__all__ = ["HOOKS", "REQUIREMENTS", "RESOURCES", "SCHEMES", "Admission", "Block", "Bound", "Build", "Builds", "Charge",
           "Gate", "Handle", "KvBlocks", "KvPool", "KvWrite", "LinearCall", "LinearWrap", "Loaded", "Order", "Scheme",
           "Step", "Value", "bound", "get", "register", "settings"]
