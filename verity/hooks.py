"""The messages between a served workload and the schemes, in one process: the events a served run emits, each stamped
with the workload's step and holding handles, never bytes, and the requests a scheme files before the workload runs.

The words are the circuit's: an adapter translates its serving system into them. Each event names its hook (`HOOKS`),
the channel a scheme subscribes to it through (`verity.protocols.resource_accounting.scheme.Scheme.hooks`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

#: the hooks a served run offers every scheme alike, each the ``hook`` of its events (`EVENTS`)
HOOKS = ("declared", "steps", "values", "blocks", "memory")

type Handle = object
"""A device buffer as the serving system holds it: opaque here, read only by the scheme's own prover code."""


@dataclass(frozen=True)
class Declared:
    """``declared``: what the workload declares before it runs: the card's file id, its properties (``stage``: serve,
    capture or commit; ``world``; ``eager``; ``async_steps``, whether a step is scheduled before the last one's outputs
    are read; ``in_process``, whether the steps are taken in this process; a precision, a cache, a sampler), the schemes enabled with it as
    ``name:scheme`` and each scheme's own settings."""

    hook: ClassVar[str] = "declared"
    card: str
    properties: Mapping[str, Any]
    schemes: tuple[str, ...] = ()
    settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    step: int = 0


@dataclass(frozen=True)
class RegisteredValues:
    """``memory``: the registered values once resident, by name; ``linear`` names those that are a linear block's
    weight."""

    hook: ClassVar[str] = "memory"
    values: tuple[tuple[str, Handle], ...]
    step: int = 0
    linear: frozenset[str] = frozenset()


@dataclass(frozen=True)
class InputArrived:
    """``steps``: input ``input`` of ``length`` values arrived, at ``step``, with the ``headers`` it came with (names
    lower-cased) and its ``first_line``."""

    hook: ClassVar[str] = "steps"
    input: str
    length: int
    step: int
    headers: Mapping[str, str] = field(default_factory=dict)
    first_line: str = ""


@dataclass(frozen=True)
class StepBegan:
    """``steps``: workload step ``step`` begins, computing for ``inputs`` over ``slots`` slots, with ``queued`` started
    inputs left waiting for a slot."""

    hook: ClassVar[str] = "steps"
    step: int
    inputs: tuple[str, ...] = ()
    slots: int = 0
    queued: int = 0


@dataclass(frozen=True)
class StepEnded:
    hook: ClassVar[str] = "steps"
    step: int


@dataclass(frozen=True)
class RunBegan:
    """``steps``: the instrumented run ``run`` begins: its steps from ``step`` on are the run's."""

    hook: ClassVar[str] = "steps"
    run: str
    step: int


@dataclass(frozen=True)
class RunEnded:
    """``steps``: the instrumented run ``run`` ends after step ``step``."""

    hook: ClassVar[str] = "steps"
    run: str
    step: int


@dataclass(frozen=True)
class ValueProduced:
    """``values``: a value at a block's boundary, at the site ``(block, port, slot)``."""

    hook: ClassVar[str] = "values"
    block: str
    port: str
    slot: int | str
    value: Handle
    step: int = 0


@dataclass(frozen=True)
class BlockDue:
    """``blocks``: block ``block`` (``key`` stable across ranks, ``shape`` this rank's, its operands in ``format``) is
    about to be computed from ``inputs``; ``weights`` names, by input, the registered value an input is."""

    hook: ClassVar[str] = "blocks"
    block: str
    key: int
    shape: Mapping[str, int]
    format: str
    inputs: Mapping[str, Handle]
    step: int = 0
    weights: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RangesDeclared:
    """``memory``: the memory ranges once allocated: ``ranges`` ranges of ``slots`` slots, each slot ``slot_bytes``
    bytes in each of ``planes`` planes, one store a layer."""

    hook: ClassVar[str] = "memory"
    ranges: int
    slots: int
    slot_bytes: int
    planes: int
    stores: tuple[Handle, ...]
    step: int = 0


@dataclass(frozen=True)
class RangeAllocated:
    """``memory``: ranges handed to ``input`` (``reused``: handed again, holding values another input wrote)."""

    hook: ClassVar[str] = "memory"
    ranges: tuple[int, ...]
    input: str | None = None
    reused: bool = False
    step: int = 0


@dataclass(frozen=True)
class RangeWritten:
    """``memory``: ``count`` slots from ``offset`` of range ``range``, written at ``step``."""

    hook: ClassVar[str] = "memory"
    range: int
    offset: int
    count: int
    step: int = 0


@dataclass(frozen=True)
class RangeFreed:
    """``memory``: ranges returned with no reader left."""

    hook: ClassVar[str] = "memory"
    ranges: tuple[int, ...]
    step: int = 0


@dataclass(frozen=True)
class OutputReleased:
    """``steps``: output ``index`` of input ``input`` left the workload at ``step``."""

    hook: ClassVar[str] = "steps"
    input: str
    index: int
    step: int


@dataclass(frozen=True)
class OutputsComplete:
    hook: ClassVar[str] = "steps"
    input: str
    step: int


#: every event a served run emits, each read through its ``hook``
EVENTS = (Declared, RegisteredValues, InputArrived, StepBegan, StepEnded, ValueProduced, BlockDue, RangesDeclared,
          RangeAllocated, RangeWritten, RangeFreed, OutputReleased, OutputsComplete, RunBegan, RunEnded)


@dataclass(frozen=True)
class RequireProperty:
    """A scheme's request that the workload declare ``property`` as ``value``; ``fixes``: the scheme's computation is
    defined by the declared build, so serving may not override it."""

    property: str
    value: Any
    why: str
    fixes: bool = False


@dataclass(frozen=True)
class RefuseProperty:
    """A scheme's refusal of a workload that declares ``property`` as ``value``."""

    property: str
    value: Any
    why: str


def unmet[T](requests: Iterable[tuple[T, Any]], properties: Mapping[str, Any]) -> list[tuple[T, Any]]:
    """The `RequireProperty` and `RefuseProperty` of ``requests`` (each with who filed it) that ``properties`` doesn't
    meet: a required property it lacks is unmet."""
    return [(who, r) for who, r in requests
            if (isinstance(r, RequireProperty) and properties.get(r.property) != r.value)
            or (isinstance(r, RefuseProperty) and r.property in properties and properties[r.property] == r.value)]


class Kernel(Protocol):
    """A scheme's computation of a due block: its output, computed itself or through ``original``, the workload's own,
    given inputs in place of the block's or None for the block's."""

    def __call__(self, due: BlockDue, original: Callable[[Mapping[str, Handle] | None], Handle]) -> Handle: ...


@dataclass(frozen=True)
class ComputeBlockWith:
    """A scheme's request to compute every due block with ``kernel``."""

    kernel: Kernel


@dataclass(frozen=True)
class StepDue:
    """Workload step ``step`` is next: ``active`` inputs started and not complete, ``queued`` of them waiting for a slot,
    and ``completing`` complete since the last step, which the step frees."""

    step: int
    active: int
    queued: int
    completing: int


class Gate(Protocol):
    """A scheme's hold on starting inputs: ``add`` starts an input when the scheme calls ``start``, and ``pending`` says
    whether it holds any not yet started. ``ready`` is asked before each step, may start inputs, and holds the step while
    False; ``open`` starts what it holds, and every later input at once."""

    def add(self, input: str, start: Callable[[], None]) -> None: ...

    def pending(self) -> bool: ...

    def ready(self, due: StepDue) -> bool: ...

    def open(self, why: str) -> None: ...


@dataclass(frozen=True)
class StartOnBoundary:
    """A scheme's request to start inputs only when ``gate`` starts them (a window's boundary)."""

    gate: Gate


class Route(Protocol):
    """A scheme's answer to a request on its route: called, on a thread of its own, with the request's first line, its
    headers (names lower-cased) and its body as it streams in, it returns the status and the text to answer with."""

    def __call__(self, first_line: str, headers: Mapping[str, str], body: Iterable[bytes]) -> tuple[int, str]: ...


@dataclass(frozen=True)
class ServeRoute:
    """A scheme's request that the workload's server answer requests on ``path`` through ``serve``."""

    path: str
    serve: Route


@dataclass(frozen=True)
class RefuseInput:
    """A scheme's refusal of input ``input``, answered with ``why``; filed at any time, serving included."""

    input: str
    why: str


class Broken(Exception):
    """Why a reader of the workload's (`StartBeside`'s readers) ends before its stream does."""


@dataclass(frozen=True)
class HoldRelease:
    """A scheme's request that no output of ``input`` leave before step ``until``."""

    input: str
    until: int


@dataclass(frozen=True)
class HoldResident:
    """A scheme's request that range ``range`` stay allocated until step ``until``."""

    range: int
    until: int


@dataclass(frozen=True)
class ReuseOldestFreed:
    """A scheme's request that a freed range be handed out again only after every range freed before it, so a freed
    range keeps its values as long as the workload has other ranges to hand out."""


@dataclass(frozen=True)
class FixRanges:
    """A scheme's request that the workload's memory hold exactly ``ranges`` ranges (`RangesDeclared`)."""

    ranges: int


@dataclass(frozen=True)
class ReleaseValue:
    """A scheme's request that registered value ``value``'s storage be released once it is resident, the scheme's
    kernels computing every block that reads it (`ComputeBlockWith`), and restored to ``restore()``, the same bytes,
    when the workload's memory hook comes off."""

    value: str
    restore: Callable[[], Handle]


@dataclass(frozen=True)
class StartBeside:
    """A scheme's request to start the process it registered as ``process``
    (`verity.protocols.resource_accounting.scheme.register_process`) beside the workload, called with the workload's
    readers it names by what they do (``server-request``, ``stream-token-ids``), in that order."""

    process: str
    readers: tuple[str, ...] = ()


type Request = (RequireProperty | RefuseProperty | ComputeBlockWith | StartOnBoundary | HoldRelease | HoldResident
                | StartBeside | ReuseOldestFreed | ServeRoute | RefuseInput | FixRanges | ReleaseValue)


__all__ = ["EVENTS", "HOOKS", "BlockDue", "ComputeBlockWith", "Declared", "Gate", "Handle", "HoldRelease", "HoldResident",
           "Broken", "FixRanges", "InputArrived", "Kernel", "OutputReleased", "OutputsComplete", "RangeAllocated", "RangeFreed", "RangeWritten",
           "RangesDeclared", "RefuseInput", "RefuseProperty", "RegisteredValues", "ReleaseValue", "Request", "RequireProperty", "ReuseOldestFreed",
           "Route", "RunBegan", "RunEnded", "ServeRoute", "StartBeside", "StartOnBoundary", "StepBegan", "StepDue", "StepEnded", "ValueProduced",
           "unmet"]
