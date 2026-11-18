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

#: ``ctx["process"]`` of a scheme's ``attach(exchange, ctx)`` where the workload's front and its engine run in two
#: processes, each with its own exchange: FRONT takes the workload's inputs (`InputArrived`, a scheme's `Route`,
#: an `AdmitInput` that refuses), ENGINE steps it (the gates and every other hook). Without the key one process does both.
#: ``ctx["feed"]`` is then a directory, one for the two, whose files a scheme's front appends to and its engine reads;
#: ``ctx["settings"]`` are the scheme's settings (`Declared.settings`).
FRONT, ENGINE = "front", "engine"
#: ``ctx["process"]`` of a scheme attached in a process of its own, which the served runner launches for its piece
#: (`Scheme.pieces`) beside the workload's front and engine, neither of them: the wipe's, at block 0's opening, before
#: any engine process is on the card
PIECE = "piece"

type Handle = object
"""A device buffer as the serving system holds it: opaque here, read only by the scheme's own prover code."""


@dataclass(frozen=True)
class Declared:
    """``declared``: what the workload declares before it runs: the card's file id, its properties (``stage``: serve,
    capture or commit; ``world``; ``eager``; ``async_steps``, whether a step is scheduled before the last one's outputs
    are read; a precision, a cache, a sampler), the schemes attached to it by name and, under each one's name, its
    settings."""

    hook: ClassVar[str] = "declared"
    card: str
    properties: Mapping[str, Any]
    schemes: tuple[str, ...] = ()
    settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    step: int = 0


@dataclass(frozen=True)
class InputArrived:
    """``steps``: input ``input`` of ``length`` values arrived, at ``step``, with the ``headers`` it came with (names
    lower-cased), its ``first_line`` and the ``user`` it names (None where it names none).  ``user`` is the input's own
    value: a scheme keeps it to itself."""

    hook: ClassVar[str] = "steps"
    input: str
    length: int
    step: int
    headers: Mapping[str, str] = field(default_factory=dict)
    first_line: str = ""
    user: str | None = None


@dataclass(frozen=True)
class StepBegan:
    """``steps``: workload step ``step`` begins, computing for ``inputs`` over ``slots`` slots (the positions the step
    computes, drafted ones excluded), with ``queued`` started inputs left waiting for a slot. Steps are numbered from 0,
    each ends in one `StepEnded`, and a step that computes nothing has ``slots == 0`` and no ``inputs``."""

    hook: ClassVar[str] = "steps"
    step: int
    inputs: tuple[str, ...] = ()
    slots: int = 0
    queued: int = 0


@dataclass(frozen=True)
class StepComputed:
    """``steps``: step ``step``'s blocks are all issued and its outputs not yet read: between its `StepBegan` and its
    `StepEnded`, after the next step's `StepBegan` where steps overlap. A step that computes nothing has none."""

    hook: ClassVar[str] = "steps"
    step: int


@dataclass(frozen=True)
class StepEnded:
    hook: ClassVar[str] = "steps"
    step: int


@dataclass(frozen=True)
class PhaseOpened:
    """``steps``: execution phase ``phase`` (numbered from 0 by its committer) opens before any of its values is
    committed; its first step is ``step``. Phases are emitted only where a committer commits the workload in phases."""

    hook: ClassVar[str] = "steps"
    phase: int
    step: int


@dataclass(frozen=True)
class PhaseClosed:
    """``steps``: execution phase ``phase`` closes after step ``step``, before the next one opens, its values committed
    under the Program whose digest is ``program``."""

    hook: ClassVar[str] = "steps"
    phase: int
    step: int
    program: bytes


@dataclass(frozen=True)
class PhaseCommitted:
    """``steps``: closed execution phase ``phase``'s committed values have root ``root`` (empty: it committed none).
    After its `PhaseClosed`, and where the committer finalizes off the serving loop, on the committer's thread, possibly
    after later phases opened."""

    hook: ClassVar[str] = "steps"
    phase: int
    root: bytes


@dataclass(frozen=True)
class BlockResult:
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


#: what a `MemoryChanged` says: of its values (``registered``) or of its ranges (the rest)
CHANGES = ("registered", "allocated", "written", "freed")


@dataclass(frozen=True)
class MemoryChanged:
    """``memory``: the workload's memory changed at ``step``: its registered ``values`` resident, by name, ``operands``
    mapping each one a block reads as its weight operand to that block's ``key`` (`BlockDue.key`) (``registered``);
    ``ranges`` handed to ``input`` (``allocated``; ``reused``: handed again, holding values another input wrote),
    ``count`` slots from ``offset`` of the one range of ``ranges`` written (``written``), or ``ranges`` returned with no
    reader left (``freed``)."""

    hook: ClassVar[str] = "memory"
    change: str
    ranges: tuple[int, ...] = ()
    step: int = 0
    input: str | None = None
    reused: bool = False
    offset: int = 0
    count: int = 0
    values: tuple[tuple[str, Handle], ...] = ()
    operands: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.change not in CHANGES:
            raise ValueError(f"MemoryChanged: change {self.change!r} is not one of {CHANGES}")
        if self.change == "registered" and self.ranges:
            raise ValueError("MemoryChanged: a registered change names its values, not ranges")
        if self.change != "registered" and (not self.ranges or self.values or self.operands):
            raise ValueError(f"MemoryChanged: a {self.change} change names its ranges, and no values")
        if self.change == "written" and len(self.ranges) != 1:
            raise ValueError(f"MemoryChanged: a write is to one range, not {len(self.ranges)}")


@dataclass(frozen=True)
class OutputLeft:
    """``steps``: input ``input``'s outputs are complete and have left the workload, at ``step``; an input refused or
    ended before it reached the engine leaves at step 0."""

    hook: ClassVar[str] = "steps"
    input: str
    step: int


#: every event a served run emits, each read through its ``hook``
EVENTS = (Declared, InputArrived, StepBegan, StepComputed, StepEnded, BlockResult, BlockDue, RangesDeclared,
          MemoryChanged, OutputLeft, PhaseOpened, PhaseClosed, PhaseCommitted)


@dataclass(frozen=True)
class RequireProperty:
    """A scheme's request that the workload declare ``property`` as ``value``; ``fixes``: the scheme's computation is
    defined by the declared build, so serving may not override it."""

    property: str
    value: Any
    why: str
    fixes: bool = False


def unmet[T](requests: Iterable[tuple[T, Any]], properties: Mapping[str, Any]) -> list[tuple[T, Any]]:
    """The `RequireProperty` of ``requests`` (each with who filed it) that ``properties`` doesn't meet: a required
    property it lacks is unmet."""
    return [(who, r) for who, r in requests
            if isinstance(r, RequireProperty) and properties.get(r.property) != r.value]


class Kernel(Protocol):
    """A scheme's computation of a due block: its output, computed itself or through ``original``, the workload's own,
    given inputs in place of the block's or None for the block's. A capability the scheme files at attach
    (``exchange.compute_with(name, kernel)``), not an entry."""

    def __call__(self, due: BlockDue, original: Callable[[Mapping[str, Handle] | None], Handle]) -> Handle: ...


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

    def take(self, record: Mapping[str, Any]) -> None:
        """One record of what the gate orders by, fed on the thread that steps the workload."""


class Front(Protocol):
    """A scheme's frontend, which the workload's server hands each request: ``key`` names a request by its headers
    (None: not one the scheme orders), ``submit`` and ``abort`` forward one, and ``passed`` says the one ``key`` named
    has left the server."""

    def key(self, headers: Mapping[str, str]) -> Any: ...

    def submit(self, admission: Any) -> None: ...

    def abort(self, input: str) -> None: ...

    def passed(self, key: Any) -> None: ...

    def summary(self) -> Mapping[str, Any]: ...


#: what an `AdmitInput` says: ``hold`` the inputs until its ``gate`` starts each (a window's boundary), or ``refuse``
#: its ``input``, answered with ``why``
ADMITS = ("hold", "refuse")


@dataclass(frozen=True)
class AdmitInput:
    """A scheme's word on admitting inputs (`ADMITS`). A hold starts an input when its ``gate`` starts it and ends when
    the gate opens; a refusal is filed at any time, serving included."""

    admit: str
    gate: Gate | None = None
    input: str | None = None
    why: str = ""

    def __post_init__(self) -> None:
        if self.admit not in ADMITS:
            raise ValueError(f"AdmitInput admits by one of {ADMITS}, not {self.admit!r}")
        if (self.admit == "hold") != (self.gate is not None) or (self.admit == "refuse") != (self.input is not None):
            raise ValueError(f"AdmitInput {self.admit!r}: a hold names a gate and no input, a refusal an input and no "
                             f"gate (gate {self.gate!r}, input {self.input!r})")


class Route(Protocol):
    """A scheme's answer to a request on its route: called, on a thread of its own, with the request's first line, its
    headers (names lower-cased) and its body as it streams in, it returns the status and the text to answer with. A
    capability the scheme files at attach (``exchange.serve(name, where, route)``), not an entry."""

    def __call__(self, first_line: str, headers: Mapping[str, str], body: Iterable[bytes]) -> tuple[int, str]: ...


class Broken(Exception):
    """Why a reader of the workload's (`StartBeside`'s readers) ends before its stream does."""


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
    kernels computing every block that reads it (`Kernel`), and restored to ``restore()``, the same bytes,
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


type Request = RequireProperty | AdmitInput | StartBeside | ReuseOldestFreed | FixRanges | ReleaseValue


__all__ = ["ADMITS", "CHANGES", "EVENTS", "HOOKS", "AdmitInput", "BlockDue", "BlockResult", "Declared",
           "Gate", "Handle", "Broken", "ENGINE", "FRONT", "PIECE", "FixRanges", "Front", "InputArrived", "Kernel",
           "MemoryChanged", "OutputLeft", "RangesDeclared", "ReleaseValue", "Request", "RequireProperty", "ReuseOldestFreed",
           "PhaseClosed", "PhaseCommitted", "PhaseOpened", "Route", "StartBeside", "StepBegan", "StepComputed",
           "StepDue", "StepEnded", "unmet"]
