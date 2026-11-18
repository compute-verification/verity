"""The messages between the adapter and the prover, in both directions.

In one process, through its one `Exchange` (`current`): the adapter's hooks emit the events of `verity.hooks` to the
subscribers of their hooks, and a scheme, attached by its name before the engine is built, files its requests, each
answered yes or refused by the scheme's name before serving starts.  A subscriber only reads, so it may attach at any
time; a request changes what is served: one on what the workload declares (a property, a process beside it) is filed
before the build's answer, and any other before serving begins, at the first step.  The exchange names no scheme; the adapter's hooks translate vLLM's engine,
requests, tokens and KV blocks into the circuit's words.

On disk, by their protocols' declarations (each protocol's `prover/workload.py`): a body is checked against its declared
type as it is written and as it is read, and its bytes are left as they are. `tests/workload_messages.toml` lists each
message's writers and readers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from verity import hooks as H
from verity.channel import Message, MessageTypes, Refused
from verity.protocols.proofs.spot.prover import workload as spot
from verity.protocols.resource_accounting import scheme as schemes
from verity.protocols.resource_accounting.compute.pouw.prover import workload as pouw
from verity.protocols.resource_accounting.memory.pous.prover import workload as pous
from verity.protocols.resource_accounting.network.prover import workload as certifier

__all__ = ["TYPES", "Exchange", "append", "body", "close", "current", "line", "lines", "read", "write"]

type Subscriber = Callable[[Any], None]


class Exchange:
    """One process's exchange: subscribers by hook, requests by scheme (filed until `answer`, fixed after it), and the
    workload's step, as the last `StepBegan` emitted it."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[tuple[str, Subscriber]]] = {h: [] for h in H.HOOKS}
        self.requests: list[tuple[str, H.Request]] = []
        self.declared: H.Declared | None = None
        self.fixed = False
        self._step: int | None = None
        #: the workload's readers a launcher offers a process beside it, by what they do; none: no launcher
        self.readers: Mapping[str, Callable[..., Any]] = {}
        #: the adapter's answer to a refused input (`RefuseInput`), called as it is filed with the input and the reason
        self.refuse: Callable[[str, str], None] | None = None

    @property
    def step(self) -> int:
        """The workload's step: 0 before serving begins."""
        return 0 if self._step is None else self._step

    @property
    def next_step(self) -> int:
        """The step the workload schedules next, which a step being scheduled is until its `StepBegan`."""
        return 0 if self._step is None else self._step + 1

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    def subscribe(self, name: str, hook: str, fn: Subscriber) -> Callable[[], None]:
        """``name`` receives every event of ``hook`` from now on, in the order the adapter emits them; returns its
        removal."""
        if hook not in H.HOOKS:
            raise Refused(name, f"{hook!r} is not a hook of a served run {H.HOOKS}")
        entry = (name, fn)
        self.subscribers[hook].append(entry)
        return lambda: self.subscribers[hook].remove(entry) if entry in self.subscribers[hook] else None

    def request(self, name: str, request: H.Request) -> None:
        """File scheme ``name``'s request; kernels and gates filed later run outside earlier ones."""
        kind = type(request).__name__
        if isinstance(request, H.RefuseInput):
            self.requests.append((name, request))
            if self.refuse is not None:
                self.refuse(request.input, request.why)
            return
        if isinstance(request, H.ServeRoute) and (other := self.route(request.path)) is not None:
            raise Refused(name, f"serves route {request.path}, which {other[0]} serves")
        if self.declared is not None and isinstance(request, (H.RequireProperty, H.RefuseProperty, H.StartBeside)):
            raise Refused(name, f"files {kind} after the build was answered; file it before the engine is built")
        if self._step is not None:
            raise Refused(name, f"files {kind} after serving began (step {self._step}); file it before the first step")
        self.requests.append((name, request))

    def _filed[R](self, kind: type[R]) -> list[tuple[str, R]]:
        return [(n, r) for n, r in self.requests if isinstance(r, kind)]

    def answer(self, declared: H.Declared) -> bool:
        """Answer every request before an engine is built: `Refused` under the first refused scheme's name, with all
        its reasons; else True when a scheme's required property fixes the build.  Emits ``declared``."""
        why: dict[str, list[str]] = {}
        for name, r in H.unmet(self.requests, declared.properties):
            verb = "requires" if isinstance(r, H.RequireProperty) else "refuses"
            why.setdefault(name, []).append(f"{verb} {r.property} = {r.value!r}: {r.why}")
        for name, r in self.requests:
            if isinstance(r, H.StartBeside) and (r.process not in schemes.PROCESSES or set(r.readers) - set(self.readers)):
                missing = sorted(set(r.readers) - set(self.readers))
                why.setdefault(name, []).append(
                    f"starts {r.process} beside the workload, which no scheme registered" if r.process not in schemes.PROCESSES
                    else f"starts {r.process} beside the workload with readers {missing} no launcher here offers")
        if why:
            name, reasons = next(iter(why.items()))
            raise Refused(name, "; ".join(reasons))
        self.declared = declared
        self.fixed = any(r.fixes for _, r in self._filed(H.RequireProperty))
        self.emit(declared)
        return self.fixed

    def wants(self, hook: str) -> bool:
        """Whether any scheme reads ``hook``'s events: an emitter builds no event nobody reads."""
        return bool(self.subscribers[hook])

    def emit(self, event: Any) -> None:
        if isinstance(event, H.StepBegan):
            self.step = event.step
        for _, fn in tuple(self.subscribers[event.hook]):
            fn(event)

    def compute(self, due: H.BlockDue, original: Callable[[Mapping[str, H.Handle] | None], H.Handle]) -> H.Handle:
        """A due block's output: emitted to ``blocks``' subscribers, then computed by the filed kernels, the workload's
        ``original`` innermost; inputs a kernel hands inward replace the block's for every kernel inside it."""
        self.emit(due)
        kernels = [r.kernel for _, r in self._filed(H.ComputeBlockWith)]

        def level(i: int, inputs: Mapping[str, H.Handle] | None) -> H.Handle:
            if i < 0:
                return original(inputs)
            seen = due if inputs is None else replace(due, inputs={**due.inputs, **inputs})
            return kernels[i](seen, lambda given: level(i - 1, inputs if given is None else {**(inputs or {}), **given}))

        return level(len(kernels) - 1, None)

    def computes(self) -> bool:
        """Whether a scheme computes due blocks: the adapter wraps its blocks only then."""
        return bool(self._filed(H.ComputeBlockWith))

    def start(self, input: str, start: Callable[[], None]) -> None:
        """Start ``input`` once every gate filed by `StartOnBoundary` starts it; with none, now."""
        for _, r in reversed(self._filed(H.StartOnBoundary)):
            start = (lambda g, s: lambda: g.add(input, s))(r.gate, start)
        start()

    def pending(self) -> bool:
        """Whether a gate holds an input not yet started."""
        return any(r.gate.pending() for _, r in self._filed(H.StartOnBoundary))

    def ready(self, due: H.StepDue) -> bool:
        """Whether step ``due.step`` may run: every gate is asked, and may start inputs."""
        return all([r.gate.ready(due) for _, r in self._filed(H.StartOnBoundary)])

    def open(self, why: str) -> None:
        """Every gate starts what it holds, and every later input at once."""
        for _, r in self._filed(H.StartOnBoundary):
            r.gate.open(why)

    def route(self, path: str) -> tuple[str, H.Route] | None:
        """The scheme serving ``path`` (`ServeRoute`) and its answer, None when none does."""
        return next(((n, r.serve) for n, r in self._filed(H.ServeRoute) if r.path == path), None)

    def refused(self, input: str) -> list[tuple[str, str]]:
        """Each scheme that refused input ``input`` (`RefuseInput`), with its reason."""
        return [(n, r.why) for n, r in self._filed(H.RefuseInput) if r.input == input]

    def resident(self, range_: int, step: int) -> bool:
        """Whether range ``range_`` must stay allocated at ``step`` (`HoldResident`)."""
        return any(r.range == range_ and step < r.until for _, r in self._filed(H.HoldResident))

    def oldest_freed_first(self) -> bool:
        """Whether a freed range is handed out again only after every range freed before it (`ReuseOldestFreed`)."""
        return bool(self._filed(H.ReuseOldestFreed))

    def beside(self) -> list[tuple[str, Callable[[], Any]]]:
        """The processes schemes asked to start beside the workload, each by its name and bound to the readers it
        named, for the launcher to call before serving."""
        out = []
        for _, r in self._filed(H.StartBeside):
            start, readers = schemes.PROCESSES[r.process], [self.readers[k] for k in r.readers]
            out.append((r.process, (lambda f, a: lambda: f(*a))(start, readers)))
        return out


_CURRENT: Exchange | None = None


def current() -> Exchange:
    """This process's exchange, opened on first use."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = Exchange()
    return _CURRENT


def close() -> None:
    """End this process's exchange; the next `current` opens a fresh one."""
    global _CURRENT
    _CURRENT = None


TYPES = MessageTypes((*spot.MESSAGES, *pouw.MESSAGES, *pous.MESSAGES, *certifier.MESSAGES))


def body(type_: str, data: bytes) -> bytes:
    """``data`` as the body of a ``type_`` message, unchanged; `verity.channel.Refused` when its declaration refuses
    it."""
    msg = Message(type_, body=data)
    if (why := TYPES.schema(msg.type, msg.version).why(msg.head(), msg.body)) is not None:
        raise Refused(*why)
    return data


def write(path: Path, type_: str, data: bytes) -> None:
    """Write ``data`` to ``path`` as a ``type_`` message's body."""
    path.write_bytes(body(type_, data))


def read(path: Path, type_: str) -> bytes:
    """``path``'s bytes, a ``type_`` message's body."""
    return body(type_, path.read_bytes())


def line(type_: str, data: bytes) -> bytes:
    """``data`` as one line of a JSON-lines file, a ``type_`` message's body."""
    return body(type_, data) + b"\n"


def append(path: Path, type_: str, data: bytes) -> None:
    """Append ``data`` to ``path`` as one line, a ``type_`` message's body."""
    with open(path, "ab") as fh:
        fh.write(line(type_, data))


def lines(path: Path, type_: str) -> list[bytes]:
    """``path``'s lines but blank ones, each a ``type_`` message's body."""
    return [body(type_, line) for line in path.read_bytes().splitlines() if line.strip()]
