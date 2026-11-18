"""The messages between the adapter and the prover, in both directions.

In one process, through the `Exchange`: the adapter's hooks emit the events of `verity.hooks`, and a scheme, attached by
its name before the engine is built, subscribes to their hooks and files its requests, each answered yes or refused by
the scheme's name before serving starts.  The exchange names no scheme; the adapter's hooks translate vLLM's engine,
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
from verity.protocols.resource_accounting.compute.pouw.prover import workload as pouw
from verity.protocols.resource_accounting.memory.pous.prover import workload as pous
from verity.protocols.resource_accounting.network.prover import workload as certifier

__all__ = ["TYPES", "Exchange", "append", "body", "close", "current", "line", "lines", "open_exchange", "read", "write"]

type Subscriber = Callable[[Any], None]


class Exchange:
    """One served run's exchange: subscribers by hook and requests by scheme, filed until `answer` and fixed after it."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[tuple[str, Subscriber]]] = {h: [] for h in H.HOOKS}
        self.requests: list[tuple[str, H.Request]] = []
        self.declared: H.Declared | None = None
        self.fixed = False

    def _filing(self, name: str, what: str) -> None:
        if self.declared is not None:
            raise Refused(name, f"files {what} after the requests were answered; attach before the engine is built")

    def subscribe(self, name: str, hook: str, fn: Subscriber) -> None:
        """Scheme ``name`` receives every event of ``hook``, in the order the adapter emits them."""
        self._filing(name, f"on {hook!r}")
        if hook not in H.HOOKS:
            raise Refused(name, f"{hook!r} is not a hook of a served run {H.HOOKS}")
        self.subscribers[hook].append((name, fn))

    def request(self, name: str, request: H.Request) -> None:
        """File scheme ``name``'s request; kernels and gates filed later run outside earlier ones."""
        self._filing(name, type(request).__name__)
        self.requests.append((name, request))

    def _filed[R](self, kind: type[R]) -> list[tuple[str, R]]:
        return [(n, r) for n, r in self.requests if isinstance(r, kind)]

    def answer(self, declared: H.Declared) -> bool:
        """Answer every request before the engine is built: `Refused` under the first refused scheme's name, with all
        its reasons; else True when a scheme's required property fixes the build.  Emits ``declared``."""
        if self.declared is not None:
            raise Refused("exchange", "the requests were answered already")
        why: dict[str, list[str]] = {}
        for name, r in self.requests:
            if isinstance(r, H.RequireProperty) and declared.properties.get(r.property) != r.value:
                why.setdefault(name, []).append(f"requires {r.property} = {r.value!r}: {r.why}")
            elif isinstance(r, H.RefuseProperty) and declared.properties.get(r.property) == r.value:
                why.setdefault(name, []).append(f"refuses {r.property} = {r.value!r}: {r.why}")
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
        for _, fn in self.subscribers[event.hook]:
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

    def held(self, input: str, step: int) -> bool:
        """Whether an output of ``input`` may not leave at ``step`` (`HoldRelease`)."""
        return any(r.input == input and step < r.until for _, r in self._filed(H.HoldRelease))

    def resident(self, range_: int, step: int) -> bool:
        """Whether range ``range_`` must stay allocated at ``step`` (`HoldResident`)."""
        return any(r.range == range_ and step < r.until for _, r in self._filed(H.HoldResident))


_CURRENT: Exchange | None = None
_IDLE = Exchange()


def open_exchange() -> Exchange:
    """The served run's exchange, opened before its engine is built; refuses a second one in the process."""
    global _CURRENT
    if _CURRENT is not None:
        raise Refused("exchange", "an exchange is open in this process already; close it first")
    _CURRENT = Exchange()
    return _CURRENT


def current() -> Exchange:
    """The open exchange, or an idle one no scheme reads."""
    return _CURRENT if _CURRENT is not None else _IDLE


def close() -> None:
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
