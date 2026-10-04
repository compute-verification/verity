"""Events: what one principal's actors send to another's, in one ordered log.

An actor is one principal's (a prover's, a verifier's, a replayer's, a pinger's, a warden's) part in one protocol: a
transducer over the events that reach its principal. Written as a generator: it yields a `Wait`, what it sends and the
kinds it waits for, each with the principal it takes that kind from (anyone can send to anyone); it is resumed with the
earliest event it waits for, and returns its last sends when it is done. An event it is not waiting for keeps until it
is, so the order it waits in loses nothing, and one it never waits for never reaches it. What it saw and did in
between is its own state and never enters the log. An event's body is a frozen dataclass whose class is its kind,
defined by the module that owns it.

A send goes to a principal, and every actor of that principal receives it; a send to no one is a record, which the
sender's own principal receives. So actors need no names: an actor reads the kinds it reads, each from the principal
it trusts for it (`receive`, `gather`, `collect`).
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Send:
    body: object
    to: str | None = None  # a principal; None for a record


@dataclass(frozen=True)
class Event:
    sender: str  # principals, both
    to: str
    body: object


@dataclass(frozen=True)
class Wait:
    sends: tuple[Send, ...]
    kinds: Mapping[type, str]  # each kind waited for -> the principal it is taken from


Actor = Generator[Wait, Event, tuple[Send, ...]]


def gather(trusted: Mapping[type, str], *sends: Send) -> Generator[Wait, Event, dict[type, object]]:
    """Send `sends`, then wait until a body of each kind in `trusted` has come from the principal it names; return the
    first of each."""
    got: dict[type, object] = {}
    while len(got) < len(trusted):
        event = yield Wait(sends, {kind: sender for kind, sender in trusted.items() if kind not in got})
        sends = ()
        got[type(event.body)] = event.body
    return got


def receive(kind: type, sender: str, *sends: Send) -> Generator[Wait, Event, object]:
    """Send `sends`, then wait for a `kind` from `sender`."""
    return (yield from gather({kind: sender}, *sends))[kind]


def collect(kind: type, until: type, sender: str, *sends: Send) -> Generator[Wait, Event, tuple]:
    """Send `sends`, then take every `kind` from `sender`, in order, until an `until` from it: a stream."""
    got = []
    while True:
        event = yield Wait(sends, {kind: sender, until: sender})
        sends = ()
        if isinstance(event.body, until):
            return tuple(got)
        got.append(event.body)


def tell(*sends: Send) -> Actor:
    """An actor that sends `sends` and nothing more: a declaration, or a reading."""
    yield from ()
    return sends
