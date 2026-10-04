"""Events: what one principal's actors send to another's, in one ordered log.

An actor is one principal's (a prover's, a verifier's, a replayer's, a pinger's, a warden's) part in one protocol: a
transducer over the events that reach its principal. Written as a generator: it yields what it sends, receives the
next event, and returns its last sends when it is done. What it saw and did in between is its own state and never
enters the log. An event's body is a frozen dataclass whose class is its kind, defined by the module that owns it.

A send goes to a principal, and every actor of that principal receives it, in log order; a send to no one is a record,
which the sender's own principal receives. So actors need no names: an actor reads the kinds it reads, and takes each
only from the principal it trusts for it (`receive`, `gather`), since anyone can send to anyone.
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


Actor = Generator[tuple[Send, ...], Event, tuple[Send, ...]]


def gather(trusted: Mapping[type, str], *sends: Send) -> Generator[tuple[Send, ...], Event, dict[type, object]]:
    """Send `sends`, then wait until a body of each kind in `trusted` has come from the principal it names, in any
    order; return the first of each, passing over everything else."""
    out, got = sends, {}
    while len(got) < len(trusted):
        event = yield out
        out = ()
        kind = type(event.body)
        if kind in trusted and kind not in got and event.sender == trusted[kind]:
            got[kind] = event.body
    return got


def receive(kind: type, sender: str, *sends: Send) -> Generator[tuple[Send, ...], Event, object]:
    """Send `sends`, then wait for a `kind` from `sender`."""
    return (yield from gather({kind: sender}, *sends))[kind]


def tell(*sends: Send) -> Actor:
    """An actor that sends `sends` and nothing more: a declaration, or a reading."""
    yield sends
    return ()
