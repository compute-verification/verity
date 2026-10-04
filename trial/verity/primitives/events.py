"""Events: what one actor sends to others, in one ordered log.

An actor is one principal's (a prover's, a verifier's, a warden's, a pinger's) part in one protocol: a transducer over
the events addressed to it. Written as a generator: it yields what it sends, receives the next event addressed to it,
and returns its last sends when it is done. What it saw and did in between is its own state and never enters the log.
An event's body is a frozen dataclass whose class is its kind, defined by the module that owns it; a send to nobody is
a record, kept only in its sender's view. An actor takes each kind only from the principal it trusts for it (`receive`,
`gather`), since anyone can address anyone.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Send:
    body: object
    to: tuple[str, ...] = ()


@dataclass(frozen=True)
class Event:
    sender: str
    to: tuple[str, ...]
    body: object


Actor = Generator[tuple[Send, ...], Event, tuple[Send, ...]]


def within(name: str, principal: str) -> bool:
    """Whether the actor `name` is `principal`'s (or is the actor `principal` names)."""
    return name == principal or name.startswith(principal + "/")


def gather(trusted: Mapping[type, str], *sends: Send) -> Generator[tuple[Send, ...], Event, dict[type, object]]:
    """Send `sends`, then wait until a body of each kind in `trusted` has come from the principal it names, in any order;
    return the first of each, passing over everything else."""
    out, got = sends, {}
    while len(got) < len(trusted):
        event = yield out
        out = ()
        kind = type(event.body)
        if kind in trusted and kind not in got and within(event.sender, trusted[kind]):
            got[kind] = event.body
    return got


def receive(kind: type, sender: str, *sends: Send) -> Generator[tuple[Send, ...], Event, object]:
    """Send `sends`, then wait for a `kind` from `sender`."""
    return (yield from gather({kind: sender}, *sends))[kind]


def tell(*sends: Send) -> Actor:
    """An actor that sends `sends` and nothing more: a declaration, or a reading."""
    yield sends
    return ()
