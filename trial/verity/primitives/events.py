"""Events: what one party sends to others, in one ordered log.

Every party (a prover, a verifier, a warden, a pinger) is a transducer over the events addressed to it. Written as a
generator: it yields what it sends, receives the next event addressed to it, and returns its last sends when it is done.
What it saw and did in between is its own state and never enters the log. An event's body is a frozen dataclass whose
class is its kind, defined by the module that owns it; a send to nobody is a record, kept only in its sender's view.
"""

from __future__ import annotations

from collections.abc import Generator
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


Party = Generator[tuple[Send, ...], Event, tuple[Send, ...]]


def receive(kind: type, *sends: Send) -> Generator[tuple[Send, ...], Event, Event]:
    """Send `sends`, then wait for the next event whose body is a `kind`, passing over any other."""
    out = sends
    while True:
        event = yield out
        out = ()
        if isinstance(event.body, kind):
            return event
