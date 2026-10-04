"""Memory accounting: how much of the site's memory PoUS shows holding useful data, and so how much is left.

verifier    reads PoUS's bound and a capacity reading from `capacity_from`; records the account
capacity    the reading, run by `capacity_from` (a pinger)
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives.events import Actor, Send, gather, tell
from verity.protocols.pous import SpaceBound


@dataclass(frozen=True)
class Capacity:
    bytes: int


@dataclass(frozen=True)
class MemoryAccount:
    """Of `capacity` bytes, at least `held` hold the useful data; at most `unaccounted` hold anything else. Except with
    probability `delta`."""

    capacity: int
    held: int
    unaccounted: int
    delta: float


@dataclass(frozen=True)
class Memory:
    capacity_from: str  # the principal the verifier takes capacity from

    def verifier(self) -> Actor:
        got = yield from gather({SpaceBound: "verifier", Capacity: self.capacity_from})
        space, capacity = got[SpaceBound], got[Capacity].bytes
        held = int(space.bytes * space.rho) if space.accepted else 0
        return (Send(MemoryAccount(capacity, held, max(capacity - held, 0), space.delta)),)

    def capacity(self, bytes_: int) -> Actor:
        return tell(Send(Capacity(bytes_), "verifier"))
