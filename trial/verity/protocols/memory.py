"""Memory accounting: how much of the site's memory PoUS shows holding useful data, and so how much is left.

    verifier    reads PoUS's bound and a capacity reading from the principal it is told to take it from; records the
                account
    pinger      measures memory capacity
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
    name: str = "memory"

    def verifier(self, capacity_from: str, to: tuple[str, ...] = ()) -> dict[str, Actor]:
        return {f"verifier/{self.name}": self._verifier(capacity_from, to)}

    def pinger(self, capacity: int) -> dict[str, Actor]:
        return {f"pinger/{self.name}": tell(Send(Capacity(capacity), to=(f"verifier/{self.name}",)))}

    def _verifier(self, capacity_from: str, to: tuple[str, ...]) -> Actor:
        got = yield from gather({SpaceBound: "verifier", Capacity: capacity_from})
        space, capacity = got[SpaceBound], got[Capacity].bytes
        held = int(space.bytes * space.rho) if space.accepted else 0
        return (Send(MemoryAccount(capacity, held, max(capacity - held, 0), space.delta), to=to),)
