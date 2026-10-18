"""The verifier's plan for an epoch: public, because the verifier's coins and draws are.

A `Plan` says which epoch is audited, which devices the challenger wipes and when, which stores it challenges, which
read windows it opens, which blocks are out for maintenance, and how many occupancy profiles the developer registered
(so the choice among them is advice worth log2 of that, once).
"""
from __future__ import annotations

from dataclasses import dataclass

from .memory import Challenge, ReadWindow, Wipe
from .time import Interval


@dataclass(frozen=True)
class Maintenance:
    """Block `block` out of service during `during`: someone the auditor doesn't control has it in hand. Its certifiers'
    records aren't required then, everything its devices hold may leave with whoever services it, and anything may come
    back in with them, so only a wipe after it bounds what the block holds when it rejoins."""

    block: str
    during: Interval


@dataclass(frozen=True)
class Plan:
    epoch: Interval
    wipes: tuple[Wipe, ...] = ()
    challenges: tuple[Challenge, ...] = ()
    reads: tuple[ReadWindow, ...] = ()
    maintenance: tuple[Maintenance, ...] = ()
    profiles: int = 1

    def __init__(self, epoch: Interval, wipes=(), challenges=(), reads=(), maintenance=(), profiles: int = 1):
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "wipes", tuple(wipes))
        object.__setattr__(self, "challenges", tuple(challenges))
        object.__setattr__(self, "reads", tuple(reads))
        object.__setattr__(self, "maintenance", tuple(maintenance))
        object.__setattr__(self, "profiles", profiles)

    def out(self, block: str, during: Interval) -> bool:
        """Whether `block` is out for maintenance at any tick of `during`."""
        return any(m.block == block and m.during.start < during.end and during.start < m.during.end
                   for m in self.maintenance)
