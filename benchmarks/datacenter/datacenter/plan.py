"""The verifier's plan for an epoch: public, because the verifier's coins and draws are.

A `Plan` says which epoch is audited, which devices the challenger wipes and when, which stores it challenges and which
reads it demands, and how many occupancy profiles the developer registered (so the choice among them is advice worth
log2 of that, once).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import Challenge, Reads, Wipe
from .time import Interval


@dataclass(frozen=True)
class Plan:
    epoch: Interval
    wipes: tuple[Wipe, ...] = ()
    challenges: tuple[Challenge, ...] = ()
    reads: tuple[Reads, ...] = ()
    profiles: int = 1

    def __init__(self, epoch: Interval, wipes=(), challenges=(), reads=(), profiles: int = 1):
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "wipes", tuple(wipes))
        object.__setattr__(self, "challenges", tuple(challenges))
        object.__setattr__(self, "reads", tuple(reads))
        object.__setattr__(self, "profiles", profiles)
