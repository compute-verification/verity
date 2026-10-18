"""Time: one trusted clock, intervals on it, and the events trusted devices put on it.

There is one physical clock: the ticks of the trusted devices (the certifiers and the challenger). Each protocol keeps
its own logical time (a placement's engine steps, a certifier's buckets, a challenge's deadline) and maps it monotonically
onto these ticks. A tick has a real duration, because some guarantees read lengths, not just order: a deadline is only as
good as the work a device can do before it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    """The trusted timebase: `tick` seconds per tick."""

    tick: float = 1e-3

    def seconds(self, ticks: int) -> float:
        return ticks * self.tick

    def ticks(self, seconds: float) -> int:
        return round(seconds / self.tick)


@dataclass(frozen=True, order=True)
class Interval:
    """The ticks [start, end) of the trusted clock."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"an interval ends before it starts: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def __contains__(self, other: Interval | int) -> bool:
        if isinstance(other, Interval):
            return self.start <= other.start and other.end <= self.end
        return self.start <= other < self.end

    def __repr__(self) -> str:
        return f"[{self.start}, {self.end})"


@dataclass(frozen=True)
class Wipe:
    """A verified memory wipe of `device` during `at`: the challenger fills its memory with labels it can recompute and
    spot-checks them before the interval ends, so state that survives the wipe fits in the device's residual."""

    device: str
    at: Interval
