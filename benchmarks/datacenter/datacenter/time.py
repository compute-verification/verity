"""Time: intervals on the one trusted clock.

There is one physical clock: the ticks of the trusted devices (the certifiers and the challenger), whose real duration
is the datacenter's `Cadence.tick`. Each protocol keeps its own logical time (a placement's engine steps, a certifier's
buckets, a challenge's deadline) and maps it monotonically onto these ticks. Every claim's `during` is an interval on
this clock. A tick has a real duration because some guarantees read lengths, not just order: a deadline is only as good
as the work a device can do before it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Interval:
    """The ticks [start, end) of the trusted clock: Lean's `Accounting.Window` with `len = end − start`."""

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
