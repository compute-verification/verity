"""A device's clock readings, as Core's clock in Lean states them (`Definitions/Core/Challenger.lean`: `Reading`,
`Reading.elapsed`, `Reading.tick`; how a device's clock runs against real time is the premise `ClockWithin`, which nothing
here checks).

A **reading** is the count in nanoseconds of the clock of the device registered under `key`, taken as the device makes
an entry of its own log (`verity.attested.Attested`). Two devices' readings are never compared: `elapsed` refuses them.
A deadline reads the elapsed bound, `0 ≤ elapsed ≤ limit` (`within`). `resolution_ns` is how finely one of the
machine's clocks ticks. What reads a clock is the runtime's `RealClock` (`verity.runtime.sources`), given the machine's
clock and its name; this module reads none.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reading:
    """A reading `ns` of the clock of the device registered under `key`."""

    key: str
    ns: int


def elapsed(a: Reading, b: Reading) -> int:
    """Reading `b` less reading `a`, in nanoseconds of their one device's clock; two devices' readings are refused."""
    if a.key != b.key:
        raise ValueError(f"the readings are of two devices, {a.key!r} and {b.key!r}, which are never compared")
    return b.ns - a.ns


def tick(period_ns: int, t: Reading) -> int:
    """Reading `t` counted in a registered period of `period_ns` nanoseconds."""
    if period_ns < 1:
        raise ValueError(f"a period is a whole number of nanoseconds, at least 1, not {period_ns}")
    return t.ns // period_ns


def within(a: Reading, b: Reading, limit_ns: int) -> bool:
    """The elapsed bound a deadline reads: `0 ≤ elapsed(a, b) ≤ limit_ns`."""
    return 0 <= elapsed(a, b) <= limit_ns


def resolution_ns(clock_id: int) -> float:
    """How finely the machine's clock `clock_id` (`time.CLOCK_MONOTONIC`, `time.CLOCK_MONOTONIC_RAW`) ticks, in ns."""
    return time.clock_getres(clock_id) * 1e9


__all__ = ["Reading", "elapsed", "resolution_ns", "tick", "within"]
