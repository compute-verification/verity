"""The one clock: the readings of an auditor's device, and what every device and verifier does with them.

It mirrors Core's clock in Lean (`Definitions/Core/Challenger.lean`, `Definitions/Assumptions/Clock.lean`):

- A **reading** (`Reading`) is the count of one auditor's device's raw monotonic clock in nanoseconds, in an entry that
  device signs. Its `key` is the key the device is registered under (`Site.Loc.key`), and two devices' readings are
  never compared: `elapsed` refuses them by name. A value in no entry of its own device's log is not a reading, and
  outside tests `stamp` is the only way to make one: it reads the device's clock as it appends the entry to the log the
  device signs, and refuses a log signed under another key.
- A **tick** (`tick`) is a view of a reading: the reading counted in a registered period.
- A prover's entry has no time of its own; it has a **bracket** (`between`): by index in the device's log, after one of
  the device's readings and before another.
- How a device's clock runs against real time is the named premise `clock-within-drift` (`ClockWithin`; `NotSlow`
  for a deadline and a salt's age, `NotFast` for a bucket's period), which nothing here checks.

The **tick source** (`ticks`) waits for tick `k` on a device's clock from an origin reading: a wait that ends early
waits again, so no tick runs before its time, and a tick later than the device's registered lateness is itself an entry
of the device's log, with its reading.

A `Device` is the key, the clock and the two figures registered with the key: the clock's resolution and how late the
device may act on it. Its clock is given, so a test or a replay gives its own; `raw_ns` is the host's raw monotonic
clock. The module imports the standard library alone.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

NS = 10**9
"""Nanoseconds a second."""


class ClockRefused(Exception):
    """The host gives no raw monotonic clock, so no device reads one here: a device runs on no other clock."""


def raw_ns() -> int:
    """The host's raw monotonic clock in nanoseconds (`CLOCK_MONOTONIC_RAW`): the count of the host's oscillator, which
    no time service steers or steps, so its rate against true time is the hardware's alone. Raises `ClockRefused` where
    the host has none."""
    try:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    except (AttributeError, OSError) as e:
        raise ClockRefused(f"this host gives no raw monotonic clock ({type(e).__name__}), and a device runs on no "
                           "other") from None


def raw_resolution_ns() -> int:
    """The resolution of `raw_ns` on this host, in whole nanoseconds, at least 1."""
    try:
        return max(1, round(time.clock_getres(time.CLOCK_MONOTONIC_RAW) * NS))
    except (AttributeError, OSError) as e:
        raise ClockRefused(f"this host gives no raw monotonic clock ({type(e).__name__})") from None


@dataclass(frozen=True, slots=True)
class Reading:
    """A reading of the auditor's device registered under `key`: its clock's count `ns`, in nanoseconds."""

    key: str
    ns: int


def elapsed(a: Reading, b: Reading) -> int:
    """Reading `b` less reading `a`, in nanoseconds of their one device's clock; two devices' readings are refused."""
    if a.key != b.key:
        raise ValueError(f"the readings are of two devices, {a.key!r} and {b.key!r}, and two devices' readings are "
                         "never compared")
    return b.ns - a.ns


def tick(period_ns: int, t: Reading) -> int:
    """Reading `t` counted in a registered period of `period_ns` nanoseconds."""
    if period_ns < 1:
        raise ValueError(f"a period is a whole number of nanoseconds, at least 1, not {period_ns}")
    return t.ns // period_ns


@dataclass(frozen=True, slots=True)
class Device:
    """An auditor's device: the key it is registered under, its clock (nanoseconds, given), the resolution of that clock
    and how late it may act on it, both in nanoseconds and registered with the key."""

    key: str
    clock: Callable[[], int]
    resolution_ns: int
    lateness_ns: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("a device is registered under a key, and this one names none")
        if self.resolution_ns < 1 or self.lateness_ns < 0:
            raise ValueError(f"device {self.key!r}: a resolution is at least 1 ns and a lateness at least 0, not "
                             f"{self.resolution_ns} and {self.lateness_ns}")


@dataclass(frozen=True, slots=True)
class Entry:
    """One entry of a device's log: what the device signs (`data`), with the reading it took as it wrote the entry, or
    with none for an entry that carries no reading (a value it recorded, a coin it released)."""

    data: Any
    reading: Reading | None = None


@dataclass
class Log:
    """A device's own log, oldest first, which the device signs under `key`."""

    key: str
    entries: list[Entry] = field(default_factory=list)

    def record(self, data: Any) -> int:
        """Append an entry that carries no reading; its index."""
        self.entries.append(Entry(data))
        return len(self.entries) - 1


def stamp(device: Device, log: Log, data: Any = None) -> Reading:
    """Read `device`'s clock and append the reading, with `data`, to `log`, the log the device signs; the
    reading. Refuses a log signed under another key, and a reading below the last reading in the log."""
    if log.key != device.key:
        raise ValueError(f"device {device.key!r} stamps only its own log, not the log signed under {log.key!r}")
    t = Reading(device.key, device.clock())
    last = next((e.reading for e in reversed(log.entries) if e.reading is not None), None)
    if last is not None and t.ns < last.ns:
        raise ValueError(f"device {device.key!r}'s clock read {t.ns} ns after it read {last.ns} ns: a device's "
                         "readings never run backwards")
    log.entries.append(Entry(data, t))
    return t


def between(log: Log, i: int, a: int, b: int) -> bool:
    """Whether entry `i` of `log` lies after the reading at index `a` and before the reading at index `b`, both of the
    device whose log it is."""
    n = len(log.entries)
    if not (0 <= a < i < b < n):
        return False
    return all((t := log.entries[j].reading) is not None and t.key == log.key for j in (a, b))


@dataclass(frozen=True, slots=True)
class Tick:
    """Tick `k`: how late it ran on the device's clock, in nanoseconds, and the reading of its entry in the device's log
    when that is past the device's lateness (None when it ran in time)."""

    k: int
    late_ns: int
    late: Reading | None = None


async def ticks(device: Device, origin: Reading, period_ns: int, *, sleep: Callable[[float], Awaitable[None]],
                log: Log) -> AsyncIterator[Tick]:
    """Tick 0, 1, 2, ... of `device`'s clock from `origin`, `period_ns` apart, each once the clock reads its time: a
    wait (`sleep`, in seconds) that ends early waits again. A tick later than the device's lateness is stamped in
    `log`."""
    if origin.key != device.key:
        raise ValueError(f"device {device.key!r} ticks from its own reading, not one of {origin.key!r}")
    tick(period_ns, origin)
    k = 0
    while True:
        deadline = origin.ns + k * period_ns
        await sleep((deadline - device.clock()) / NS)
        while (late := device.clock() - deadline) < 0:
            await sleep(-late / NS)
        yield Tick(k, late, stamp(device, log, {"late_tick": k}) if late > device.lateness_ns else None)
        k += 1


__all__ = ["NS", "ClockRefused", "Device", "Entry", "Log", "Reading", "Tick", "between", "elapsed", "raw_ns",
           "raw_resolution_ns", "stamp", "tick", "ticks"]
