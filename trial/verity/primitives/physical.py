"""The physical world, as the protocols represent it: a clock's readings (`Window`), a device type's facts (`Device`)
and a site's devices (`Site`).

Who reads a clock or counts a site decides what a reading is worth: the prover's word is an assumption, a reading by
the verifier's hardware a measurement. A device type's facts are the census's, and assumptions either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """A clock's readings, in seconds: when something opened, and when it closed."""

    opened: float
    closed: float

    @property
    def seconds(self) -> float:
        return self.closed - self.opened


@dataclass(frozen=True)
class Device:
    """A device type's facts, the census's."""

    work: float = 0.0  # work a second, at peak
    memory: int = 0  # bytes, volatile
    egress: float = 0.0  # bits a second it can send out of the site


@dataclass(frozen=True)
class Site:
    """The devices at a site, by type: the most there at any time in a window."""

    devices: Mapping[str, int]  # device type -> how many

    def supply(self, census: Mapping[str, Device], fact: str) -> tuple[float, tuple[str, ...]]:
        """The site's devices' `fact`, summed; and what in the report the census cannot vouch for (a count that is not
        one, a type it has no entry for), which supplies nothing."""
        counted = {d: n for d, n in self.devices.items() if isinstance(n, int) and n >= 0}
        faults = (
            *(f"{n!r} {d}s" for d, n in self.devices.items() if d not in counted),
            *(f"no census entry for {d}" for d in counted if d not in census),
        )
        return sum(n * getattr(census[d], fact) for d, n in counted.items() if d in census), faults
