"""Compute coverage: how much of the fleet's compute over the work's window went to the registered work, and so how
much is left where anything else could have run.

    verifier    reads integrity's profile and the work bound over one stream, and the inventory from `inventory_from`;
                records the coverage

Capacity is the window's seconds, by the work's clock, times the fleet's peak. Covered is the work bound, less the share
of the gates in units integrity could not rule out being wrong: work counts for the registered computation only where
integrity says that is what ran. More work than the fleet could have done means the inventory or a peak is wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.accounting.coverage.inventory import Fleet
from verity.accounting.integrity.profile import IntegrityProfile
from verity.accounting.utilization.work import Window, WorkBound
from verity.primitives.events import Actor, Send, gather


@dataclass(frozen=True)
class ComputeCoverage:
    """Of the `capacity` the fleet `inventory_from` reported could have done in the work's window, at least `covered`
    went to the registered work as integrity checked it, and at most `headroom` to anything else. Except with
    probability `delta`."""

    stream: str
    window: Window
    capacity: int
    inventory_from: str
    covered: int
    headroom: int
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Compute:
    peak: Mapping[str, float]  # work per second, per device type: the census's, an assumption

    def verifier(self, inventory_from: str) -> Actor:
        got = yield from gather({IntegrityProfile: "verifier", WorkBound: "verifier", Fleet: inventory_from})
        profile, work, fleet = got[IntegrityProfile], got[WorkBound], got[Fleet]
        seconds = work.window.closed - work.window.opened
        capacity = int(seconds * sum(n * self.peak.get(device, 0) for device, n in fleet.devices.items()))
        reasons = (
            *(("integrity not established",) if not profile.accepted else ()),
            *(("integrity and work are over different streams",) if profile.stream != work.stream else ()),
            *(f"no peak for {device}" for device in fleet.devices if device not in self.peak),
            *((f"{work.work} work in a window the fleet could do {capacity} in",) if work.work > capacity else ()),
        )
        gates = profile.gates
        covered = 0 if reasons or not gates.total else work.work * (gates.total - gates.wrong) // gates.total
        delta = profile.delta + work.delta
        coverage = ComputeCoverage(
            work.stream, work.window, capacity, inventory_from, covered, capacity - covered, delta, reasons
        )
        return (Send(coverage),)
