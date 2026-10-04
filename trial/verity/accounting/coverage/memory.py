"""Memory coverage: how much of the fleet's memory held the encoded data when space challenged it, and so how much is
left where anything else could have been kept. Volatile memory only: the site has no disks.

    verifier    reads the space bound, and the inventory from `inventory_from`; records the coverage
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.accounting.coverage.inventory import Fleet
from verity.accounting.utilization.space import SpaceBound
from verity.primitives.events import Actor, Send, gather


@dataclass(frozen=True)
class MemoryCoverage:
    """Of the `capacity` bytes the fleet `inventory_from` reported, at least `covered` held the encoded data when
    challenged, and at most `headroom` anything else. Except with probability `delta`."""

    capacity: int
    inventory_from: str
    covered: int
    headroom: int
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Memory:
    size: Mapping[str, int]  # bytes per device type: the census's

    def verifier(self, inventory_from: str) -> Actor:
        got = yield from gather({SpaceBound: "verifier", Fleet: inventory_from})
        space, fleet = got[SpaceBound], got[Fleet]
        capacity = sum(n * self.size.get(device, 0) for device, n in fleet.devices.items())
        held = int(space.bytes * space.rho)
        reasons = (
            *(("space not established",) if not space.accepted else ()),
            *(f"no size for {device}" for device in fleet.devices if device not in self.size),
            *((f"{held} bytes held in a fleet of {capacity}",) if held > capacity else ()),
        )
        covered = 0 if reasons else held
        return (Send(MemoryCoverage(capacity, inventory_from, covered, capacity - covered, space.delta, reasons)),)
