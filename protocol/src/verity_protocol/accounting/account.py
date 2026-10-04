"""The account: the developer's aggregated representation of one epoch of its datacenter.

It says which circuits ran, in which isolation units, on which devices, using how much work and declared state. The
developer computes it, commits to it at the end of the epoch, and accounting checks the commitment against what the
subprotocols established.

A circuit is named by its digest under the circuit format. What a circuit is, and what it computes, is the accounting
spec's (`lean/`, with `verity/ir/PROTOCOL.md` until the encoding is stated there too); the library that builds, queries
and evaluates circuits (`verity.ir`, `verity.evaluation` today) implements those definitions and is held to them by
conformance vectors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import NewType

CircuitId = NewType("CircuitId", str)


@dataclass(frozen=True)
class Device:
    id: str
    hardware: str  # a census hardware id
    work: int  # capacity per epoch, in work units of record: a named assumption, measured by the probes
    memory_bits: int


@dataclass(frozen=True)
class SpatialUnit:
    """Machines whose every boundary link a warden taps."""

    id: str
    devices: tuple[str, ...]
    links: tuple[str, ...]


@dataclass(frozen=True)
class Inventory:
    devices: tuple[Device, ...]
    spatial_units: tuple[SpatialUnit, ...]

    def device(self, id: str) -> Device:
        return next(d for d in self.devices if d.id == id)


@dataclass(frozen=True)
class IsolationUnit:
    """A part of the declared computation that runs on one spatial unit between two wipes; the policy bounds each."""

    id: str
    spatial_unit: str


@dataclass(frozen=True)
class Execution:
    circuit: CircuitId
    unit: str  # its isolation unit
    device: str
    work: int  # work units of record
    state_bits: int  # declared state it keeps resident


@dataclass(frozen=True)
class Account:
    epoch: int
    units: tuple[IsolationUnit, ...]
    executions: tuple[Execution, ...]

    def declared(self, device: str) -> tuple[int, int]:
        """(work, state bits) the account declares on `device`."""
        mine = [e for e in self.executions if e.device == device]
        return sum(e.work for e in mine), sum(e.state_bits for e in mine)

    def encode(self) -> bytes:
        body = {"epoch": self.epoch, "units": [asdict(u) for u in self.units], "executions": [asdict(e) for e in self.executions]}
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
