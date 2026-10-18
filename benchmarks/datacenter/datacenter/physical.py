"""The physical layer's declaration: devices, blocks, links and datacenters, with every capacity a named assumption.

This mirrors Lean's physical model (`Security/Definitions/Accounting/Physical.lean`, `Hardware`, and `Device.lean`,
`Device`). A datacenter is a set of blocks; a block is a set of devices behind one boundary; a link joins two devices,
and a link that crosses a block boundary must pass a network certifier (coverage, `Accounting.Covered`). Nothing here is
proved: each number is an `Assumption` with its source, and the accounting layer reads them by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["gpu", "cpu", "dram", "nic", "dpu", "bmc", "switch", "certifier", "challenger", "outside"]


@dataclass(frozen=True)
class Assumption:
    """A named physical premise: `value` in `unit`, from a data sheet, a measurement, or the developer's declaration."""

    name: str
    value: float
    unit: str
    source: Literal["datasheet", "measured", "declared"] = "datasheet"

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return f"{self.name}={self.value:g} {self.unit} ({self.source})"


@dataclass(frozen=True)
class Device:
    """A device, by its limits. `rate` is work per second for each op class (tensor-core multiply-adds, scalar ops),
    `memory` its bits of state, and `residual` the bits a verified wipe can't reach on it (secure-firmware flash, the
    labeller's own working memory)."""

    name: str
    kind: Kind
    rate: dict[str, Assumption] = field(default_factory=dict)
    memory: Assumption | None = None
    residual: Assumption | None = None
    trusted: bool = False
    #: bits per second the device reads from its own memory
    bandwidth: Assumption | None = None
    #: bits of on-chip state (SRAM, registers, caches) a read pass needn't touch memory for
    on_chip: Assumption | None = None

    def __repr__(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class Link:
    """A link between two devices, with its capacity in bits per second and its one-way latency in seconds."""

    a: str
    b: str
    capacity: Assumption
    latency: float = 0.0
    kind: Literal["pcie", "nvlink", "ethernet", "wan"] = "ethernet"


@dataclass
class Block:
    """Devices behind one boundary: what the accounting layer credits as a unit."""

    name: str
    devices: list[Device]

    def __contains__(self, device: str) -> bool:
        return any(d.name == device for d in self.devices)


@dataclass
class Datacenter:
    """Blocks, the fabric devices between them (switches, certifiers), and the links. `boundary` lists the links that
    cross a block boundary; `validate` checks that every one of them has a certifier at its block's end."""

    name: str
    blocks: list[Block]
    fabric: list[Device] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)

    @property
    def devices(self) -> dict[str, Device]:
        out = {d.name: d for b in self.blocks for d in b.devices}
        out.update({d.name: d for d in self.fabric})
        return out

    def block_of(self, device: str) -> Block | None:
        return next((b for b in self.blocks if device in b), None)

    @property
    def boundary(self) -> list[Link]:
        """The links with exactly one end inside a block, or with ends in two blocks."""
        return [l for l in self.links if self.block_of(l.a) is not self.block_of(l.b)]

    def certifier_of(self, link: Link) -> Device | None:
        """The certifier on a boundary link: the link's fabric end, when that end is a certifier."""
        for end in (link.a, link.b):
            d = self.devices[end]
            if d.kind == "certifier":
                return d
        return None

    def validate(self) -> list[str]:
        """Every way the declaration falls short of the model; empty when it is an instance of it."""
        problems = []
        names = self.devices
        for l in self.links:
            for end in (l.a, l.b):
                if end not in names:
                    problems.append(f"link {l.a}-{l.b} names an unknown device {end!r}")
        for l in self.boundary:
            if self.certifier_of(l) is None:
                problems.append(f"boundary link {l.a}-{l.b} passes no certifier (coverage)")
        for b in self.blocks:
            for d in b.devices:
                if d.kind == "gpu" and not d.rate:
                    problems.append(f"{d!r} declares no rate")
        return problems

    def assumptions(self) -> list[Assumption]:
        """Every named premise the declaration makes, once each."""
        out: dict[str, Assumption] = {}
        for d in self.devices.values():
            for a in [*d.rate.values(), d.memory, d.residual, d.bandwidth, d.on_chip]:
                if a is not None:
                    out[a.name] = a
        for l in self.links:
            out[l.capacity.name] = l.capacity
        return list(out.values())


@dataclass
class Fleet:
    """Several datacenters under one developer. A link between two datacenters is a boundary like any other, so it
    passes a certifier at each end; the outside world is everything no certifier covers."""

    datacenters: list[Datacenter]
    links: list[Link] = field(default_factory=list)

    def merged(self) -> Datacenter:
        """The fleet as one datacenter: every block, fabric device and link, the inter-datacenter links included, so a
        circuit can be placed across sites with the same `place`. Device names must be unique across the fleet."""
        return Datacenter("+".join(dc.name for dc in self.datacenters),
                          [b for dc in self.datacenters for b in dc.blocks],
                          [d for dc in self.datacenters for d in dc.fabric],
                          [l for dc in self.datacenters for l in dc.links] + list(self.links))

    def validate(self) -> list[str]:
        problems = [f"{dc.name}: {p}" for dc in self.datacenters for p in dc.validate()]
        seen: dict[str, str] = {}
        for dc in self.datacenters:
            for d in dc.devices:
                if d in seen:
                    problems.append(f"device {d!r} is in both {seen[d]} and {dc.name}")
                seen[d] = dc.name
        for l in self.links:
            for end in (l.a, l.b):
                owner = next((dc for dc in self.datacenters if end in dc.devices), None)
                if owner is None:
                    problems.append(f"inter-datacenter link {l.a}-{l.b} names an unknown device {end!r}")
                elif owner.devices[end].kind != "certifier":
                    problems.append(f"inter-datacenter link {l.a}-{l.b} ends at {end!r}, not a certifier")
        return problems
