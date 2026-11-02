"""The physical layer's declaration: devices, blocks, links and datacenters, with every number a named assumption.

This mirrors Lean's physical model (`Security/Definitions/Accounting/Physical.lean`, `Hardware`): a datacenter is a
tree of locations whose leaves are devices; a block is a set of devices behind one boundary (`Hardware.SpatialUnit`);
a link joins two devices, and a link that crosses a block boundary must pass a network certifier (coverage,
`Accounting.Covered`). Certifiers and challengers are the trusted devices: the auditor's hardware. Nothing here is
proved: each number is an `Assumption` with its source, and the account reads them by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Kind = Literal["gpu", "cpu", "dram", "nic", "dpu", "bmc", "switch", "certifier", "challenger", "outside"]

#: the kinds of device the auditor trusts: its own hardware, on machines the prover doesn't control
TRUSTED: frozenset[str] = frozenset({"certifier", "challenger"})


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
class DeviceEntry:
    """A device, by its limits: `rate` is work per second in each unit of work it can do (a unit is whatever the
    audits time: Boolean gates, tensor-core multiply-adds), `memory` its bits of state, `bandwidth` the bits a second it
    reads from that memory, `write` the bits a second a wipe can write into it (None: its links'), `on_chip` its bits
    of SRAM, registers and caches, and `residual` the bits a verified wipe can't reach (secure-firmware flash, the
    labeller's own working memory)."""

    name: str
    kind: Kind
    rate: dict[str, Assumption] = field(default_factory=dict)
    memory: Assumption | None = None
    bandwidth: Assumption | None = None
    on_chip: Assumption | None = None
    residual: Assumption | None = None
    write: Assumption | None = None

    @property
    def trusted(self) -> bool:
        return self.kind in TRUSTED

    def __repr__(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class LinkEntry:
    """A link between two devices: its capacity in bits a second and its one-way latency, each an assumption."""

    a: str
    b: str
    capacity: Assumption
    latency: Assumption | None = None
    kind: Literal["pcie", "nvlink", "upi", "ethernet", "wan"] = "ethernet"

    @property
    def delay(self) -> float:
        return float(self.latency) if self.latency else 0.0


@dataclass(frozen=True)
class Port:
    """One direction of a certifier: what leaves its block (`egress`) or enters it (`ingress`). The certifier keeps one
    grid per port (`NetTiming.Grid`) and signs one record per port and window."""

    certifier: str
    direction: Literal["egress", "ingress"]

    def __repr__(self) -> str:
        return f"{self.certifier}:{'out' if self.direction == 'egress' else 'in'}"


@dataclass(frozen=True)
class Cadence:
    """The certifiers' fixed parameters (W1) and the trusted clock. Frames are `frame_bits` long (Lean's B), a bucket is
    `bucket_ticks` ticks (`Hardware.bucket` maps a tick to its bucket), and a record covers a window of `buckets`
    buckets (T: a window's `NetTiming.Grid` is T buckets of slots). Each window picks one of `jitter_classes` public
    clock-sync values (the real certifier's sync set, Lean's jitter classes), and `tick` is one tick's real duration."""

    frame_bits: int = 32
    bucket_ticks: int = 1
    buckets: int = 8
    jitter_classes: int = 128
    tick: Assumption = Assumption("tick", 1e-3, "s", "declared")

    @property
    def window_ticks(self) -> int:
        return self.bucket_ticks * self.buckets

    def seconds(self, ticks: float) -> float:
        return ticks * float(self.tick)


@dataclass(frozen=True)
class Zone:
    """Devices behind one boundary: the unit the account credits."""

    name: str
    devices: tuple[DeviceEntry, ...]

    def __init__(self, name: str, devices):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "devices", tuple(devices))

    def __contains__(self, device: str) -> bool:
        return any(d.name == device for d in self.devices)


@dataclass
class Datacenter:
    """Blocks, the fabric devices between them (switches, certifiers, the challenger, the outside world), the links,
    the certifiers' cadence, and whether they seal their pair links."""

    name: str
    blocks: list[Zone]
    fabric: list[DeviceEntry] = field(default_factory=list)
    links: list[LinkEntry] = field(default_factory=list)
    cadence: Cadence = field(default_factory=Cadence)
    #: every link between two blocks' certifiers is a sealed pair link (the certifier's `Proxy(pair=…)`), so what
    #: crosses from one block to another is ciphertext off the two certifiers, and the export is charged at the
    #: facility boundary only (`Placed._exported`). Off by default.
    sealed: bool = False

    @property
    def devices(self) -> dict[str, DeviceEntry]:
        out = {d.name: d for b in self.blocks for d in b.devices}
        out.update({d.name: d for d in self.fabric})
        return out

    def block_of(self, device: str) -> Zone | None:
        return next((b for b in self.blocks if device in b), None)

    def block(self, name: str) -> Zone:
        return next(b for b in self.blocks if b.name == name)

    @property
    def boundary(self) -> list[LinkEntry]:
        """The links with exactly one end in a block, or with ends in two blocks."""
        return [l for l in self.links if self.block_of(l.a) is not self.block_of(l.b)]

    def uncovered_links(self) -> list[LinkEntry]:
        """The links outside every block with no trusted device at either end (a switch-to-switch fabric link, an outside
        host's link to a switch), where two machines the auditor doesn't control meet with no certifier between them.
        The network certifiers cover every link when this is empty."""
        devices = self.devices
        inside = lambda l: self.block_of(l.a) is not None and self.block_of(l.a) is self.block_of(l.b)  # noqa: E731
        return [l for l in self.links if not inside(l)
                and not any(devices[e].trusted for e in (l.a, l.b) if e in devices)]

    def port(self, a: str, b: str) -> Port | None:
        """The certifier port a hop from device `a` to device `b` passes, if the hop crosses a boundary at a certifier."""
        devices = self.devices
        if self.block_of(a) is self.block_of(b):
            return None
        if devices[b].kind == "certifier" and self.block_of(a) is not None:
            return Port(b, "egress")
        if devices[a].kind == "certifier" and self.block_of(b) is not None:
            return Port(a, "ingress")
        return None

    @property
    def ports(self) -> list[Port]:
        out = []
        for l in self.boundary:
            for hop in ((l.a, l.b), (l.b, l.a)):
                if (p := self.port(*hop)) is not None and p not in out:
                    out.append(p)
        return sorted(out, key=repr)

    def link(self, a: str, b: str) -> LinkEntry:
        return next(l for l in self.links if {l.a, l.b} == {a, b})

    def uplink(self, certifier: str) -> LinkEntry:
        """The link between a certifier and the block it watches."""
        return next(l for l in self.links if certifier in (l.a, l.b)
                    and self.block_of(l.b if l.a == certifier else l.a) is not None)

    def watched(self, port: Port) -> Zone:
        """The block whose boundary `port` watches."""
        l = self.uplink(port.certifier)
        return self.block_of(l.b if l.a == port.certifier else l.a)

    def validate(self) -> list[str]:
        """Every way the declaration falls short of the model; empty when it is an instance of it."""
        problems = []
        names = self.devices
        for l in self.links:
            for end in (l.a, l.b):
                if end not in names:
                    problems.append(f"link {l.a}-{l.b} names an unknown device {end!r}")
        for l in self.boundary:
            if l.a in names and l.b in names and self.port(l.a, l.b) is None and self.port(l.b, l.a) is None:
                problems.append(f"boundary link {l.a}-{l.b} passes no certifier (coverage)")
        return problems

    def assumptions(self) -> list[Assumption]:
        """Every named premise the declaration makes, once each."""
        out: dict[str, Assumption] = {self.cadence.tick.name: self.cadence.tick}
        for d in self.devices.values():
            for a in [*d.rate.values(), d.memory, d.bandwidth, d.on_chip, d.residual, d.write]:
                if a is not None:
                    out[a.name] = a
        for l in self.links:
            for a in (l.capacity, l.latency):
                if a is not None:
                    out[a.name] = a
        return list(out.values())

    def prefixed(self, prefix: str, keep: frozenset[str] = frozenset({"world"})) -> Datacenter:
        """The same datacenter with every device, block and the datacenter itself renamed `prefix.<name>` (but the
        devices in `keep`), so several can join a fleet."""
        r = lambda s: s if s in keep else f"{prefix}.{s}"  # noqa: E731
        return Datacenter(prefix, [Zone(r(b.name), [replace(d, name=r(d.name)) for d in b.devices]) for b in self.blocks],
                          [replace(d, name=r(d.name)) for d in self.fabric],
                          [replace(l, a=r(l.a), b=r(l.b)) for l in self.links], self.cadence, self.sealed)


@dataclass
class Fleet:
    """Several datacenters under one developer, and the links between them. Blocks are the only units the account
    credits, and coverage is per block: every link off a block passes a certifier of its own, wherever the link's other
    end is. A link between datacenters is fabric, as a switch is, so it needs no certifier, and the outside world is
    everything off every block."""

    datacenters: list[Datacenter]
    links: list[LinkEntry] = field(default_factory=list)

    def merged(self) -> Datacenter:
        """The fleet as one datacenter, so a circuit can be placed across sites with the same `place`. Devices shared by
        name (the outside world) appear once. Its pair links are sealed when every site's are."""
        seen, fabric = set(), []
        for dc in self.datacenters:
            for d in dc.fabric:
                if d.name not in seen:
                    seen.add(d.name)
                    fabric.append(d)
        return Datacenter("+".join(dc.name for dc in self.datacenters), [b for dc in self.datacenters for b in dc.blocks],
                          fabric, [l for dc in self.datacenters for l in dc.links] + list(self.links),
                          self.datacenters[0].cadence, all(dc.sealed for dc in self.datacenters))

    def validate(self) -> list[str]:
        """Each datacenter's own problems, a device name in two of them (but the outside world's), and a link between
        them that names an unknown device or leaves a block without passing a certifier."""
        problems = [f"{dc.name}: {p}" for dc in self.datacenters for p in dc.validate()]
        seen: dict[str, str] = {}
        for dc in self.datacenters:
            for d, dev in dc.devices.items():
                if d in seen and dev.kind != "outside":
                    problems.append(f"device {d!r} is in both {seen[d]} and {dc.name}")
                seen[d] = dc.name
        merged = self.merged()
        for l in self.links:
            unknown = [end for end in (l.a, l.b) if end not in merged.devices]
            problems += [f"inter-datacenter link {l.a}-{l.b} names an unknown device {end!r}" for end in unknown]
            if unknown or merged.block_of(l.a) is merged.block_of(l.b):
                continue
            if merged.port(l.a, l.b) is None and merged.port(l.b, l.a) is None:
                problems.append(f"inter-datacenter link {l.a}-{l.b} passes no certifier (coverage)")
        return problems
