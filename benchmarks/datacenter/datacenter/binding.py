"""The binding between the layers: a placement of a circuit onto a datacenter, and what it implies.

A `Placement` mirrors Lean's `Accounting.Requirement.Placement` (PR #2399): each value has the device it is produced on
(`leaf`) and the engine step it is produced at (`step`), and each device maps its steps onto ticks of the trusted clock
(`ticks`, strictly increasing: paced). The placement is private and committed; what is public are its properties, chief
among them the occupancy profile: how many frames each certified link carries in each bucket.

A crossing is a value produced in one block and read in another. Crossings are derived, never declared: there is no
"transfer gate", since moving a value doesn't change it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .logical import Circuit, Value
from .physical import Datacenter, Link


@dataclass(frozen=True)
class Stream:
    """Declared traffic that isn't a circuit value: `bits` bits a window from device `src` to device `dst`, for `windows`
    windows. Its frames count in the occupancy profile like any other. With a `digest` it is *registered* (a container
    image, a checkpoint restore): its content must match the digest, declared before the run, and it is not charged.
    Without one it is an *off-ramp* (telemetry, logs, metrics): its content isn't checked, and the account charges every
    bit."""

    name: str
    src: str
    dst: str
    bits: int
    windows: int = 1
    digest: bytes | None = None

    @property
    def registered(self) -> bool:
        return self.digest is not None


def Offramp(name: str, src: str, dst: str, bits: int, windows: int = 1) -> Stream:
    """A stream of data the circuit doesn't explain, charged in full."""
    return Stream(name, src, dst, bits, windows)


@dataclass
class Placement:
    """Where and when each value is produced."""

    circuit: Circuit
    datacenter: Datacenter
    leaf: dict[int, str]
    step: dict[int, int]
    ticks: dict[str, list[int]] = field(default_factory=dict)
    #: the cell of its device at which each value is produced, so one cell produces at most one value a step
    #: (`Accounting.Placed.cell`, `cell_inj`)
    cell: dict[int, int] = field(default_factory=dict)
    #: the device the circuit's outputs are delivered to (usually the outside world), or None
    deliver: str | None = None
    #: declared traffic that isn't a circuit value: registered streams (images) and off-ramps (telemetry)
    streams: tuple[Stream, ...] = ()
    #: registered state, each on the device the developer says holds it
    stores: tuple = ()

    def tick(self, v: Value | int) -> int:
        i = v.id if isinstance(v, Value) else v
        return self.ticks[self.leaf[i]][self.step[i]]

    def paced(self) -> bool:
        return all(all(a < b for a, b in zip(t, t[1:])) for t in self.ticks.values())

    def block(self, v: Value | int) -> str | None:
        i = v.id if isinstance(v, Value) else v
        b = self.datacenter.block_of(self.leaf[i])
        return b.name if b else None


def place(circuit: Circuit, datacenter: Datacenter, where: dict[str, str], ticks_per_step: int = 1,
          transfer_steps: int = 2, deliver: str | None = None, streams: tuple[Stream, ...] = (),
          stores: tuple = ()) -> Placement:
    """Place `circuit` on `datacenter`: `where` maps a value name (or `*`, the default) to a device; an unnamed value is
    computed where it is needed, on the device of its first reader that sits in a block (else the default). Each
    value's step is one past its operands' (plus `transfer_steps` when an operand crosses a block), and each device's
    step s begins at tick s·ticks_per_step. The outputs are delivered to `deliver`; `streams` and `stores` are declared beside the circuit."""
    readers: dict[int, list[int]] = {}
    for i, g in enumerate(circuit.gates):
        for j in g.operands:
            readers.setdefault(j, []).append(i)
    leaf: dict[int, str] = {}
    for i in reversed(range(len(circuit.gates))):
        name = circuit.names.get(i)
        inside = [leaf[r] for r in readers.get(i, []) if datacenter.block_of(leaf[r]) is not None]
        leaf[i] = where[name] if name in where else inside[0] if inside else where["*"]
    step: dict[int, int] = {}
    for i, g in enumerate(circuit.gates):
        here = datacenter.block_of(leaf[i])
        step[i] = 1 + max((step[j] + (transfer_steps if datacenter.block_of(leaf[j]) is not here else 0)
                           for j in g.operands), default=-1)
    horizon = max(step.values(), default=0) + 1 + transfer_steps
    devices = set(leaf.values()) | ({deliver} if deliver else set())
    ticks = {d: [s * ticks_per_step for s in range(horizon)] for d in devices}
    cell: dict[int, int] = {}
    used: dict[tuple[str, int], int] = {}
    for i in range(len(circuit.gates)):
        cell[i] = used.get((leaf[i], step[i]), 0)
        used[leaf[i], step[i]] = cell[i] + 1
    return Placement(circuit, datacenter, leaf, step, ticks, cell, deliver, tuple(streams), tuple(stores))


@dataclass(frozen=True)
class Crossing:
    """Value `value` leaves `src` (a device of one block) for `dst` (a device of another) along `path`, a list of links
    whose first and last cross the two blocks' boundaries. `at` is the tick it is read at: the reader's (Lean's `readOn`),
    when it must have arrived."""

    value: Value
    src: str
    dst: str
    path: tuple[Link, ...]
    at: int = 0


def route(datacenter: Datacenter, src: str, dst: str) -> tuple[Link, ...]:
    """The shortest path of links from device `src` to device `dst`."""
    adj: dict[str, list[tuple[str, Link]]] = {}
    for l in datacenter.links:
        adj.setdefault(l.a, []).append((l.b, l))
        adj.setdefault(l.b, []).append((l.a, l))
    prev: dict[str, tuple[str, Link] | None] = {src: None}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        if u == dst:
            break
        for v, l in adj.get(u, []):
            if v not in prev:
                prev[v] = (u, l)
                queue.append(v)
    if dst not in prev:
        raise ValueError(f"no path from {src} to {dst}")
    path, u = [], dst
    while prev[u] is not None:
        u, l = prev[u]
        path.append(l)
    return tuple(reversed(path))


def hops(path: tuple[Link, ...], src: str) -> list[tuple[str, str]]:
    """The links of `path` as directed hops (from, to), walking from `src`: a certifier keeps each direction's grid
    apart, egress from ingress."""
    out, here = [], src
    for l in path:
        there = l.b if l.a == here else l.a
        out.append((here, there))
        here = there
    return out


def crossings(p: Placement) -> list[Crossing]:
    """Every (value, reader device) pair where a value is read in a block other than its own, once each, at its first
    read."""
    first: dict[tuple[int, str], int] = {}
    for i, g in enumerate(p.circuit.gates):
        for j in g.operands:
            key = (j, p.leaf[i])
            first[key] = min(first.get(key, p.tick(i)), p.tick(i))
    if p.deliver:
        for j in p.circuit.outputs:
            ticks = p.ticks[p.deliver]
            first.setdefault((j, p.deliver), ticks[min(p.step[j] + 1, len(ticks) - 1)])
    out = []
    for (j, dst), at in first.items():
        src = p.leaf[j]
        if p.datacenter.block_of(src) is not p.datacenter.block_of(dst):
            out.append(Crossing(p.circuit.value(j), src, dst, route(p.datacenter, src, dst), at))
    return out


@dataclass
class Slot:
    """What an honest run puts on one directed boundary link in one bucket: committed crossings, packed together into
    frames, and frames of declared streams."""

    values: list[Value] = field(default_factory=list)
    streams: dict[str, int] = field(default_factory=dict)

    def frames(self, frame_bits: int) -> int:
        return -(-sum(v.bits for v in self.values) // frame_bits) + sum(self.streams.values())


def traffic(p: Placement, frame_bits: int, bucket: int, window: int) -> dict[tuple[str, str], dict[int, Slot]]:
    """The honest run's traffic on each directed boundary link (from, to), bucket by bucket: each crossing in the
    bucket of its read tick, each stream's frames at the first bucket of each of its windows."""
    boundary = {frozenset((l.a, l.b)) for l in p.datacenter.boundary}
    out: dict[tuple[str, str], dict[int, Slot]] = {}

    def slot(hop: tuple[str, str], b: int) -> Slot | None:
        return out.setdefault(hop, {}).setdefault(b, Slot()) if frozenset(hop) in boundary else None

    for x in crossings(p):
        for hop in hops(x.path, x.src):
            if (s := slot(hop, x.at // bucket)) is not None:
                s.values.append(x.value)
    for o in p.streams:
        for w in range(o.windows):
            for hop in hops(route(p.datacenter, o.src, o.dst), o.src):
                if (s := slot(hop, w * window)) is not None:
                    s.streams[o.name] = -(-o.bits // frame_bits)
    return out


def occupancy(p: Placement, frame_bits: int, bucket: int, window: int = 8) -> dict[tuple[str, str], dict[int, int]]:
    """The public occupancy profile: for each directed boundary link and each bucket of `bucket` ticks, the number of
    frames of `frame_bits` the honest run puts on it (`traffic`, packed)."""
    return {hop: {b: s.frames(frame_bits) for b, s in row.items()} for hop, row in traffic(p, frame_bits, bucket, window).items()}
