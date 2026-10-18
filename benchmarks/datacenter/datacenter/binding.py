"""The bridge between the layers: a placement of a circuit onto a datacenter, what it implies, and what of it is public.

A `Placement` mirrors Lean's `Accounting.Placed` (and #2399's `Accounting.Requirement.Placement`): each value has the
device it is produced on (`leaf`), the engine step it is produced at (`step`) and a cell of that device (`cell`, one
value a cell a step), and each device maps its steps onto ticks of the trusted clock (`ticks`, strictly increasing).
The placement is private and committed. A crossing is a value produced in one block and read in another; crossings are
derived, never declared, since moving a value doesn't change it.

What is public is the placement's `Profile`: the occupancy each certifier port must show, the work each block is to be
credited with, the trusted data each block consumes, and the registered stores. `audit` and `account` read the profile;
only `audit`, which runs inside the proof, reads the placement itself.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .logical import Circuit, Value
from .memory import Store
from .physical import Datacenter, Link, Port
from .plan import Plan
from .time import Interval


@dataclass
class Placement:
    """Where and when each value is produced."""

    circuit: Circuit
    datacenter: Datacenter
    leaf: dict[int, str]
    step: dict[int, int]
    ticks: dict[str, list[int]] = field(default_factory=dict)
    cell: dict[int, int] = field(default_factory=dict)
    #: the device the circuit's outputs are delivered to (usually the outside world), or None
    deliver: str | None = None
    #: registered state, each on the device the registration says holds it
    stores: tuple[Store, ...] = ()

    def tick(self, v: Value | int) -> int:
        i = v.id if isinstance(v, Value) else v
        return self.ticks[self.leaf[i]][self.step[i]]

    def paced(self) -> bool:
        return all(all(a < b for a, b in zip(t, t[1:])) for t in self.ticks.values())

    def block(self, v: Value | int) -> str | None:
        i = v.id if isinstance(v, Value) else v
        b = self.datacenter.block_of(self.leaf[i])
        return b.name if b else None

    def public(self, plan: Plan) -> Profile:
        """The placement's public properties for the epoch `plan` audits."""
        work: dict[str, float] = {b.name: 0.0 for b in self.datacenter.blocks}
        forced: dict[str, float] = {b.name: 0.0 for b in self.datacenter.blocks}
        trusted = {i for i, g in enumerate(self.circuit.gates) if g.op == "input" and g.input.source == "trusted"}
        consumed: set[tuple[int, str]] = set()
        for i, g in enumerate(self.circuit.gates):
            if (b := self.block(i)) is not None:
                work[b] += g.work
                consumed.update((j, b) for j in trusted.intersection(g.operands))
        for j, b in consumed:
            forced[b] += self.circuit.gates[j].bits
        stores = {s.name: s for s in self.stores}
        stores.update({w.store.name: w.store for w in plan.wipes})
        return Profile(self.datacenter, plan, occupancy(self), work, forced, stores, self.circuit.unit)


@dataclass(frozen=True)
class Profile:
    """A placement's public properties, with the declaration and the verifier's plan they are read against: the frames
    each certifier port shows in each bucket (`occupancy`), the work each block is to be credited with (`work`, in the
    circuit's `unit`), the trusted data each block consumes (`forced`, in bits), and the registered stores."""

    datacenter: Datacenter
    plan: Plan
    occupancy: dict[Port, dict[int, int]]
    work: dict[str, float]
    forced: dict[str, float]
    stores: dict[str, Store]
    unit: str = "gate"

    @property
    def windows(self) -> range:
        """The record windows of the epoch: every port must have a record for each."""
        g, e = self.datacenter.grid, self.plan.epoch
        per = g.bucket * g.window
        return range(e.start // per, -(-e.end // per))


def place(c: Circuit, dc: Datacenter, default: str, *, deliver: str | None = None, stores: tuple[Store, ...] = (),
          ticks_per_step: int = 1, transfer_steps: int = 2, **where: str) -> Placement:
    """Place `c` on `dc`: each keyword names a value and the device it is produced on; an unnamed value is computed
    where it is needed, on the device of its first reader that sits in a block, else on `default`. A value's step is one
    past its operands' (plus `transfer_steps` when an operand crosses a block), and each device's step s begins at tick
    s·ticks_per_step. The outputs are delivered to `deliver`."""
    devices = dc.devices
    unknown = [n for n in where if n not in c.names.values()] + [d for d in [default, deliver, *where.values()]
                                                                 if d is not None and d not in devices]
    if unknown:
        raise ValueError(f"place names what isn't there: {unknown}")
    for v in c.inputs:
        if v.source == "trusted" and not devices[where.get(v.name, default)].trusted:
            raise ValueError(f"trusted input {v.name!r} must come from a trusted device, not {where.get(v.name, default)!r}")
    readers: dict[int, list[int]] = {}
    for i, g in enumerate(c.gates):
        for j in g.operands:
            readers.setdefault(j, []).append(i)
    leaf: dict[int, str] = {}
    for i in reversed(range(len(c.gates))):
        name = c.names.get(i)
        inside = [leaf[r] for r in readers.get(i, []) if dc.block_of(leaf[r]) is not None]
        leaf[i] = where[name] if name in where else inside[0] if inside else default
    step: dict[int, int] = {}
    for i, g in enumerate(c.gates):
        here = dc.block_of(leaf[i])
        step[i] = 1 + max((step[j] + (transfer_steps if dc.block_of(leaf[j]) is not here else 0) for j in g.operands),
                          default=-1)
    horizon = max(step.values(), default=0) + 1 + transfer_steps
    ticks = {d: [s * ticks_per_step for s in range(horizon)] for d in set(leaf.values()) | ({deliver} if deliver else set())}
    cell: dict[int, int] = {}
    used: dict[tuple[str, int], int] = {}
    for i in range(len(c.gates)):
        cell[i] = used.get((leaf[i], step[i]), 0)
        used[leaf[i], step[i]] = cell[i] + 1
    for s in stores:
        if s.device not in devices:
            raise ValueError(f"store {s.name!r} is registered on an unknown device {s.device!r}")
    return Placement(c, dc, leaf, step, ticks, cell, deliver, tuple(stores))


@dataclass(frozen=True)
class Crossing:
    """Value `value` leaves `src` (a device of one block) for `dst` (a device of another) along `path`. `at` is the tick
    it is read at, the reader's (Lean's `readOn`): when it must have arrived."""

    value: Value
    src: str
    dst: str
    path: tuple[Link, ...]
    at: int = 0


def route(dc: Datacenter, src: str, dst: str) -> tuple[Link, ...]:
    """The shortest path of links from device `src` to device `dst`."""
    adj: dict[str, list[tuple[str, Link]]] = {}
    for l in dc.links:
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
    """The links of `path` as directed hops (from, to), walking from `src`."""
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


def traffic(p: Placement) -> dict[Port, dict[int, list[Value]]]:
    """The honest run's traffic at each certifier port, bucket by bucket: the crossing values it carries, each in the
    bucket of its read tick."""
    out: dict[Port, dict[int, list[Value]]] = {}
    bucket = p.datacenter.grid.bucket
    for x in crossings(p):
        for hop in hops(x.path, x.src):
            if (port := p.datacenter.port(*hop)) is not None:
                out.setdefault(port, {}).setdefault(x.at // bucket, []).append(x.value)
    return out


def frames(values: list[Value], frame_bits: int) -> int:
    """The frames that carry `values`, packed together."""
    return -(-sum(v.bits for v in values) // frame_bits)


def occupancy(p: Placement) -> dict[Port, dict[int, int]]:
    """The public occupancy profile: for each certifier port and each bucket, the frames the honest run shows there."""
    fb = p.datacenter.grid.frame_bits
    return {port: {b: frames(vs, fb) for b, vs in row.items()} for port, row in traffic(p).items()}
