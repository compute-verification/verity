"""The bridge between the layers: a placement of a Program onto a datacenter, what it implies, and what of it is public.

`Placed` puts each root node of the Program on a device at an engine step: each Call where it runs, each run input
where it enters (the outside world, a block's own device, a trusted device), and each device maps its steps onto ticks of
the trusted clock (`ticks`, strictly increasing). A registered input needs no device: registration put a copy of it in
every block that reads it, before the epoch, as serving holds its weights and as the remote audit registers each copy.
A registered value kept in one copy instead (`held`, or a registered parameter named in `place`) is moved to its readers
in other blocks, and those crossings are bandwidth like any other. Every gate of a node is placed with it, so this is
Lean's `Accounting.Placed` (and `Accounting.Requirement.Placement`) at the census's grain: a gate's leaf and step are its
node's. The placement is private and committed. A crossing is a census
value produced in one block and read in another; crossings are derived, never declared, since moving a value doesn't
change it.

What is public is the placement's `Profile`: the occupancy each certifier port must show, the committed bits each block
exports through it, the work each block runs, the ANDs it is credited and the proof blocks they rest on, the trusted
data each block consumes, the registered stores and the registration's root. `audit` and `account` read the profile; only `audit`, which runs inside the proof, reads the
placement itself.
"""
from __future__ import annotations

import weakref
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from verity.ml.circuits.census import NoBooleanWork
from verity.ml.circuits.program import Program

from . import logical as L
from .logical import Registration, Value
from .memory import StoreEntry
from .physical import Datacenter, LinkEntry, Port
from .plan import Plan

#: where a node's bits come from: registration, a trusted device's fresh data (part of the verifier's coins), a device in a
#: block (telemetry, a clock reading), or anywhere else (the outside world: a request)
SOURCES = ("registered", "trusted", "inside", "outside")

_partitions: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _partition(evaluate: Callable, program: Program, query: Mapping[str, Any]) -> Any:
    """`evaluate(program, query)`, once per Program, query and evaluator (a refusal too)."""
    from verity.core.protocols.sampled_proofs.one_stage.registration import query_digest

    memo = _partitions.setdefault(program, {})
    key = (evaluate, query_digest(query))
    if key not in memo:
        try:
            memo[key] = evaluate(program, query)
        except Exception as e:
            memo[key] = e
    if isinstance(memo[key], Exception):
        raise memo[key]
    return memo[key]


@dataclass
class Placed:
    """Where and when each root node runs."""

    program: Program
    datacenter: Datacenter
    device: dict[int, str]
    step: dict[int, int]
    ticks: dict[str, list[int]] = field(default_factory=dict)
    #: the device the Program's outputs are delivered to (usually the outside world), or None
    deliver: str | None = None
    #: registered state, each on the device the registration says holds it
    stores: tuple[StoreEntry, ...] = ()
    #: the registration of the Program's registered inputs
    registration: Registration | None = None
    #: registered values (by census value id) kept in one copy, on this device, rather than in every reading block
    held: dict[int, str] = field(default_factory=dict)

    @cached_property
    def values(self) -> list[Value]:
        return L.values(self.program)

    @cached_property
    def names(self) -> list[str]:
        return L.names(self.program)

    def home(self, v: Value) -> str | None:
        """The one device value `v` is produced on, or held on before the epoch; None for a registered value with a copy
        in every block that reads it."""
        if v.node in self.device:
            return self.device[v.node]
        return self.held.get(v.id)

    def tick(self, k: int) -> int:
        return self.ticks[self.device[k]][self.step[k]]

    def paced(self) -> bool:
        return all(all(a < b for a, b in zip(t, t[1:])) for t in self.ticks.values())

    def block(self, device: str | None) -> str | None:
        b = self.datacenter.block_of(device) if device is not None else None
        return b.name if b else None

    def source(self, k: int) -> str | None:
        """Where input node `k`'s bits come from (`SOURCES`); None for a Call."""
        when = L.binding(self.program, k)
        if when is None:
            return None
        if when == L.REGISTRATION:
            return "registered"
        d = self.device[k]
        return "trusted" if self.datacenter.devices[d].trusted else "inside" if self.block(d) else "outside"

    def gates(self, v: Value) -> list[int]:
        return self._gates.setdefault(v.id, L.gates(self.program, v))

    @cached_property
    def _gates(self) -> dict[int, list[int]]:
        return {}

    def public(self, plan: Plan, ticks: dict[str, float] | None = None, record: Mapping[str, Any] | None = None,
               query: Mapping[str, Any] | None = None) -> Profile:
        """The placement's public properties for the epoch `plan` audits. `ticks` is PoUW's proved bound per block
        (`benchmarks/pouw/proved_ticks.py`): (1 − ε) of the block-ticks the floor program ♭ takes for the block's census
        at the data sheet's rates, which outside δ no accepted window undercuts over any route. `record` is the sampled
        proofs' one-stage registration of the Program and `query` the verifier's copy of its partition query: each
        block's proof blocks are the units of that partition whose gates run there, and the record must name that query,
        its partition and its population (proofs, review 4). Without a record they are the served query's
        (`partition_object.query()`, Q_word v1 at X = 16), which the profile marks as assumed (`Profile.partition`), and
        none for a Program that query doesn't apply to."""
        dc = self.datacenter
        try:
            doc, per_call, uncounted = L.census_of(self.program, boolean=True), L.ands(self.program), None
        except NoBooleanWork as e:
            doc, per_call, uncounted = L.census_of(self.program), {}, str(e)
        work: dict[str, dict[str, float]] = {b.name: {} for b in dc.blocks}
        ands = {b.name: 0 for b in dc.blocks}
        for k, w in L.work(self.program).items():
            if (b := self.block(self.device[k])) is not None:
                ands[b] += per_call.get(k, 0)
                for kind, n in w.items():
                    work[b][kind] = work[b].get(kind, 0.0) + n
        units, partition = self._units(record, query)
        forced: dict[str, float] = {b.name: 0.0 for b in dc.blocks}
        for v in self.values:
            if self.source(v.node) == "trusted":
                for b in {self.block(self.device[r]) for r in v.reads} - {None}:
                    forced[b] += v.bits
        stores = {s.name: s for s in self.stores}
        stores.update({w.store.name: w.store for w in plan.wipes})
        salt, salt_out, pinned = self._custody(plan.registration)
        return Profile(dc, plan, occupancy(self), work, forced, stores, ands, salt, salt_out, pinned, dict(ticks or {}),
                       self.registration.root if self.registration else None, units, doc["digest"], partition,
                       self._exported(plan), uncounted)

    def _exported(self, plan: Plan) -> dict[Port, float]:
        """The committed bits each egress port carries out of its block in the epoch: every crossing from a block, to
        another block or outside, at the first egress port on its path, read within `plan.epoch`. With the
        datacenter's pair links sealed (`Datacenter.sealed`), a crossing to another block is ciphertext between the two
        blocks' certifiers, which only they open, so only a crossing to a reader off every block counts: the export at
        the facility boundary."""
        dc, out = self.datacenter, {}
        for x in crossings(self):
            if self.block(x.src) is None or x.at not in plan.epoch or (dc.sealed and self.block(x.dst) is not None):
                continue
            port = next((q for hop in hops(x.path, x.src) if (q := dc.port(*hop)) and q.direction == "egress"), None)
            if port is not None:
                out[port] = out.get(port, 0.0) + x.value.bits
        return out

    def _units(self, record: Mapping[str, Any] | None, query: Mapping[str, Any] | None) -> tuple[dict[str, int], str | None]:
        """Each block's proof blocks under the registered partition, as core's one-stage verifier evaluates and checks it,
        or the served query's as circuits evaluates it, assumed; and which it was."""
        from verity.core.protocols.sampled_proofs.one_stage import partition as P
        from verity.core.protocols.sampled_proofs.one_stage.registration import query_digest
        from verity.ml.circuits import partition_object as PO

        program = self.program
        if record is None:
            try:
                part = _partition(PO.evaluate, program, PO.query())
            except (PO.QueryRefused, ValueError):
                return {}, None
            how = "assumed"
        else:
            if query is None or query_digest(query) != record["query"]:
                raise ValueError("a registration's partition is read with the verifier's copy of its query")
            part = _partition(P.evaluate, program, query)
            if P.digest(P.partition(program, query)) != record["partition"] or part.population != record["population"]:
                raise ValueError("the registration's partition isn't this Program's under its query")
            how = "registered"
        body = program.root.body
        per_node = ([(k, c.units) for k, c in part.calls] if isinstance(part, PO.Partition)
                    else [(body.node_at(part.gates(u)[0]), 1) for u in range(part.population)])
        units: dict[str, int] = {}
        for k, n in per_node:
            if n and (b := self.block(self.device.get(k))) is not None:
                units[b] = units.get(b, 0) + n
        return units, how

    def _custody(self, registration: int) -> tuple[dict[str, float], ...]:
        """For each block whose own certifier releases a salt into it: the salt's bits, the bits other than timing that
        could carry it out before registration (the values the salt reaches that cross out, and the block's inside inputs
        that reach anything crossing out, both at the census's grain, which can only over-count), and the ANDs of the
        block's work that depends on it, gate by gate within the Calls it reaches (`logical.reach`; none past the
        explicit limit nor on a word gate)."""
        dc = self.datacenter
        readers: dict[int, set[int]] = {}
        for v in self.values:
            readers.setdefault(v.node, set()).update(v.reads)

        def cone(roots: set[int]) -> set[int]:
            seen, todo = set(roots), list(roots)
            while todo:
                for r in readers.get(todo.pop(), ()):
                    if r not in seen:
                        seen.add(r)
                        todo.append(r)
            return seen

        out = [x for x in crossings(self) if x.at < registration and self.block(x.src) != self.block(x.dst)]
        salt, salt_out, pinned = ({b.name: 0.0 for b in dc.blocks} for _ in range(3))
        salts: dict[str, set[int]] = {}
        for k in self.device:
            if self.source(k) == "trusted" and dc.devices[self.device[k]].kind == "certifier":
                salts.setdefault(dc.watched(Port(self.device[k], "ingress")).name, set()).add(k)
        for b, roots in salts.items():
            salt[b] = sum(v.bits for v in self.values if v.node in roots)
            reach = cone(roots)
            salt_out[b] = sum(x.value.bits for x in out if x.value.node in reach and self.block(x.src) == b)
            for k in self.device:
                if self.source(k) == "inside" and self.block(self.device[k]) == b:
                    reach_k = cone({k})
                    if any(x.value.node in reach_k and self.block(x.src) == b for x in out):
                        salt_out[b] += sum(v.bits for v in self.values if v.node == k)
            pinned[b] = sum(n for k, n in L.reach(self.program, frozenset(roots)).items()
                            if self.block(self.device.get(k)) == b)
        return salt, salt_out, pinned


@dataclass(frozen=True)
class Profile:
    """A placement's public properties, with the declaration and the verifier's plan they are read against: the frames
    each certifier port shows in each bucket (`occupancy`), the work each block runs, by kind (`work`, never credited),
    its Calls' ANDs (`ands`, its credit; none, and `uncounted` says why, for a Program the census can't lower to bits),
    the trusted data each block consumes (`forced`, in bits), the registered stores and the registration's root
    (`registered`), and the digest of the census/v1 it is read from (`census`, which names the Program's digest and the
    lowering). For a block whose certifier holds a salt for it: the salt's bits (`salt`), what besides timing could
    carry it out before registration (`salt_out`, in bits), and the ANDs of the block's work that depends on it
    (`pinned`). For a block whose work PoUW timed: its proved bound (`ticks`, in block-ticks), which the account credits
    in place of the per-facet loads, since a prover may route work to another facet than the honest design does. Each
    block's proof blocks (`units`): the units of the Program's partition whose gates run there, which its `Computed` claim
    rests on, and whether that partition was the one-stage registration's or the served query's, assumed (`partition`:
    "registered", "assumed", or None when no partition applies). The export (`exported`): the committed bits each
    egress port carries out of its block in the epoch, to another block or outside, from the declared crossings."""

    datacenter: Datacenter
    plan: Plan
    occupancy: dict[Port, dict[int, int]]
    work: dict[str, dict[str, float]]
    forced: dict[str, float]
    stores: dict[str, StoreEntry]
    ands: dict[str, int] = field(default_factory=dict)
    salt: dict[str, float] = field(default_factory=dict)
    salt_out: dict[str, float] = field(default_factory=dict)
    pinned: dict[str, float] = field(default_factory=dict)
    ticks: dict[str, float] = field(default_factory=dict)
    registered: bytes | None = None
    units: dict[str, int] = field(default_factory=dict)
    census: str | None = None
    partition: str | None = None
    exported: dict[Port, float] = field(default_factory=dict)
    uncounted: str | None = None

    @property
    def windows(self) -> range:
        """The record windows of the epoch: every port must have a record for each."""
        per, e = self.datacenter.cadence.window_ticks, self.plan.epoch
        return range(e.start // per, -(-e.end // per))


def place(program: Program, dc: Datacenter, default: str, *, deliver: str | None = None,
          stores: tuple[StoreEntry, ...] = (), registration: Registration | None = None, start: int = 0,
          ticks_per_step: int = 1, transfer_steps: int = 2, **where: str) -> Placed:
    """Place `program` on `dc`: each keyword names a root node (a parameter or a named Call) and the device it runs or
    enters on; any other Call or run input goes where its first reader runs, if that is in a block, else on `default`.
    A registered input goes nowhere unless named: registration put a copy in every block that reads it. Named, it is
    held in one copy on that device and moved to its readers elsewhere. A node's step is one past those of the nodes it
    reads (plus `transfer_steps` when a value crosses a block), and each device's step s begins at tick
    start + s·ticks_per_step. The outputs are delivered to `deliver`. A Program with registered inputs is placed with
    their `registration`."""
    devices, ns = dc.devices, L.names(program)
    named: dict[int, str] = {}
    unknown = [d for d in [default, deliver, *where.values()] if d is not None and d not in devices]
    for name, d in where.items():
        ks = L.nodes(program, name)
        unknown += [] if ks else [name]
        named.update((k, d) for k in ks)
    if unknown:
        raise ValueError(f"place names what isn't there: {unknown}")
    regd = [n for n, b in L.parameters(program) if b == L.REGISTRATION]
    if regd and (registration is None or set(registration.values) != set(regd)):
        raise ValueError(f"a Program with registered inputs {regd} is placed with their registration")
    vals = L.values(program)
    readers: dict[int, set[int]] = {}
    for v in vals:
        readers.setdefault(v.node, set()).update(v.reads)
    device: dict[int, str] = {}
    for k in reversed(range(len(ns))):
        if k in named:
            device[k] = named[k]
        elif L.binding(program, k) != L.REGISTRATION:
            inside = [device[r] for r in sorted(readers.get(k, ())) if dc.block_of(device[r]) is not None]
            device[k] = inside[0] if inside else default
    p = Placed(program, dc, device, {}, {}, deliver, tuple(stores), registration)
    read: dict[int, list[Value]] = {}
    for v in vals:
        for r in v.reads:
            read.setdefault(r, []).append(v)
    for k in sorted(device):
        p.step[k] = 1 + max((p.step.get(v.node, 0) + (transfer_steps if crosses(p, v, device[k]) else 0)
                             for v in read.get(k, ())), default=-1)
    horizon = max(p.step.values(), default=0) + 1 + transfer_steps
    used = set(device.values()) | {p.home(v) for v in vals} | ({deliver} if deliver else set())
    p.ticks = {d: [start + s * ticks_per_step for s in range(horizon)] for d in used - {None}}
    for s in stores:
        if s.device not in devices:
            raise ValueError(f"store {s.name!r} is registered on an unknown device {s.device!r}")
    return p


@dataclass(frozen=True)
class Crossing:
    """Value `value` leaves `src` (a device of one block) for `dst` (a device of another) along `path`. `at` is the tick
    it is read at, the reader's (Lean's `readOn`): when it must have arrived."""

    value: Value
    src: str
    dst: str
    path: tuple[LinkEntry, ...]
    at: int = 0


def route(dc: Datacenter, src: str, dst: str) -> tuple[LinkEntry, ...]:
    """The shortest path of links from device `src` to device `dst`."""
    adj: dict[str, list[tuple[str, LinkEntry]]] = {}
    for l in dc.links:
        adj.setdefault(l.a, []).append((l.b, l))
        adj.setdefault(l.b, []).append((l.a, l))
    prev: dict[str, tuple[str, LinkEntry] | None] = {src: None}
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


def hops(path: tuple[LinkEntry, ...], src: str) -> list[tuple[str, str]]:
    """The links of `path` as directed hops (from, to), walking from `src`."""
    out, here = [], src
    for l in path:
        there = l.b if l.a == here else l.a
        out.append((here, there))
        here = there
    return out


def crosses(p: Placed, v: Value, dst: str) -> bool:
    """Whether value `v`, read at device `dst`, comes from another block: never for a registered value with a copy in
    every block that reads it."""
    src = p.home(v)
    return src is not None and p.datacenter.block_of(src) is not p.datacenter.block_of(dst)


def crossings(p: Placed) -> list[Crossing]:
    """Every (value, reader device) pair where a value is read in a block other than its own, once each, at its first
    read; and each output the Program returns, at the delivery device the step after it is produced."""
    first: dict[tuple[int, str], int] = {}
    for v in p.values:
        for r in v.reads:
            key = (v.id, p.device[r])
            first[key] = min(first.get(key, p.tick(r)), p.tick(r))
        if v.output and p.deliver:
            ticks = p.ticks[p.deliver]
            first.setdefault((v.id, p.deliver), ticks[min(p.step.get(v.node, 0) + 1, len(ticks) - 1)])
    out = []
    for (i, dst), at in first.items():
        v = p.values[i]
        if crosses(p, v, dst):
            out.append(Crossing(v, p.home(v), dst, route(p.datacenter, p.home(v), dst), at))
    return out


def traffic(p: Placed) -> dict[Port, dict[int, list[Value]]]:
    """The honest run's traffic at each certifier port, bucket by bucket: the crossing values it carries, each in the
    bucket of its read tick."""
    out: dict[Port, dict[int, list[Value]]] = {}
    bucket = p.datacenter.cadence.bucket_ticks
    for x in crossings(p):
        for hop in hops(x.path, x.src):
            if (port := p.datacenter.port(*hop)) is not None:
                out.setdefault(port, {}).setdefault(x.at // bucket, []).append(x.value)
    return out


def frames(values: list[Value], frame_bits: int) -> int:
    """The frames that carry `values`, packed together."""
    return -(-sum(v.bits for v in values) // frame_bits)


def occupancy(p: Placed) -> dict[Port, dict[int, int]]:
    """The public occupancy profile: for each certifier port and each bucket, the frames the honest run shows there."""
    fb = p.datacenter.cadence.frame_bits
    return {port: {b: frames(vs, fb) for b, vs in row.items()} for port, row in traffic(p).items()}
