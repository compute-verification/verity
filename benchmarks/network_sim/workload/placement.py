"""The private placement of the circuit on the hardware, and the crossing edges it implies.

`StepPlacement` has the shape of Lean's `Accounting.Requirement.Placement` and its Lean-code mirror
(`CircuitRequirement.Placement`): the values are the circuit's gates in canonical order (`Fin n`), each with its engine
step (`step`) and the device leaf at which it is produced or, for an input, stored (`leaf`), and `ticks[x][s]` is the tick
at which device x begins engine step s. A value's tick is `ticks[leaf v][step v]` (`StepPlacement.tick`), and `paced(X)`
is `Placement.Paced`: each device of X begins its steps at strictly increasing ticks. As `Placement.placed` does, a
value's cell is its index, so the placed circuit's `x`, `t` and `cell` are `leaf`, `tick` and the identity, and no two
gates share a cell.

`place` builds one from a plan that puts each root Call on a device at an engine step; the Call's gates go with it.
Registered inputs (the weights) are stored at the device of their first reader (one no gate reads, such as a rotary-table
row of a position the request never reaches, beside the registered input before it), and run inputs (the requests) at the
client's leaf, outside every block. Step 0 is the inputs'; a Call runs at its model's step (`Model.step`: the MLP's
request r's layer l at 1 + r + l), on one of the block's `DeviceGraph.devices`, and every device begins step s at tick
s · `step_ticks`.

**Crossing edges** (`crossings`): a value v read by a gate whose leaf is in another unit, one edge per (v, reading unit),
read at the earliest tick of a reader in that unit. A root output is also an edge, to `OUTSIDE`, read by the client at
its step after the output's. Constants are an operation's own bits (`verity/ml/circuits/PROTOCOL.md` §11): every block has
them, so none crosses. These edges are the transfers: each happens between its value's tick and its read tick.

`commit` registers the placement as hiding rows (`verity.ml.commitments.advice`, hm96-sha512 leaves in a frame-v3-sha512
tree) under `BINDING`, the SHA-512 of its label; the root is public, the rows and salts are the prover's.
"""
from __future__ import annotations

import bisect
import hashlib
import struct
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Sequence

from verity.ml.commitments import advice

from .hardware import OUTSIDE, DeviceGraph
from .model import CallInfo, Model

BINDING = hashlib.sha512(b"verity/netsim/placement/v1").digest()
ROW_BITS = 512
_MAGIC = b"netsim/placement"
_ENTRY = struct.Struct("<HI")


@dataclass(frozen=True)
class StepPlacement:
    step: tuple[int, ...]
    leaf: tuple[int, ...]
    ticks: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.step) != len(self.leaf):
            raise ValueError("one step and one leaf per value")

    @property
    def n(self) -> int:
        return len(self.leaf)

    def tick(self, v: int) -> int:
        return self.ticks[self.leaf[v]][self.step[v]]

    @cached_property
    def tick_of(self) -> tuple[int, ...]:
        return tuple(self.ticks[x][s] for x, s in zip(self.leaf, self.step))

    def at(self, xs: frozenset[int]) -> list[int]:
        """The values placed at leaves xs, in canonical order."""
        return [v for v, x in enumerate(self.leaf) if x in xs]

    def paced(self, xs: Sequence[int]) -> bool:
        return all(all(a < b for a, b in zip(self.ticks[x], self.ticks[x][1:])) for x in xs)

    def rows(self) -> list[bytes]:
        """The placement as `ROW_BITS`-bit rows: a header (magic, value count, leaf count, steps a device), then each
        value's (leaf u16, step u32), ten to a row, then each device's ticks, u32, sixteen to a row; little-endian."""
        width = ROW_BITS // 8
        steps = len(self.ticks[0]) if self.ticks else 0
        if any(len(t) != steps for t in self.ticks):
            raise ValueError("every device has a tick for every step")
        rows = [(_MAGIC + struct.pack("<III", self.n, len(self.ticks), steps)).ljust(width, b"\0")]
        per = width // _ENTRY.size
        for i in range(0, self.n, per):
            rows.append(b"".join(_ENTRY.pack(x, s) for x, s in zip(self.leaf[i:i + per], self.step[i:i + per]))
                        .ljust(width, b"\0"))
        flat = [t for ts in self.ticks for t in ts]
        for i in range(0, len(flat), width // 4):
            rows.append(struct.pack(f"<{len(flat[i:i + width // 4])}I", *flat[i:i + width // 4]).ljust(width, b"\0"))
        return rows

    @classmethod
    def from_rows(cls, rows: Sequence[bytes]) -> "StepPlacement":
        width = ROW_BITS // 8
        if rows[0][:len(_MAGIC)] != _MAGIC:
            raise ValueError("not a placement's rows")
        n, leaves, steps = struct.unpack_from("<III", rows[0], len(_MAGIC))
        per = width // _ENTRY.size
        k = -(-n // per)
        entries = [_ENTRY.unpack_from(r, j * _ENTRY.size) for r in rows[1:1 + k] for j in range(per)][:n]
        flat = [t for r in rows[1 + k:] for t in struct.unpack(f"<{width // 4}I", r)][:leaves * steps]
        return cls(tuple(s for _, s in entries), tuple(x for x, _ in entries),
                   tuple(tuple(flat[x * steps:(x + 1) * steps]) for x in range(leaves)))

    def commit(self, salts: Sequence[bytes] | None = None) -> advice.Committed:
        return advice.commit(self.rows(), ROW_BITS, BINDING, salts)


Plan = Callable[[CallInfo], tuple[int, int]]


def pipeline(model: Model, hw: DeviceGraph) -> Plan:
    """Layer l on block ⌊l·k/L⌋ of k blocks, its groups spread over the block's devices, at the model's step for the Call."""
    blocks = sorted(hw.units)
    layers = model.spec.layers

    def plan(c: CallInfo) -> tuple[int, int]:
        devices = hw.devices(blocks[c.layer * len(blocks) // layers])
        return devices[c.group % len(devices)], model.step(c)

    return plan


def tensor(model: Model, hw: DeviceGraph) -> Plan:
    """Group g of every layer on block g mod k, on the block's device l mod its device count."""
    blocks = sorted(hw.units)

    def plan(c: CallInfo) -> tuple[int, int]:
        devices = hw.devices(blocks[c.group % len(blocks)])
        return devices[c.layer % len(devices)], model.step(c)

    return plan


PLANS = {"pipeline": pipeline, "tensor": tensor}


def place(model: Model, hw: DeviceGraph, plan: Plan, step_ticks: int = 1) -> StepPlacement:
    """Every device begins step s at tick s · `step_ticks`."""
    n = model.n
    leaf = [-1] * n
    step = [0] * n
    for c in model.calls:
        x, s = plan(c)
        if hw.unit(x) == OUTSIDE:
            raise ValueError(f"a Call placed at {hw.leaves[x]}, outside every block")
        for g in range(c.lo, c.hi):
            leaf[g], step[g] = x, s
    kind, ops = model.kind, model.ops
    for g in range(n):
        if kind[g] == "run":
            leaf[g] = hw.client
    for g in range(n):
        if kind[g] != "computed":
            continue
        for a in ops[g]:
            if leaf[a] < 0:
                leaf[a] = leaf[g]
    stored = [g for g in range(n) if kind[g] == "registered" and leaf[g] >= 0]
    for g in range(n):
        if kind[g] == "registered" and leaf[g] < 0 and stored:
            leaf[g] = leaf[stored[max(bisect.bisect_left(stored, g) - 1, 0)]]
    unplaced = [g for g in range(n) if leaf[g] < 0]
    if unplaced:
        raise ValueError(f"{len(unplaced)} values have no device, the first gate {unplaced[0]} ({kind[unplaced[0]]})")
    steps = max(step) + 2
    ticks = tuple(tuple(s * step_ticks for s in range(steps)) for _ in hw.leaves)
    return StepPlacement(tuple(step), tuple(leaf), ticks)


@dataclass(frozen=True, order=True)
class Edge:
    """A crossing value: `value` produced at leaf `src_leaf` and first read in unit `dst` at leaf `dst_leaf`, tick
    `read_tick`."""

    read_tick: int
    src: str
    dst: str
    value: int
    width: int
    src_leaf: int
    dst_leaf: int
    tick: int


def crossings(model: Model, p: StepPlacement, hw: DeviceGraph) -> list[Edge]:
    """The crossing edges, in canonical order: by read tick, source unit, destination unit and value."""
    kind, ops, ticks = model.kind, model.ops, p.tick_of
    unit = [hw.unit(x) for x in p.leaf]
    first: dict[tuple[int, str], tuple[int, int]] = {}
    for g in range(model.n):
        if kind[g] != "computed":
            continue
        for a in ops[g]:
            if kind[a] == "constant" or unit[a] == unit[g]:
                continue
            key = (a, unit[g])
            if key not in first or (ticks[g], p.leaf[g]) < first[key]:
                first[key] = (ticks[g], p.leaf[g])
    width = model.width
    edges = [Edge(t, unit[a], u, a, width[a], p.leaf[a], x, ticks[a]) for (a, u), (t, x) in first.items()]
    for o in sorted(set(model.outputs)):
        if unit[o] != OUTSIDE:
            edges.append(Edge(p.ticks[hw.client][p.step[o] + 1], unit[o], OUTSIDE, o, width[o], p.leaf[o], hw.client,
                              ticks[o]))
    return sorted(edges)


def causal(model: Model, p: StepPlacement, hw: DeviceGraph) -> list[tuple[int, int]]:
    """The reads that break `Placed.Causal` at the hardware's latencies, as (reader, value); empty when causal."""
    bad = []
    ticks = p.tick_of
    for g in range(model.n):
        if model.kind[g] != "computed":
            continue
        for a in model.ops[g]:
            if ticks[a] + hw.latency(p.leaf[a], p.leaf[g]) > ticks[g]:
                bad.append((g, a))
    return bad
