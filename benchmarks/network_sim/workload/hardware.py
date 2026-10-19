"""The hardware a placement runs on, in the shape of Lean's `Accounting.Hardware` (`Security/Definitions/Accounting/Physical.lean`).

Leaves are devices, numbered in order (`L = Nat`). A *unit* is a block: a set of leaves behind one boundary (`units`).
Links (`E`) each join two leaves (`ends`) with a capacity `beta` in bits a tick each way. A link is certified when one of
its ends is a certifier's leaf (`cert`). `boundary(X)` is the links with exactly one end in X, and X is covered when every
boundary link is certified (`Covered`, `Hardware.SpatialUnit`). `latency(x, y)` is `ℓ`, the ticks a read from leaf x to
leaf y takes at least (`Placed.Causal`), and `bucket(t)` is the bucket of tick t over which a certifier's grid counts.

A leaf in no unit holds no circuit gate unless it is outside: the client that sends requests and reads answers stands at
`OUTSIDE`'s leaf, and certifiers and switches hold only frames. Inside a unit, the leaves in `compute` run gates (every
leaf of the unit when `compute` is empty).

`mock(blocks, devices)` is the toy datacenter: block b has devices `b<b>.gpu<i>`, star-wired over uncertified PCIe links to
`b<b>.gpu0`, whose uplink goes to the block's certifier `b<b>.cert`; each certifier's wire and the client's go to one
switch. Every route between two blocks therefore crosses the source block's uplink and the destination block's uplink.

`Hardware.from_datacenter` reads a declaration of `benchmarks/datacenter` (a `datacenter.Datacenter`, as
`datacenter.examples` declares them and `mockdc.py` stands them up): its blocks are the units and its GPUs compute.
"""
from __future__ import annotations

import dataclasses
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property

OUTSIDE = "outside"


@dataclass(frozen=True)
class Link:
    name: str
    ends: tuple[int, int]
    beta: float  # bits a tick each way; infinite while unmeasured


@dataclass(frozen=True)
class Hardware:
    leaves: tuple[str, ...]
    units: dict[str, frozenset[int]]
    links: tuple[Link, ...]
    cert: frozenset[int]
    bucket_ticks: int = 1
    cross_latency: int = 1
    client: int = 0
    compute: frozenset[int] = frozenset()
    _index: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._index.update({name: i for i, name in enumerate(self.leaves)})
        seen: set[int] = set()
        for u, xs in self.units.items():
            if u == OUTSIDE:
                raise ValueError(f"{OUTSIDE!r} names the complement of the units, not a unit")
            if seen & xs:
                raise ValueError(f"unit {u} shares leaves with another unit")
            seen |= xs
        if self.client in seen or self.client in self.cert:
            raise ValueError("the client's leaf is outside every unit and is no certifier")

    def leaf(self, name: str) -> int:
        return self._index[name]

    @cached_property
    def _unit_of(self) -> dict[int, str]:
        return {x: u for u, xs in self.units.items() for x in xs}

    def unit(self, x: int) -> str:
        """The unit of leaf x, or `OUTSIDE`."""
        return self._unit_of.get(x, OUTSIDE)

    def devices(self, unit: str) -> list[int]:
        """The leaves of a unit that run gates, in order."""
        xs = self.units[unit]
        return sorted(xs & self.compute if self.compute else xs)

    def leaves_of(self, unit: str) -> frozenset[int]:
        """The leaves at which a unit holds values: a block's devices, or the client's leaf for `OUTSIDE`."""
        return frozenset({self.client}) if unit == OUTSIDE else self.units[unit]

    def unit_names(self) -> list[str]:
        """The units in canonical order: the blocks by name, then `OUTSIDE`."""
        return sorted(self.units) + [OUTSIDE]

    def bucket(self, t: int) -> int:
        return t // self.bucket_ticks

    def latency(self, x: int, y: int) -> int:
        """`ℓ x y`: 0 inside a unit, `cross_latency` between units."""
        return 0 if self.unit(x) == self.unit(y) else self.cross_latency

    def certified(self, e: int) -> bool:
        a, b = self.links[e].ends
        return a in self.cert or b in self.cert

    def boundary(self, unit: str) -> list[int]:
        xs = self.units[unit]
        return [e for e, link in enumerate(self.links) if (link.ends[0] in xs) != (link.ends[1] in xs)]

    def covered(self, unit: str) -> bool:
        """`Covered`: every boundary link of the unit is certified, so its uncertified capacity is 0."""
        return all(self.certified(e) for e in self.boundary(unit))

    @cached_property
    def _adjacent(self) -> dict[int, list[tuple[int, int]]]:
        adj: dict[int, list[tuple[int, int]]] = {x: [] for x in range(len(self.leaves))}
        for e, link in enumerate(self.links):
            a, b = link.ends
            adj[a].append((b, e))
            adj[b].append((a, e))
        for x in adj:
            adj[x].sort()
        return adj

    def route(self, src: int, dst: int) -> list[tuple[int, int]]:
        """The links from leaf src to leaf dst, each as (link, the end it delivers to): a shortest path, ties broken by
        the lowest leaf."""
        prev: dict[int, tuple[int, int]] = {src: (src, -1)}
        todo = deque([src])
        while todo:
            x = todo.popleft()
            if x == dst:
                break
            for y, e in self._adjacent[x]:
                if y not in prev:
                    prev[y] = (x, e)
                    todo.append(y)
        if dst not in prev:
            raise ValueError(f"no route from {self.leaves[src]} to {self.leaves[dst]}")
        hops = []
        x = dst
        while x != src:
            p, e = prev[x]
            hops.append((e, x))
            x = p
        return hops[::-1]

    def watched(self, src: int, dst: int) -> list[tuple[int, int]]:
        """The hops of `route(src, dst)` that cross a unit's certified boundary: the links whose frames a certifier
        records and an occupancy profile counts."""
        out = []
        for e, to in self.route(src, dst):
            if not self.certified(e):
                continue
            a, b = self.links[e].ends
            if self.unit(a) != self.unit(b):
                out.append((e, to))
        return out

    def key(self, e: int, to: int) -> str:
        """The name of one direction of a link: `<link>><leaf it delivers to>`."""
        return f"{self.links[e].name}>{self.leaves[to]}"

    def to_json(self) -> dict:
        return {"leaves": list(self.leaves), "units": {u: sorted(xs) for u, xs in sorted(self.units.items())},
                "links": [{"name": k.name, "ends": list(k.ends), "beta": k.beta} for k in self.links],
                "cert": sorted(self.cert), "bucket_ticks": self.bucket_ticks, "cross_latency": self.cross_latency,
                "client": self.client, "compute": sorted(self.compute)}

    @classmethod
    def from_json(cls, j: dict) -> "Hardware":
        return cls(tuple(j["leaves"]), {u: frozenset(xs) for u, xs in j["units"].items()},
                   tuple(Link(k["name"], tuple(k["ends"]), k["beta"]) for k in j["links"]), frozenset(j["cert"]),
                   j["bucket_ticks"], j["cross_latency"], j["client"], frozenset(j.get("compute", ())))

    @classmethod
    def from_datacenter(cls, dc, client: str | None = None) -> "Hardware":
        """A `datacenter.Datacenter`: its devices as the leaves (the blocks', then the fabric's), its blocks as the units,
        its GPUs as `compute`, its certifiers as `cert`, the client at `client` or its device of kind `outside`, and its
        cadence's `bucket_ticks`. Link `a-b` carries its capacity times the cadence's tick, and `cross_latency` is the
        least latency in whole ticks, rounded down, from a unit's compute leaves (or the client) to another's, a link
        with no latency counting 0."""
        devices = dc.devices
        leaves = tuple(devices)
        index = {x: i for i, x in enumerate(leaves)}
        tick = float(dc.cadence.tick)
        if client is None:
            client = next((x for x, d in devices.items() if d.kind == "outside"), None)
            if client is None:
                raise ValueError(f"{dc.name} has no device of kind `outside` for the client")
        hw = cls(leaves, {b.name: frozenset(index[d.name] for d in b.devices) for b in dc.blocks},
                 tuple(Link(f"{k.a}-{k.b}", (index[k.a], index[k.b]), float(k.capacity) * tick) for k in dc.links),
                 frozenset(index[x] for x, d in devices.items() if d.kind == "certifier"), dc.cadence.bucket_ticks, 1,
                 index[client], frozenset(index[x] for x, d in devices.items() if d.kind == "gpu"))
        # The delays are floats in seconds, so a path of whole ticks can come out a hair under its count.
        ell = hw.least_latency_across([k.delay / tick for k in dc.links])
        return dataclasses.replace(hw, cross_latency=math.floor(ell + 1e-9))

    def least_latency_across(self, ell: list[float]) -> float:
        """The least latency, over paths weighted by `ell` (ticks a link), from a unit's compute leaves to another unit's
        or the client's, or from the client to a unit's."""
        readers = {u: set(self.devices(u)) for u in self.units} | {OUTSIDE: {self.client}}
        best = None
        for u, sources in readers.items():
            targets = set().union(*(xs for v, xs in readers.items() if v != u))
            dist = {x: 0 for x in sources}
            todo = [(0, x) for x in sources]
            while todo:
                d, x = heapq.heappop(todo)
                if d > dist[x]:
                    continue
                if x in targets:
                    best = d if best is None else min(best, d)
                    break
                for y, e in self._adjacent[x]:
                    if d + ell[e] < dist.get(y, d + ell[e] + 1):
                        dist[y] = d + ell[e]
                        heapq.heappush(todo, (d + ell[e], y))
        return best or 0


def mock(blocks: int, devices: int = 1, *, nic_bits: int = 400_000, pcie_bits: int = 500_000,
         bucket_ticks: int = 1) -> Hardware:
    if blocks < 1 or devices < 1:
        raise ValueError("a datacenter has at least one block of one device")
    leaves = ["client", "switch"]
    units: dict[str, set[int]] = {}
    links: list[Link] = [Link("client.wire", (0, 1), nic_bits)]
    cert = set()
    for b in range(blocks):
        gpus = []
        for i in range(devices):
            gpus.append(len(leaves))
            leaves.append(f"b{b}.gpu{i}")
        c = len(leaves)
        leaves.append(f"b{b}.cert")
        cert.add(c)
        units[f"b{b}"] = set(gpus)
        links += [Link(f"b{b}.pcie{i}", (gpus[0], gpus[i]), pcie_bits) for i in range(1, devices)]
        links += [Link(f"b{b}.uplink", (gpus[0], c), nic_bits), Link(f"b{b}.wire", (c, 1), nic_bits)]
    return Hardware(tuple(leaves), {u: frozenset(xs) for u, xs in units.items()}, tuple(links), frozenset(cert),
                    bucket_ticks=bucket_ticks)
