"""The device graph: one physical model of a datacenter that compute, memory and network accounting all read, and the
plans that stand it up as a mock datacenter.

Its fields are the Lean model's, under their names (β is `beta`, ℓ is `ell`): `Hardware` is `Accounting.Hardware`
(`Security/Definitions/Accounting/Physical.lean`) and `Device` is `Accounting.Device` (`Device.lean`). Time is in ticks
and sizes in bits, as there. `tick_ns` says what a tick is, 1 µs by default, the tick at which every rate stated here is
a whole number of bits or ops; a tick at which one isn't is refused.

- **Leaves** (`L`): the devices. A GPU, its HBM, a socket's CPU, host DRAM and PCIe root complex, a NIC, a switch, a
  network certifier, or a host outside every block. A GPU's and a memory's capacities are on its leaf.
- **Links** (`E`): each joins two leaves (`ends`) with capacity `beta` (bits a tick each way) and latency `ell` (ticks).
  `Hardware.ell(x, y)` is ℓ, the least latency over paths. A link is certified when a certifier sits at one of its ends
  (`cert`).
- **Locations** (`units`): the tree of locations, by name. A **block** is the location of the leaves that name it: every
  link across its boundary must be certified (`Hardware.covered`, Lean's `Covered`), so it is a spatial unit
  (`spatial_unit`). Here each node is a block, with its sockets and its GPUs (each with its HBM, the device of
  `Device.lean`) as locations inside it.
- `compute(X)` and `M(X)` sum a location's devices: Φ_X at a price (`Device.priced`) and its memories' bits.

Every capacity is an `Assumption`: a name, a value (None while a measurement is pending), a unit and a source
(datasheet, measured or declared). The same name always carries the same record. Beyond the Lean fields, each link has
a kind (memory, pcie, socket, nic_uplink, fabric) and a delay jitter, each memory a measured read rate beside its link's
datasheet bound, and each GPU its codec's squaring rate per width (remote bandwidth's assumption 5).

`leaf_spine` generates nodes as vy-nebius-1 and -2 are, behind a leaf-spine fabric; `Hardware.validate` lists what is
wrong. The exports collapse each block to one node: `netns_plan`, the mock datacenter's Linux network namespaces, and
`shadow_gml`, Shadow's network graph.

    python graph.py --blocks 2 (--json | --netns | --gml)
"""

from __future__ import annotations

import argparse
import heapq
import ipaddress
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from fractions import Fraction

SOURCES = ("datasheet", "measured", "declared")
KINDS = ("gpu", "hbm", "host_dram", "cpu", "nic", "switch", "certifier", "host")
MEMORIES = ("hbm", "host_dram")
LINK_KINDS = ("memory", "pcie", "socket", "nic_uplink", "fabric")
OPS = ("tc", "cuda", "oracle")  # `Accounting.Op`
BITS, PER_TICK, TICKS = "bits", "bits/tick", "ticks"
NETNS_PREFIX = "na-"


@dataclass(frozen=True)
class Assumption:
    """A named physical assumption: one capacity, its unit, and where its value comes from. `value` is None while a
    measurement is pending."""

    name: str
    value: int | None
    unit: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class Device:
    """`Accounting.Device`: on-chip state `chip` and device memory `mem` (bits), ops of each class a tick (`rate`),
    bits a tick from device memory to the chip (`bw`), and bits a tick over each side channel (`cap`, by link)."""

    chip: Assumption
    mem: Assumption
    rate: Mapping[str, Assumption]
    bw: Assumption
    cap: Mapping[str, Assumption]

    def priced(self, price: Mapping[str, int]) -> int | None:
        """`Device.priced`: the priced work it can do in a tick, or None if a priced rate is pending."""
        rates = [self.rate[o].value for o in price if price[o]]
        return None if None in rates else sum(price[o] * self.rate[o].value for o in price if price[o])


@dataclass(frozen=True)
class Leaf:
    """A leaf of the tree of locations: one device. `block` names the block it is in, None outside every block."""

    id: str
    kind: str
    block: str | None
    chip: Assumption | None = None  # bits of on-chip state (a GPU's S_chip, a socket's caches)
    M: Assumption | None = None  # a memory's bits
    read: Assumption | None = None  # a memory's measured read rate, bits a tick
    rate: Mapping[str, Assumption] = field(default_factory=dict)  # a GPU's ops a tick, by `OPS` class
    squarings: Mapping[str, Assumption] = field(default_factory=dict)  # a GPU's codec squarings a tick, by width


@dataclass(frozen=True)
class Link:
    id: str
    ends: tuple[str, str]
    kind: str
    beta: Assumption  # bits a tick, each way
    ell: Assumption  # latency, ticks
    delay_jitter: Assumption | None = None  # the emulator's per-packet delay jitter, ticks (not Lean's `jitter`)


@dataclass(frozen=True)
class Hardware:
    """`Accounting.Hardware` on named leaves and links. `bucket_ticks` is `bucket` (tick t is in bucket
    t // bucket_ticks); `jitter` is each certifier's jitter bits a bucket (0: one class, D = 1, as NET-7 assumes);
    `record` each certifier's record cells, empty until a circuit is placed; `ring`, `D` and `span` are
    `SequentialSpeed`'s."""

    leaves: Mapping[str, Leaf]
    links: Mapping[str, Link]
    units: Mapping[str, frozenset[str]]
    cert: frozenset[str]
    trusted: frozenset[str]
    bucket_ticks: int
    jitter: int = 0
    record: Mapping[str, frozenset[int]] = field(default_factory=dict)
    ring: frozenset[str] = frozenset()
    D: int = 0
    span: int = 0
    tick_ns: int = 1_000

    def blocks(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for leaf in self.leaves.values():
            if leaf.block is not None:
                out.setdefault(leaf.block, set()).add(leaf.id)
        return {b: frozenset(xs) for b, xs in out.items()}

    def at(self, leaf: str) -> list[Link]:
        return [e for e in self.links.values() if leaf in e.ends]

    def boundary(self, X: frozenset[str]) -> list[Link]:
        """The links across X's boundary: one end in X, the other outside."""
        return [e for e in self.links.values() if (e.ends[0] in X) != (e.ends[1] in X)]

    def certified(self, e: Link) -> bool:
        return e.ends[0] in self.cert or e.ends[1] in self.cert

    def link_capacity(self, X: frozenset[str]) -> int | None:
        """β_X, or None if a boundary link's capacity is pending."""
        return _total(e.beta for e in self.boundary(X))

    def uncertified_capacity(self, X: frozenset[str]) -> int | None:
        """β°_X: the capacity of X's boundary links with no end at a certifier, or None if one is pending."""
        return _total(e.beta for e in self.boundary(X) if not self.certified(e))

    def covered(self, X: frozenset[str]) -> bool:
        """`Covered`: β°_X = 0, so nothing leaves X but through a certifier."""
        return self.uncertified_capacity(X) == 0

    def spatial_unit(self, name: str) -> bool:
        return name in self.units and self.covered(self.units[name])

    def device(self, gpu: str) -> Device:
        """The GPU with its HBM as `Device.lean`'s device: its HBM's bits and link, and every other link as a side
        channel."""
        leaf = self.leaves[gpu]
        memory = [e for e in self.at(gpu) if e.kind == "memory"]
        if leaf.kind != "gpu" or len(memory) != 1:
            raise ValueError(f"{gpu} is not a GPU with one memory link")
        hbm = self.leaves[_other(memory[0], gpu)]
        return Device(leaf.chip, hbm.M, leaf.rate, memory[0].beta,
                      {e.id: e.beta for e in self.at(gpu) if e is not memory[0]})

    def compute(self, X: frozenset[str], price: Mapping[str, int] | None = None) -> int | None:
        """Φ_X: the priced work of X's GPUs a tick, at `price` (one per tensor-core multiply-add by default)."""
        return _total_values(self.device(x).priced(price or {"tc": 1}) for x in sorted(X)
                             if self.leaves[x].kind == "gpu")

    def M(self, X: frozenset[str]) -> int | None:
        """M_X: the bits of X's memories, or None if one is pending."""
        return _total(self.leaves[x].M for x in sorted(X) if self.leaves[x].kind in MEMORIES)

    def ell(self, x: str, y: str) -> int | None:
        """ℓ(x, y): the least latency over paths from x to y, a pending latency counted as 0 so that it stays a lower
        bound; None if there is no path."""
        best, todo = {x: 0}, [(0, x)]
        while todo:
            d, u = heapq.heappop(todo)
            if u == y:
                return d
            if d > best[u]:
                continue
            for e in self.at(u):
                v, nd = _other(e, u), d + (e.ell.value or 0)
                if nd < best.get(v, nd + 1):
                    best[v] = nd
                    heapq.heappush(todo, (nd, v))
        return None

    def assumptions(self) -> list[tuple[str, Assumption, str]]:
        """Every capacity as (where, assumption, the unit its field takes)."""
        out = []
        for leaf in self.leaves.values():
            for name, a, unit in (("chip", leaf.chip, BITS), ("M", leaf.M, BITS), ("read", leaf.read, PER_TICK)):
                if a is not None:
                    out.append((f"{leaf.id}.{name}", a, unit))
            out += [(f"{leaf.id}.rate.{o}", a, "ops/tick") for o, a in leaf.rate.items()]
            out += [(f"{leaf.id}.squarings.{w}", a, "squarings/tick") for w, a in leaf.squarings.items()]
        for e in self.links.values():
            out += [(f"{e.id}.beta", e.beta, PER_TICK), (f"{e.id}.ell", e.ell, TICKS)]
            if e.delay_jitter is not None:
                out.append((f"{e.id}.delay_jitter", e.delay_jitter, TICKS))
        return out

    def pending(self) -> list[str]:
        return sorted({a.name for _, a, _ in self.assumptions() if a.value is None})

    def validate(self) -> list[str]:
        """What is wrong, one line each: an empty list when the graph is well formed, every capacity is named, and
        every block is covered."""
        errors = []
        for leaf_id, leaf in self.leaves.items():
            if leaf_id != leaf.id or leaf.kind not in KINDS:
                errors.append(f"leaf {leaf_id}: id {leaf.id!r}, kind {leaf.kind!r} (one of {KINDS})")
        for e_id, e in self.links.items():
            if e_id != e.id or e.kind not in LINK_KINDS:
                errors.append(f"link {e_id}: id {e.id!r}, kind {e.kind!r} (one of {LINK_KINDS})")
            if any(x not in self.leaves for x in e.ends) or e.ends[0] == e.ends[1]:
                errors.append(f"link {e_id}: ends {e.ends} are not two leaves")
        errors += self._validate_assumptions()
        errors += self._validate_devices()
        errors += self._validate_locations()
        for b, X in sorted(self.blocks().items()):
            for e in self.boundary(X):
                if not self.certified(e) and e.beta.value != 0:
                    errors.append(f"block {b}: boundary link {e.id} has no certifier at either end (coverage)")
        return errors

    def check(self) -> Hardware:
        if errors := self.validate():
            raise ValueError("the device graph is invalid:\n" + "\n".join(errors))
        return self

    def _validate_assumptions(self) -> list[str]:
        errors, seen = [], {}
        for where, a, unit in self.assumptions():
            if not isinstance(a, Assumption) or not a.name:
                errors.append(f"{where}: a capacity must be a named Assumption, not {a!r}")
                continue
            if a.source not in SOURCES:
                errors.append(f"{where}: {a.name}'s source {a.source!r} is not one of {SOURCES}")
            if a.unit != unit:
                errors.append(f"{where}: {a.name} is in {a.unit!r}, where the field takes {unit!r}")
            if a.value is not None and (not isinstance(a.value, int) or a.value < 0):
                errors.append(f"{where}: {a.name}'s value {a.value!r} is not a whole number (or None, pending)")
            if seen.setdefault(a.name, a) != a:
                errors.append(f"{where}: {a.name} is stated twice, as {seen[a.name]} and as {a}")
        return errors

    def _validate_devices(self) -> list[str]:
        errors = []
        for leaf in self.leaves.values():
            if leaf.kind == "gpu":
                if leaf.chip is None or set(leaf.rate) != set(OPS):
                    errors.append(f"GPU {leaf.id}: a GPU states its chip and its rate for each of {OPS}")
                try:
                    hbm = self.device(leaf.id).mem
                except ValueError as err:
                    errors.append(str(err))
                else:
                    if hbm is None:
                        errors.append(f"GPU {leaf.id}: its memory link doesn't reach a memory with M")
            if leaf.kind in MEMORIES and leaf.M is None:
                errors.append(f"memory {leaf.id}: a memory states its M")
        return errors

    def _validate_locations(self) -> list[str]:
        errors = []
        names = sorted(self.units)
        for name in names:
            if not self.units[name] or not self.units[name] <= self.leaves.keys():
                errors.append(f"location {name}: not a nonempty set of leaves")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                A, B = self.units[a], self.units[b]
                if A & B and not (A <= B or B <= A):
                    errors.append(f"locations {a} and {b} overlap without nesting: the locations form a tree")
        for b, X in sorted(self.blocks().items()):
            if self.units.get(b) != X:
                errors.append(f"block {b}: its location must be exactly the leaves that name it")
            if X & self.trusted:
                errors.append(f"block {b}: holds trusted leaves {sorted(X & self.trusted)}, which run off the prover's"
                              " machines")
        if not self.cert <= self.trusted:
            errors.append(f"certifiers {sorted(self.cert - self.trusted)} are not trusted (cert ⊆ trusted)")
        for c in sorted(self.cert):
            if c not in self.leaves or self.leaves[c].kind != "certifier":
                errors.append(f"cert {c}: not a certifier leaf")
        return errors

    def to_dict(self) -> dict:
        def plain(x):
            if isinstance(x, (set, frozenset)):
                return sorted(x)
            if isinstance(x, Mapping):
                return {k: plain(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [plain(v) for v in x]
            return x

        return plain(asdict(self))


def _other(e: Link, x: str) -> str:
    return e.ends[1] if e.ends[0] == x else e.ends[0]


def _total_values(values) -> int | None:
    values = list(values)
    return None if None in values else sum(values)


def _total(assumptions) -> int | None:
    return _total_values(a.value for a in assumptions)


# The generator. Each rate is stated per second in the unit its source gives, and `per_tick` makes it a whole number a
# tick or refuses the tick.

def per_tick(per_second: int | Fraction, tick_ns: int, scale: int = 1) -> int:
    """`per_second` × `scale` (8 for bytes to bits) in one tick of `tick_ns`, a whole number or ValueError."""
    v = Fraction(per_second) * scale * tick_ns / 1_000_000_000
    if v.denominator != 1:
        raise ValueError(f"{per_second}/s is {float(v)} a tick at a tick of {tick_ns} ns: pick a tick at which every "
                         "rate is a whole number (1,000 ns is)")
    return int(v)


def _gpu(tick_ns: int) -> dict:
    """One RTX PRO 6000 Blackwell Server Edition, as remote bandwidth's `timing` and the census state it."""
    t = tick_ns
    return {
        "chip": Assumption("rtx-pro-6000/S_chip", 263_163_904 * 8, BITS, "measured",
                           "ra-pcie-exhaust #1852, chip.cu (timing.GPU_CHIP_BYTES)"),
        "rate": {
            "tc": Assumption("rtx-pro-6000-server/e4m3", per_tick(500_000_000_000_000, t), "ops/tick", "datasheet",
                             "dense FP8 tensor-core multiply-adds at FP32 accumulate, half the census's 1 PFLOPS"),
            "cuda": Assumption("rtx-pro-6000/cuda-ops", None, "ops/tick", "measured", "not measured yet"),
            "oracle": Assumption("rtx-pro-6000/sha512", per_tick(7_000_000_000, t), "ops/tick", "measured",
                                 "SHA-512 compressions, #1916 (timing.GPU_SHA512_PER_S)"),
        },
        "squarings": {
            "p1024": Assumption("rtx-pro-6000/squarings-p1024", per_tick(14_990_000_000, t), "squarings/tick",
                                "measured", "kv.cu at 1,024 bits; timing's ceiling is 1.597 x this with the host's share"),
            "p512": Assumption("rtx-pro-6000/squarings-p512", per_tick(51_300_000_000, t), "squarings/tick",
                               "measured", "codec-mix at 512 bits; timing's ceiling is 1.567 x this"),
        },
        "hbm_M": Assumption("rtx-pro-6000/memory", 103_079_215_104 * 8, BITS, "datasheet", "96 GB GDDR7"),
        "hbm_beta": Assumption("rtx-pro-6000/memory-bandwidth", per_tick(1_597_000_000_000, t, 8), PER_TICK,
                               "datasheet", "1,597 GB/s (timing.GPU_SPEC_BYTES_PER_S)"),
        "hbm_read": Assumption("rtx-pro-6000/memory-read", per_tick(1_539_000_000_000, t, 8), PER_TICK, "measured",
                               "the fastest bare read, r20261014-193057-8fec (timing.GPU_READ_BYTES_PER_S)"),
    }


def _host(tick_ns: int) -> dict:
    """One socket of the host's two Xeon 6776P, each with 8 of its 16 DDR5-6400 channels."""
    t = tick_ns
    return {
        "chip": Assumption("xeon-6776p/caches", 940_435_456 // 2 * 8, BITS, "declared",
                           "an estimate, half the host's (timing.HOST_CHIP_BYTES)"),
        "dram_M": Assumption("host-dram/socket-capacity", None, BITS, "measured", "not read yet"),
        "dram_beta": Assumption("host-dram/socket-bandwidth", per_tick(409_600_000_000, t, 8), PER_TICK, "datasheet",
                                "8 channels of DDR5-6400, half of timing.HOST_SPEC_BYTES_PER_S"),
        "dram_read": Assumption("host-dram/socket-read", per_tick(303_400_000_000, t, 8), PER_TICK, "measured",
                                "the host's 606.8 GB/s split evenly over its sockets, r20261015-081037-5ecf "
                                "(timing.HOST_READ_BYTES_PER_S)"),
    }


def _pending(name: str, unit: str, note: str) -> Assumption:
    return Assumption(name, None, unit, "measured", note)


PCIE_GPU = _pending("pcie/gpu", PER_TICK, "one GPU's host-to-device rate over its own root port, under gpu-lease")
PCIE_SOCKET = _pending("pcie/socket-aggregate", PER_TICK,
                       "4 GPUs of one socket at once; below 4 x pcie/gpu means a switch the hypervisor hides")
PCIE_NIC = _pending("pcie/nic", PER_TICK, "the NIC's PCIe link")
SOCKET_LINK = _pending("socket/inter-socket", PER_TICK, "the inter-socket link")
NODE_LATENCY = _pending("node/latency", TICKS, "a link inside a node")


def leaf_spine(blocks: int, *, leaves: int = 1, spines: int = 0, outside: int = 1, uplink_gbps: int = 400,
               spine_gbps: int = 400, fabric_ell_ns: int = 2_000, fabric_jitter_ns: int = 0,
               uplink_ell_ns: int = 1_000, tick_ns: int = 1_000, bucket_ms: int = 100) -> Hardware:
    """`blocks` nodes behind a leaf-spine fabric, each a block with its certifier on its uplink.

    A node is as vy-nebius-1 and -2 read with `nvidia-smi topo -m` and `lspci -tv` (18 Oct): 8 GPUs, each with its HBM
    and its own root port, GPUs 0–3 on socket 0 and 4–7 on socket 1, host DRAM on each socket, and one NIC on socket 1.
    A socket's root complex is a switch leaf: each GPU's link to it is the per-GPU rate, and its link to the CPU the
    per-socket aggregate, both pending measurement. The NIC's uplink runs to the block's certifier, outside the block,
    and the certifier's to a leaf switch (blocks spread round-robin over `leaves`), every leaf to every spine, and
    `outside` hosts outside every block hang off the leaves too. With one leaf, `spines` may be 0."""
    if blocks < 1 or leaves < 1 or (leaves > 1 and spines < 1):
        raise ValueError("at least one block and one leaf, and spines to join more than one leaf")
    gpu, host = _gpu(tick_ns), _host(tick_ns)
    up = Assumption("fabric/uplink", per_tick(uplink_gbps * 1_000_000_000, tick_ns), PER_TICK, "declared",
                    f"{uplink_gbps} Gb/s")
    trunk = Assumption("fabric/spine", per_tick(spine_gbps * 1_000_000_000, tick_ns), PER_TICK, "declared",
                       f"{spine_gbps} Gb/s")
    if any(ns % tick_ns for ns in (fabric_ell_ns, uplink_ell_ns, fabric_jitter_ns, bucket_ms * 1_000_000)):
        raise ValueError(f"latencies, jitter and the bucket are whole ticks of {tick_ns} ns")
    hop = Assumption("fabric/latency", fabric_ell_ns // tick_ns, TICKS, "declared", f"{fabric_ell_ns} ns a hop")
    wire = Assumption("fabric/uplink-latency", uplink_ell_ns // tick_ns, TICKS, "declared",
                      f"{uplink_ell_ns} ns from the NIC through its certifier")
    jit = Assumption("fabric/jitter", fabric_jitter_ns // tick_ns, TICKS, "declared", f"{fabric_jitter_ns} ns")

    L: dict[str, Leaf] = {}
    E: dict[str, Link] = {}
    units: dict[str, frozenset[str]] = {}

    def leaf(id_: str, kind: str, block: str | None, **caps) -> str:
        L[id_] = Leaf(id_, kind, block, **caps)
        return id_

    def link(id_: str, a: str, b: str, kind: str, beta: Assumption, ell: Assumption, jitter=None) -> None:
        E[id_] = Link(id_, (a, b), kind, beta, ell, jitter)

    switches = [leaf(f"leaf{j}", "switch", None) for j in range(leaves)]
    for i in range(blocks):
        b = f"b{i}"
        sockets = []
        for s in range(2):
            cpu = leaf(f"{b}.cpu{s}", "cpu", b, chip=host["chip"])
            dram = leaf(f"{b}.dram{s}", "host_dram", b, M=host["dram_M"], read=host["dram_read"])
            rc = leaf(f"{b}.pcie{s}", "switch", b)
            link(f"{dram}.bus", cpu, dram, "memory", host["dram_beta"], NODE_LATENCY)
            link(f"{rc}.up", rc, cpu, "pcie", PCIE_SOCKET, NODE_LATENCY)
            members = {cpu, dram, rc}
            for k in range(4 * s, 4 * s + 4):
                g = leaf(f"{b}.gpu{k}", "gpu", b, chip=gpu["chip"], rate=gpu["rate"], squarings=gpu["squarings"])
                h = leaf(f"{b}.hbm{k}", "hbm", b, M=gpu["hbm_M"], read=gpu["hbm_read"])
                link(f"{h}.bus", g, h, "memory", gpu["hbm_beta"], NODE_LATENCY)
                link(f"{g}.pcie", g, rc, "pcie", PCIE_GPU, NODE_LATENCY)
                units[f"{b}.dev{k}"] = frozenset({g, h})
                members |= {g, h}
            sockets.append(members)
        nic = leaf(f"{b}.nic0", "nic", b)
        link(f"{nic}.pcie", nic, f"{b}.pcie1", "pcie", PCIE_NIC, NODE_LATENCY)
        sockets[1].add(nic)
        link(f"{b}.socket", f"{b}.cpu0", f"{b}.cpu1", "socket", SOCKET_LINK, NODE_LATENCY)
        units[f"{b}.s0"], units[f"{b}.s1"] = frozenset(sockets[0]), frozenset(sockets[1])
        units[b] = frozenset(sockets[0] | sockets[1])
        cert = leaf(f"{b}.cert", "certifier", None)
        link(f"{b}.uplink", nic, cert, "nic_uplink", up, wire)
        link(f"{cert}.fabric", cert, switches[i % leaves], "fabric", up, hop, jit)
    for k in range(spines):
        sp = leaf(f"spine{k}", "switch", None)
        for sw in switches:
            link(f"{sw}.{sp}", sw, sp, "fabric", trunk, hop, jit)
    for c in range(outside):
        x = leaf(f"ext{c}", "host", None)
        link(f"{x}.fabric", x, switches[c % leaves], "fabric", up, hop, jit)
    cert = frozenset(x for x, lf in L.items() if lf.kind == "certifier")
    return Hardware(L, E, units, cert, cert, bucket_ms * 1_000_000 // tick_ns, tick_ns=tick_ns)


# The exports. Each block collapses to one node, standing in for the whole node; every other leaf is its own node.

def network(hw: Hardware) -> tuple[dict[str, str], list[tuple[Link, str, str]]]:
    """The network the exports stand up: each node's role (block, certifier, switch or host) and the links between
    two nodes, with the node at each end."""
    def node(x: str) -> str:
        return hw.leaves[x].block or x

    roles = {b: "block" for b in hw.blocks()}
    for x, lf in hw.leaves.items():
        if lf.block is None:
            roles[x] = {"certifier": "certifier", "switch": "switch", "host": "host"}.get(lf.kind, lf.kind)
    links = [(e, node(e.ends[0]), node(e.ends[1])) for e in hw.links.values() if node(e.ends[0]) != node(e.ends[1])]
    return roles, links


def netns_name(node: str) -> str:
    return NETNS_PREFIX + node.replace(".", "-")


@dataclass(frozen=True)
class Interface:
    name: str
    address: str  # with its prefix length
    link: str
    peer: tuple[str, str]  # (namespace, interface)
    netem: tuple[str, ...] = ()  # `tc qdisc add dev NAME root netem` arguments; none when empty


@dataclass(frozen=True)
class Namespace:
    name: str
    node: str
    role: str
    forward: bool  # IP forwarding: on in switches only, so a block reaches nothing past its certifier but through it
    interfaces: tuple[Interface, ...]
    routes: tuple[tuple[str, str], ...]  # (destination, via)

    def interface(self, link: str) -> Interface:
        return next(i for i in self.interfaces if i.link == link)


@dataclass(frozen=True)
class NetnsPlan:
    """The mock datacenter as Linux network namespaces on one host, one per node, named `na-<node>`: veth pairs for
    links, created straight into their namespaces so that the root namespace never holds one; static routes and IP
    forwarding in the switches only; netem on fabric links, at both ends, for their latency, jitter and rate. A block's
    only interface runs to its certifier, whose namespace forwards nothing: past it the block reaches the fabric only
    through the certifier's own connections. The certifier also drops whatever arrives from its block that isn't part
    of a connection it opened (`guard`)."""

    namespaces: tuple[Namespace, ...]

    def namespace(self, node: str) -> Namespace:
        return next(n for n in self.namespaces if n.node == node)

    def address(self, node: str, link: str) -> str:
        return self.namespace(node).interface(link).address.split("/")[0]

    def up(self) -> list[list[str]]:
        """The commands that stand it up, each an argv to run as root, in order."""
        out = [["ip", "netns", "add", n.name] for n in self.namespaces]
        out += [["ip", "-n", n.name, "link", "set", "lo", "up"] for n in self.namespaces]
        done = set()
        for n in self.namespaces:
            for i in n.interfaces:
                if (n.name, i.name) not in done:
                    out.append(["ip", "link", "add", "name", i.name, "netns", n.name, "type", "veth", "peer", "name",
                                i.peer[1], "netns", i.peer[0]])
                    done |= {(n.name, i.name), i.peer}
        for n in self.namespaces:
            exec_ = ["ip", "netns", "exec", n.name]
            out.append([*exec_, "sysctl", "-qw", f"net.ipv4.ip_forward={int(n.forward)}"])
            for i in n.interfaces:
                out.append(["ip", "-n", n.name, "addr", "add", i.address, "dev", i.name])
                out.append(["ip", "-n", n.name, "link", "set", i.name, "up"])
                if i.netem:
                    out.append([*exec_, "tc", "qdisc", "add", "dev", i.name, "root", "netem", *i.netem])
            out += [["ip", "-n", n.name, "route", "add", dst, "via", via] for dst, via in n.routes]
            out += self.guard(n)
        return out

    def guard(self, n: Namespace) -> list[list[str]]:
        if n.role != "certifier":
            return []
        out = []
        for i in n.interfaces:
            if self.namespace_of(i.peer[0]).role == "block":
                out.append(["ip", "netns", "exec", n.name, "iptables", "-A", "INPUT", "-i", i.name, "-m", "conntrack",
                            "!", "--ctstate", "ESTABLISHED,RELATED", "-j", "DROP"])
        return out

    def namespace_of(self, name: str) -> Namespace:
        return next(n for n in self.namespaces if n.name == name)

    def down(self) -> list[list[str]]:
        """Deleting a namespace deletes its veths with it."""
        return [["ip", "netns", "del", n.name] for n in self.namespaces]


def _netem(hw: Hardware, e: Link) -> tuple[str, ...]:
    if e.kind != "fabric":
        return ()
    out: list[str] = []
    if e.ell.value is not None:
        out += ["delay", f"{e.ell.value * hw.tick_ns / 1000:g}us"]
        if e.delay_jitter is not None and e.delay_jitter.value:
            out.append(f"{e.delay_jitter.value * hw.tick_ns / 1000:g}us")
    if e.beta.value is not None:
        out += ["rate", f"{e.beta.value * 1_000_000_000 // hw.tick_ns}bit"]
    return tuple(out)


def netns_plan(hw: Hardware, base: str = "10.77.0.0") -> NetnsPlan:
    """The netns plan of `hw`: each link between two nodes a /30 from `base` in link order, its first end's address
    the lower. A switch routes every other switch-reachable subnet via the first hop of a shortest path through
    switches; a certifier and an outside host route by default via their switch, and a block via its certifier."""
    hw.check()
    roles, links = network(hw)
    start = ipaddress.IPv4Address(base)
    ifaces: dict[str, list[tuple[str, str, str, str, Link]]] = {x: [] for x in roles}  # name, address, peer node, peer
    for n, (e, a, b) in enumerate(links):
        ia, ib = f"eth{len(ifaces[a]) + 1}", f"eth{len(ifaces[b]) + 1}"
        net = start + 4 * n
        ifaces[a].append((ia, f"{net + 1}/30", b, ib, e))
        ifaces[b].append((ib, f"{net + 2}/30", a, ia, e))
    addr = {(x, e.id): address.split("/")[0] for x, xs in ifaces.items() for _, address, _, _, e in xs}
    nbrs = {x: [(peer, e) for _, _, peer, _, e in xs] for x, xs in ifaces.items()}
    subnets = {e.id: f"{start + 4 * n}/30" for n, (e, _, _) in enumerate(links)}
    switches = sorted(x for x, r in roles.items() if r == "switch")

    def first_hops(src: str) -> dict[str, tuple[str, str]]:
        """For each switch reachable from `src` through switches, the neighbour and link of a first hop to it."""
        out, frontier = {src: None}, [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v, e in nbrs[u]:
                    if roles[v] == "switch" and v not in out:
                        out[v] = out[u] or (v, e.id)
                        nxt.append(v)
            frontier = nxt
        return {k: v for k, v in out.items() if v is not None}

    namespaces = []
    for x in roles:
        routes: list[tuple[str, str]] = []
        if roles[x] == "switch":
            hops = first_hops(x)
            for s, (v, via_link) in sorted(hops.items()):
                for _, e in nbrs[s]:
                    if (x, e.id) not in addr:
                        dst = subnets[e.id]
                        if dst not in (d for d, _ in routes):
                            routes.append((dst, addr[(v, via_link)]))
        else:
            ups = [(v, e) for v, e in nbrs[x] if roles[v] == ("certifier" if roles[x] == "block" else "switch")]
            if ups:
                routes.append(("default", addr[(ups[0][0], ups[0][1].id)]))
        interfaces = tuple(Interface(name, address, e.id, (netns_name(peer), pname), _netem(hw, e))
                           for name, address, peer, pname, e in ifaces[x])
        namespaces.append(Namespace(netns_name(x), x, roles[x], roles[x] == "switch", interfaces, tuple(routes)))
    return NetnsPlan(tuple(namespaces))


def shadow_gml(hw: Hardware) -> str:
    """Shadow's network graph (GML): a node per network node, its host bandwidth its links' capacity each way, and an
    edge per link with its latency and jitter. Shadow limits bandwidth per host, not per edge."""
    hw.check()
    roles, links = network(hw)
    ids = {x: i for i, x in enumerate(roles)}
    per_s = 1_000_000_000 // hw.tick_ns if 1_000_000_000 % hw.tick_ns == 0 else None

    def bandwidth(x: str) -> int | None:
        caps = [e.beta.value for e, a, b in links if x in (a, b)]
        return None if None in caps or per_s is None else sum(caps) * per_s

    lines = ["graph [", "  directed 0"]
    for x, i in ids.items():
        lines += ["  node [", f"    id {i}", f'    label "{x}"', f'    role "{roles[x]}"']
        if (bw := bandwidth(x)) is not None:
            lines += [f'    host_bandwidth_down "{bw} bit"', f'    host_bandwidth_up "{bw} bit"']
        lines.append("  ]")
    for e, a, b in links:
        ell = (e.ell.value or 0) * hw.tick_ns
        jitter = (e.delay_jitter.value or 0) * hw.tick_ns if e.delay_jitter else 0
        lines += ["  edge [", f"    source {ids[a]}", f"    target {ids[b]}", f'    label "{e.id}"',
                  f'    latency "{ell} ns"', f'    jitter "{jitter} ns"', "    packet_loss 0.0", "  ]"]
    lines.append("]")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--leaves", type=int, default=1)
    ap.add_argument("--spines", type=int, default=0)
    ap.add_argument("--outside", type=int, default=1)
    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("--json", action="store_true", help="the device graph")
    out.add_argument("--netns", action="store_true", help="the netns plan's commands, one per line")
    out.add_argument("--gml", action="store_true", help="Shadow's network graph")
    args = ap.parse_args(argv)
    hw = leaf_spine(args.blocks, leaves=args.leaves, spines=args.spines, outside=args.outside).check()
    if args.json:
        print(json.dumps(hw.to_dict(), indent=1))
    elif args.netns:
        print("\n".join(" ".join(c) for c in netns_plan(hw).up()))
    else:
        print(shadow_gml(hw), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
