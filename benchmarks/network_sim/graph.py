"""The device graph: the measured node, and a leaf-spine datacenter of them, as the accounting layer's declarations
(`datacenter.Datacenter`, `benchmarks/datacenter`), and the two exports a datacenter's network needs: the plan of Linux
network namespaces that `mockdc.py` stands up, and Shadow's network graph.

The types are `datacenter.physical`'s, the Lean physical model's mirror: a `Datacenter` of `Block`s, fabric `Device`s
and `Link`s, in bits, seconds and bits per second, every capacity a named `Assumption` with its source.

`rtx_node` is one node as vy-nebius-1 and -2 read with `nvidia-smi topo -m` and `lspci -tv` (18 Oct): two sockets, each
a CPU with its host DRAM and a PCIe root complex (a switch); eight GPUs, GPUs 0–3 on socket 0 and 4–7 on socket 1, each
on a root port of its own; one NIC, on socket 1. Its capacities are the ones memory accounting reads, at the values
`verity.core.protocols.remote_bandwidth.timing` states (the tests pin them): each GPU's memory, read bandwidth and
on-chip state (S_chip), its rate per op class (`Accounting.Op`'s tensor cores, CUDA cores and oracle, and the codec's
squarings at each width), each socket's host DRAM read bandwidth and caches, and each PCIe link's bandwidth. A capacity
not measured yet is `math.inf`: every reading of a capacity takes it as an upper bound, and infinity is the bound that
claims nothing.

`leaf_spine` puts such nodes behind a leaf-spine fabric, each node's NIC through its block's certifier.

The exports read any `Datacenter`. Each block is one node, standing in for the whole block, each fabric device is another,
and the links between two nodes are the network.

    python graph.py --blocks 2 (--json | --netns | --gml)
"""

from __future__ import annotations

import argparse
import dataclasses
import ipaddress
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DATACENTER = str(Path(__file__).resolve().parents[1] / "datacenter")
if DATACENTER not in sys.path:
    sys.path.insert(0, DATACENTER)

from datacenter import Assumption, Block, Cadence, Datacenter, Device, Link  # noqa: E402

NETNS_PREFIX = "na-"
INF = math.inf

S_CHIP = Assumption("rtx-pro-6000/S_chip", 263_163_904 * 8, "bits", "measured")
GPU_MEMORY = Assumption("rtx-pro-6000/memory", 103_079_215_104 * 8, "bits", "datasheet")
GPU_READ = Assumption("rtx-pro-6000/memory-read", 1_539_000_000_000 * 8, "bit/s", "measured")
GPU_RATE = {
    "tc": Assumption("rtx-pro-6000-server/e4m3", 500_000_000_000_000, "multiply-adds/s", "datasheet"),
    "cuda": Assumption("rtx-pro-6000/cuda-ops", INF, "ops/s", "measured"),
    "oracle": Assumption("rtx-pro-6000/sha512", 7_000_000_000, "compressions/s", "measured"),
    "square-1024": Assumption("rtx-pro-6000/squarings-p1024", 14_990_000_000, "squarings/s", "measured"),
    "square-512": Assumption("rtx-pro-6000/squarings-p512", 51_300_000_000, "squarings/s", "measured"),
}
HOST_CHIP = Assumption("xeon-6776p/caches", 940_435_456 // 2 * 8, "bits", "declared")
HOST_DRAM = Assumption("host-dram/socket-capacity", INF, "bits", "measured")
HOST_READ = Assumption("host-dram/socket-read", 303_400_000_000 * 8, "bit/s", "measured")
PCIE_GPU = Assumption("pcie/gpu", INF, "bit/s", "measured")
PCIE_SOCKET = Assumption("pcie/socket-aggregate", INF, "bit/s", "measured")
PCIE_NIC = Assumption("pcie/nic", INF, "bit/s", "measured")
SOCKET_LINK = Assumption("socket/inter-socket", INF, "bit/s", "measured")


def rtx_node(name: str = "node") -> Datacenter:
    """One node, a datacenter of one block named `name` and no fabric. A link inside it has no latency until one is
    measured: a delay of 0, the bound that claims nothing for a path's delay."""
    devices, links = [], []
    for s in range(2):
        cpu = Device(f"{name}.cpu{s}", "cpu", memory=HOST_DRAM, bandwidth=HOST_READ, on_chip=HOST_CHIP)
        rc = Device(f"{name}.pcie{s}", "switch")
        devices += [cpu, rc]
        links.append(Link(rc.name, cpu.name, PCIE_SOCKET, kind="pcie"))
        for k in range(4 * s, 4 * s + 4):
            gpu = Device(f"{name}.gpu{k}", "gpu", rate=dict(GPU_RATE), memory=GPU_MEMORY, bandwidth=GPU_READ,
                         on_chip=S_CHIP)
            devices.append(gpu)
            links.append(Link(gpu.name, rc.name, PCIE_GPU, kind="pcie"))
    nic = Device(f"{name}.nic0", "nic")
    devices.append(nic)
    links += [Link(nic.name, f"{name}.pcie1", PCIE_NIC, kind="pcie"),
              Link(f"{name}.cpu0", f"{name}.cpu1", SOCKET_LINK, kind="upi")]
    return Datacenter(name, [Block(name, devices)], [], links)


def leaf_spine(blocks: int, *, leaves: int = 1, spines: int = 0, outside: int = 1, uplink_gbps: int = 400,
               spine_gbps: int = 400, fabric_latency: float = 2e-6, uplink_latency: float = 1e-6,
               cadence: Cadence | None = None) -> Datacenter:
    """`blocks` nodes (`rtx_node`, named b0, b1, …) behind a leaf-spine fabric: each node's NIC to its certifier
    (`bI.cert`), the certifier to a leaf switch (round-robin over `leaves`), every leaf to every spine, and `outside`
    hosts outside every block (`extC`, kind `outside`) on the leaves too. With one leaf, `spines` may be 0. Latencies
    are one way, in seconds: `uplink_latency` from a NIC to its certifier, `fabric_latency` a fabric hop. `cadence` is
    the certifiers' (`datacenter.Cadence`'s default when None)."""
    if blocks < 1 or leaves < 1 or (leaves > 1 and spines < 1):
        raise ValueError("at least one block and one leaf, and spines to join more than one leaf")
    up = Assumption("fabric/uplink", uplink_gbps * 1_000_000_000, "bit/s", "declared")
    trunk = Assumption("fabric/spine", spine_gbps * 1_000_000_000, "bit/s", "declared")
    nic_hop = Assumption("fabric/uplink-latency", uplink_latency, "s", "declared")
    hop = Assumption("fabric/hop-latency", fabric_latency, "s", "declared")
    switches = [Device(f"leaf{j}", "switch") for j in range(leaves)]
    nodes, certs, links = [], [], []
    for i in range(blocks):
        node = rtx_node(f"b{i}")
        cert = Device(f"b{i}.cert", "certifier")
        nodes += node.blocks
        certs.append(cert)
        links += node.links
        links += [Link(f"b{i}.nic0", cert.name, up, nic_hop, "ethernet"),
                  Link(cert.name, switches[i % leaves].name, up, hop, "ethernet")]
    tops = [Device(f"spine{k}", "switch") for k in range(spines)]
    links += [Link(sw.name, sp.name, trunk, hop, "ethernet") for sp in tops for sw in switches]
    hosts = [Device(f"ext{c}", "outside") for c in range(outside)]
    links += [Link(x.name, switches[c % leaves].name, up, hop, "ethernet") for c, x in enumerate(hosts)]
    return Datacenter(f"leaf-spine-{blocks}", nodes, switches + certs + tops + hosts, links, cadence or Cadence())


def check(dc: Datacenter) -> Datacenter:
    """`dc`, or ValueError listing how it falls short of the model (`Datacenter.validate`)."""
    if problems := dc.validate():
        raise ValueError(f"{dc.name}: " + "; ".join(problems))
    return dc


def pending(dc: Datacenter) -> list[str]:
    """The capacities still to be measured."""
    return sorted(a.name for a in dc.assumptions() if a.value == INF)


# The exports.

def network(dc: Datacenter) -> tuple[dict[str, str], list[tuple[Link, str, str]]]:
    """The network the exports stand up: each node's role (`block`, or a fabric device's kind) and each link between
    two nodes, with the node at each end."""
    def node(x: str) -> str:
        b = dc.block_of(x)
        return b.name if b else x

    roles = {b.name: "block" for b in dc.blocks} | {d.name: d.kind for d in dc.fabric}
    links = [(e, node(e.a), node(e.b)) for e in dc.links if node(e.a) != node(e.b)]
    return roles, links


def netns_name(node: str) -> str:
    return NETNS_PREFIX + node.replace(".", "-")


@dataclass(frozen=True)
class Interface:
    name: str
    address: str  # with its prefix length
    link: tuple[str, str]  # the link's ends, as the declaration states them
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

    def toward(self, node: str) -> Interface:
        return next(i for i in self.interfaces if i.peer[0] == netns_name(node))


@dataclass(frozen=True)
class NetnsPlan:
    """The mock datacenter as Linux network namespaces on one host, one per node, named `na-<node>`: veth pairs for
    links, created straight into their namespaces so that the root namespace never holds one; static routes and IP
    forwarding in the switches only; netem on the links between fabric devices, at both ends, for their latency, jitter
    and rate. A block's only interfaces run to certifiers, whose namespaces forward nothing: past one, the block reaches
    the fabric only through the certifier's own connections. The certifier also drops whatever arrives from its block
    that isn't part of a connection it opened (`guard`)."""

    namespaces: tuple[Namespace, ...]

    def namespace(self, node: str) -> Namespace:
        return next(n for n in self.namespaces if n.node == node)

    def namespace_of(self, name: str) -> Namespace:
        return next(n for n in self.namespaces if n.name == name)

    def address(self, node: str, toward: str) -> str:
        """`node`'s address on its link to `toward`."""
        return self.namespace(node).toward(toward).address.split("/")[0]

    def fabric_address(self, certifier: str) -> str:
        """A certifier's address on its link into the fabric, the side away from its block."""
        n = self.namespace(certifier)
        return next(i.address.split("/")[0] for i in n.interfaces if self.namespace_of(i.peer[0]).role != "block")

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

    def down(self) -> list[list[str]]:
        """Deleting a namespace deletes its veths with it."""
        return [["ip", "netns", "del", n.name] for n in self.namespaces]


def _us(seconds: float) -> str:
    return f"{round(seconds * 1e6, 3):g}us"


def _netem(e: Link, jitter: float) -> tuple[str, ...]:
    out = ["delay", _us(e.delay)] + ([_us(jitter)] if jitter else [])
    if math.isfinite(float(e.capacity)):
        out += ["rate", f"{int(float(e.capacity))}bit"]
    return tuple(out)


def netns_plan(dc: Datacenter, jitter: float = 0.0, base: str = "10.77.0.0") -> NetnsPlan:
    """The netns plan of `dc`: each link between two nodes a /30 from `base` in link order, its first end's address the
    lower; netem on each link between two fabric devices, with its latency, `jitter` seconds of delay jitter, and its
    capacity when finite. A switch routes every other switch-reachable subnet via the first hop of a shortest path
    through switches; a block routes by default via its certifier, and any other fabric device via its switch. A block
    whose link ends anywhere but at a certifier is refused."""
    check(dc)
    roles, links = network(dc)
    start = ipaddress.IPv4Address(base)
    ifaces: dict[str, list[tuple[str, str, str, str, Link, int]]] = {x: [] for x in roles}
    for n, (e, a, b) in enumerate(links):
        if "block" in (roles[a], roles[b]) and "certifier" not in (roles[a], roles[b]):
            raise ValueError(f"the link {e.a}-{e.b} leaves block {a if roles[a] == 'block' else b} past its certifier")
        ia, ib = f"eth{len(ifaces[a]) + 1}", f"eth{len(ifaces[b]) + 1}"
        net = start + 4 * n
        ifaces[a].append((ia, f"{net + 1}/30", b, ib, e, n))
        ifaces[b].append((ib, f"{net + 2}/30", a, ia, e, n))
    addr = {(x, n): address.split("/")[0] for x, xs in ifaces.items() for _, address, _, _, _, n in xs}
    nbrs = {x: [(peer, n) for _, _, peer, _, _, n in xs] for x, xs in ifaces.items()}
    subnets = {n: f"{start + 4 * n}/30" for n in range(len(links))}

    def first_hops(src: str) -> dict[str, tuple[str, int]]:
        """For each switch reachable from `src` through switches, the neighbour and link of a first hop to it."""
        out, frontier = {src: None}, [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v, n in nbrs[u]:
                    if roles[v] == "switch" and v not in out:
                        out[v] = out[u] or (v, n)
                        nxt.append(v)
            frontier = nxt
        return {k: v for k, v in out.items() if v is not None}

    fabric = {n for n, (_, a, b) in enumerate(links) if "block" not in (roles[a], roles[b])}
    namespaces = []
    for x in roles:
        routes: list[tuple[str, str]] = []
        if roles[x] == "switch":
            for s, (v, via) in sorted(first_hops(x).items()):
                for _, n in nbrs[s]:
                    if (x, n) not in addr and subnets[n] not in (d for d, _ in routes):
                        routes.append((subnets[n], addr[(v, via)]))
        else:
            ups = [(v, n) for v, n in nbrs[x] if roles[v] == ("certifier" if roles[x] == "block" else "switch")]
            if ups:
                routes.append(("default", addr[ups[0]]))
        interfaces = tuple(Interface(name, address, (e.a, e.b), (netns_name(peer), pname),
                                     _netem(e, jitter) if n in fabric else ())
                           for name, address, peer, pname, e, n in ifaces[x])
        namespaces.append(Namespace(netns_name(x), x, roles[x], roles[x] == "switch", interfaces, tuple(routes)))
    return NetnsPlan(tuple(namespaces))


def shadow_gml(dc: Datacenter, jitter: float = 0.0) -> str:
    """Shadow's network graph (GML): a node per network node, its host bandwidth its links' capacity each way (none
    when one is infinite), and an edge per link with its latency and `jitter`. Shadow limits bandwidth per host, not per
    edge."""
    check(dc)
    roles, links = network(dc)
    ids = {x: i for i, x in enumerate(roles)}
    lines = ["graph [", "  directed 0"]
    for x, i in ids.items():
        lines += ["  node [", f"    id {i}", f'    label "{x}"', f'    role "{roles[x]}"']
        bw = sum(float(e.capacity) for e, a, b in links if x in (a, b))
        if math.isfinite(bw):
            lines += [f'    host_bandwidth_down "{int(bw)} bit"', f'    host_bandwidth_up "{int(bw)} bit"']
        lines.append("  ]")
    for e, a, b in links:
        lines += ["  edge [", f"    source {ids[a]}", f"    target {ids[b]}", f'    label "{e.a}-{e.b}"',
                  f'    latency "{round(e.delay * 1e9)} ns"', f'    jitter "{round(jitter * 1e9)} ns"',
                  "    packet_loss 0.0", "  ]"]
    lines.append("]")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--leaves", type=int, default=1)
    ap.add_argument("--spines", type=int, default=0)
    ap.add_argument("--outside", type=int, default=1)
    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("--json", action="store_true", help="the declaration")
    out.add_argument("--netns", action="store_true", help="the netns plan's commands, one per line")
    out.add_argument("--gml", action="store_true", help="Shadow's network graph")
    args = ap.parse_args(argv)
    dc = check(leaf_spine(args.blocks, leaves=args.leaves, spines=args.spines, outside=args.outside))
    if args.json:
        print(json.dumps(dataclasses.asdict(dc), indent=1))
    elif args.netns:
        print("\n".join(" ".join(c) for c in netns_plan(dc).up()))
    else:
        print(shadow_gml(dc), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
