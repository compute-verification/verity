"""The two exports a datacenter's network needs: the plan of Linux network namespaces that `mockdc.py` stands up, and
Shadow's network graph.

They read any `datacenter.Datacenter` (`benchmarks/datacenter`, the Lean physical model's mirror: `Zone`s, fabric
`DeviceEntry`s and `LinkEntry`s, in bits, seconds and bit/s, every capacity a named `Assumption` with its source). The
declarations themselves, the measured node and the leaf-spine datacenter of them among them, are
`datacenter.examples`'s, the node's constants read from `verity.core.protocols.remote_bandwidth.timing`. Each block is
one node, standing in for the whole block, each fabric device is another, and the links between two nodes are the
network.

    python graph.py --blocks 2 [--fleet] (--json | --netns | --gml)
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

from datacenter import Datacenter, LinkEntry  # noqa: E402
from datacenter.examples import leaf_spine, two_sites  # noqa: E402

NETNS_PREFIX = "na-"


def check(dc: Datacenter) -> Datacenter:
    """`dc`, or ValueError listing how it falls short of the model (`Datacenter.validate`)."""
    if problems := dc.validate():
        raise ValueError(f"{dc.name}: " + "; ".join(problems))
    return dc


def network(dc: Datacenter) -> tuple[dict[str, str], list[tuple[LinkEntry, str, str]]]:
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


def jitter_of(e: LinkEntry, jitter: float, wan_jitter: float | None) -> float:
    return wan_jitter if e.kind == "wan" and wan_jitter is not None else jitter


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


def _netem(e: LinkEntry, jitter: float) -> tuple[str, ...]:
    out = ["delay", _us(e.delay)] + ([_us(jitter)] if jitter else [])
    if math.isfinite(float(e.capacity)):
        out += ["rate", f"{int(float(e.capacity))}bit"]
    return tuple(out)


def netns_plan(dc: Datacenter, jitter: float = 0.0, base: str = "10.77.0.0",
               wan_jitter: float | None = None) -> NetnsPlan:
    """The netns plan of `dc`: each link between two nodes a /30 from `base` in link order, its first end's address the
    lower; netem on each link between two fabric devices, with its latency, `jitter` seconds of delay jitter
    (`wan_jitter` on a WAN link, when given), and its capacity when finite. A switch routes every other
    switch-reachable subnet via the first hop of a shortest path through switches; a block routes by default via its
    certifier, and any other fabric device via its switch. A block whose link ends anywhere but at a certifier is
    refused."""
    check(dc)
    roles, links = network(dc)
    start = ipaddress.IPv4Address(base)
    ifaces: dict[str, list[tuple[str, str, str, str, LinkEntry, int]]] = {x: [] for x in roles}
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
                                     _netem(e, jitter_of(e, jitter, wan_jitter)) if n in fabric else ())
                           for name, address, peer, pname, e, n in ifaces[x])
        namespaces.append(Namespace(netns_name(x), x, roles[x], roles[x] == "switch", interfaces, tuple(routes)))
    return NetnsPlan(tuple(namespaces))


def shadow_gml(dc: Datacenter, jitter: float = 0.0, wan_jitter: float | None = None) -> str:
    """Shadow's network graph (GML): a node per network node, its host bandwidth its links' capacity each way (none
    when one is infinite), and an edge per link with its latency and `jitter` (`wan_jitter` on a WAN link, when given).
    Shadow limits bandwidth per host, not per edge."""
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
                  f'    latency "{round(e.delay * 1e9)} ns"',
                  f'    jitter "{round(jitter_of(e, jitter, wan_jitter) * 1e9)} ns"',
                  "    packet_loss 0.0", "  ]"]
    lines.append("]")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--leaves", type=int, default=1)
    ap.add_argument("--spines", type=int, default=0)
    ap.add_argument("--outside", type=int, default=1, help="hosts outside every block, ext0, ext1, …")
    ap.add_argument("--fleet", action="store_true", help="datacenter.examples.two_sites' fleet, merged, with --blocks, "
                    "--leaves and --spines a site's (its one outside host on the west site)")
    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("--json", action="store_true", help="the declaration (datacenter.examples.leaf_spine's)")
    out.add_argument("--netns", action="store_true", help="the netns plan's commands, one per line")
    out.add_argument("--gml", action="store_true", help="Shadow's network graph")
    args = ap.parse_args(argv)
    dc = check(two_sites(args.blocks, leaves=args.leaves, spines=args.spines).merged() if args.fleet else
               leaf_spine(args.blocks, leaves=args.leaves, spines=args.spines,
                          outside=tuple(f"ext{c}" for c in range(args.outside))))
    if args.json:
        print(json.dumps(dataclasses.asdict(dc), indent=1))
    elif args.netns:
        print("\n".join(" ".join(c) for c in netns_plan(dc).up()))
    else:
        print(shadow_gml(dc), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
