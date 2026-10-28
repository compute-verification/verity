"""The exports of `datacenter.examples`' declarations (coverage, the netns plan, Shadow's GML, the command line), and the
declarations' tensor-core rates pinned against the census."""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
CENSUS = HERE.parents[1] / "benchmarks" / "census" / "hardware.json"


@pytest.fixture(scope="module")
def g():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location("network_sim_graph", HERE / "graph.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)  # dataclasses resolve string annotations through sys.modules
        spec.loader.exec_module(mod)
        yield mod


@pytest.fixture(scope="module")
def ex(g):
    from datacenter import examples

    return examples


def test_a_path_around_a_certifier_breaks_coverage_and_no_export_takes_it(g, ex):
    from datacenter import Link

    dc = ex.leaf_spine(2)
    dc.links.append(Link("b0.gpu3", "leaf0", ex.PCIE_GPU))
    assert dc.validate() == ["boundary link b0.gpu3-leaf0 passes no certifier (coverage)"]
    for export in (g.check, g.netns_plan, g.shadow_gml):
        with pytest.raises(ValueError, match="passes no certifier"):
            export(dc)


def test_the_tensor_core_rates_are_the_censuss(ex):
    """The tensor-core rates are the census's dense datasheet peaks, a multiply-add being two FLOP. What isn't measured
    yet is infinite, so a reading that takes it as an upper bound concludes nothing from it."""
    from datacenter import serve_time

    dc = ex.leaf_spine(1)
    gpu, cpu = dc.devices["b0.gpu0"], dc.devices["b0.cpu0"]
    census = {e["id"]: e for e in json.loads(CENSUS.read_text())}
    macs = [a for k, a in gpu.rate.items() if k.startswith("mac:")]
    assert [a.name for a in macs] == ["rtx-pro-6000-server/bf16", "rtx-pro-6000-server/e4m3", "rtx-pro-6000-server/e2m1"]
    assert all(2 * a.value == census[a.name]["flop_per_second"] and a.source == "datasheet" for a in macs)
    assert sorted(a.name for a in dc.assumptions() if a.value == math.inf) == [
        "pcie/socket-aggregate", "rtx-pro-6000/cuda-ops", "socket/inter-socket"]
    bits = 8e9
    # a store registered on a GPU but kept in host DRAM: the GPU's PCIe 5.0 x16 link is narrower than the host's read
    assert ex.PCIE_GPU.value < cpu.bandwidth.value and serve_time(dc, "b0.gpu0", "b0.cpu0", bits) == bits / ex.PCIE_GPU.value


def walk(plan, src: str, dst: str) -> str | None:
    """The namespace a packet from `src` to `dst` reaches by the plan's addresses and routes, or None where it's
    dropped: no route, or a hop that would have to forward and doesn't."""
    owner = {i.address.split("/")[0]: n for n in plan.namespaces for i in n.interfaces}
    ns, ip = plan.namespace_of(src), ipaddress.ip_address(dst)
    for hops in range(len(plan.namespaces) + 1):
        if any(i.address.split("/")[0] == dst for i in ns.interfaces):
            return ns.name
        if hops and not ns.forward:
            return None
        if any(ip in ipaddress.ip_interface(i.address).network for i in ns.interfaces):
            nxt = dst
        else:
            routes = [(ipaddress.ip_network("0.0.0.0/0" if d == "default" else d), via) for d, via in ns.routes]
            fits = sorted((net.prefixlen, via) for net, via in routes if ip in net)
            if not fits:
                return None
            nxt = fits[-1][1]
        ns = owner[nxt]
    return None


def test_the_netns_plan_puts_each_block_behind_its_certifier(g, ex):
    dc = ex.leaf_spine(2)
    plan = g.netns_plan(dc)
    assert [(n.name, n.role, n.forward) for n in plan.namespaces] == [
        ("na-b0", "block", False), ("na-b1", "block", False), ("na-leaf0", "switch", True),
        ("na-b0-cert", "certifier", False), ("na-b1-cert", "certifier", False), ("na-ext0", "outside", False)]
    for b in ("b0", "b1"):
        (i,) = plan.namespace(b).interfaces
        assert i.peer == (f"na-{b}-cert", "eth1") and i.link == (f"{b}.nic0", f"{b}.cert") and i.netem == ()
        assert plan.namespace(b).routes == (("default", plan.address(f"{b}.cert", b)),)
    addresses = [i.address for n in plan.namespaces for i in n.interfaces]
    assert len(addresses) == len(set(addresses)) == 10
    for n in plan.namespaces:
        for i in n.interfaces:
            back = plan.namespace_of(i.peer[0]).interfaces
            peer = next(p for p in back if p.name == i.peer[1])
            assert peer.peer == (n.name, i.name) and peer.link == i.link
            assert ipaddress.ip_interface(peer.address).network == ipaddress.ip_interface(i.address).network
            assert bool(i.netem) == ("block" not in (n.role, plan.namespace_of(i.peer[0]).role))
    assert plan.namespace("ext0").interfaces[0].netem == ("delay", "2us", "rate", "400000000000bit")
    jittery = g.netns_plan(dc, jitter=5e-4)
    assert jittery.namespace("ext0").interfaces[0].netem == ("delay", "2us", "500us", "rate", "400000000000bit")
    up = plan.up()
    veths = [c for c in up if c[:3] == ["ip", "link", "add"]]
    assert len(veths) == 5 and all(c.count("netns") == 2 for c in veths)  # never in the root namespace
    assert ["ip", "netns", "exec", "na-b0-cert", "iptables", "-A", "INPUT", "-i", "eth1", "-m", "conntrack", "!",
            "--ctstate", "ESTABLISHED,RELATED", "-j", "DROP"] in up
    assert plan.down() == [["ip", "netns", "del", n.name] for n in plan.namespaces]
    assert walk(plan, "na-ext0", plan.fabric_address("b0.cert")) == "na-b0-cert"
    assert walk(plan, "na-ext0", plan.address("b0", "b0.cert")) is None
    assert walk(plan, "na-b0", plan.address("ext0", "leaf0")) is None


def test_an_infinite_capacity_puts_no_rate_on_its_link(g, ex):
    from datacenter import Assumption

    dc = ex.leaf_spine(1)
    dc.links = [replace(e, capacity=Assumption("fabric/open", math.inf, "bit/s", "measured")) if e.a == "ext0" else e
                for e in dc.links]
    assert g.netns_plan(dc).namespace("ext0").interfaces[0].netem == ("delay", "2us")


def test_switches_route_every_fabric_subnet_through_a_two_tier_fabric(g, ex):
    dc = ex.leaf_spine(4, leaves=2, spines=2, outside=("ext0", "ext1"))
    plan = g.netns_plan(dc)
    certs = [d.name for d in dc.fabric if d.kind == "certifier"]
    for src in ("na-ext0", "na-ext1", "na-b2-cert"):
        for c in certs:
            assert walk(plan, src, plan.fabric_address(c)) == g.netns_name(c)
        for b in ("b0", "b1", "b3"):
            assert walk(plan, src, plan.address(b, f"{b}.cert")) is None
    assert walk(plan, "na-b2-cert", plan.address("b2", "b2.cert")) == "na-b2"


def test_a_fleets_wan_joins_its_sites_switches_and_has_its_own_jitter(g, ex):
    """The WAN runs between the two spines, which route across it; only switches forward, and every certifier watches a
    block and has its guard. The outside host, on the west site, reaches every block's certifier, the east ones across
    the WAN, and no block; the WAN's netem is its latency, its own jitter and its capacity."""
    dc = ex.two_sites().merged()
    plan = g.netns_plan(dc, jitter=5e-4, wan_jitter=2e-3)
    assert len(plan.namespaces) == 15
    assert sorted(n.node for n in plan.namespaces if n.forward) == [f"{s}.{x}" for s in ("east", "west")
                                                                   for x in ("leaf0", "leaf1", "spine0")]
    guarded = [c[3] for c in plan.up() if "iptables" in c]
    assert guarded == [f"na-{s}-b{i}-cert" for s in ("east", "west") for i in range(2)]
    assert guarded == [n.name for n in plan.namespaces if n.role == "certifier"]
    (wan,) = [i for i in plan.namespace("east.spine0").interfaces if i.peer[0] == "na-west-spine0"]
    assert wan.netem == ("delay", "20000us", "2000us", "rate", "100000000000bit")
    assert plan.namespace("ext0").interfaces[0].netem == ("delay", "2us", "500us", "rate", "400000000000bit")
    ext = plan.address("ext0", "west.leaf0")
    out = str(ipaddress.ip_interface(plan.namespace("ext0").interfaces[0].address).network)
    assert dict(plan.namespace("east.spine0").routes)[out] == plan.address("west.spine0", "east.spine0")
    for b in (b.name for b in dc.blocks):
        cert = g.netns_name(f"{b}.cert")
        assert walk(plan, "na-ext0", plan.fabric_address(f"{b}.cert")) == cert and walk(plan, cert, ext) == "na-ext0"
        assert walk(plan, "na-ext0", plan.address(b, f"{b}.cert")) is None and walk(plan, g.netns_name(b), ext) is None
    assert walk(plan, "na-east-b0-cert", plan.fabric_address("west.b1.cert")) == "na-west-b1-cert"
    (_, graph), = parse_gml(g.shadow_gml(dc, jitter=5e-4, wan_jitter=2e-3))
    edges = {e["label"]: (e["latency"], e["jitter"]) for e in (dict(v) for k, v in graph if k == "edge")}
    assert edges["east.spine0-west.spine0"] == ("20000000 ns", "2000000 ns")
    assert edges["ext0-west.leaf0"] == ("2000 ns", "500000 ns")


def parse_gml(text: str) -> list:
    tokens = re.findall(r'"[^"]*"|\[|\]|[^\s\[\]]+', text)

    def parse(i: int) -> tuple[list, int]:
        out = []
        while i < len(tokens) and tokens[i] != "]":
            key, value = tokens[i], tokens[i + 1]
            if value == "[":
                value, i = parse(i + 2)
                i += 1
            else:
                value = value[1:-1] if value.startswith('"') else float(value) if "." in value else int(value)
                i += 2
            out.append((key, value))
        return out, i

    return parse(0)[0]


def test_the_shadow_gml_has_a_node_per_network_node_and_an_edge_per_link(g, ex):
    (key, graph), = parse_gml(g.shadow_gml(ex.leaf_spine(2)))
    nodes = [dict(v) for k, v in graph if k == "node"]
    edges = [dict(v) for k, v in graph if k == "edge"]
    assert key == "graph" and dict(graph)["directed"] == 0
    assert [(n["id"], n["label"], n["role"]) for n in nodes] == [
        (0, "b0", "block"), (1, "b1", "block"), (2, "leaf0", "switch"), (3, "b0.cert", "certifier"),
        (4, "b1.cert", "certifier"), (5, "ext0", "outside")]
    assert nodes[0]["host_bandwidth_up"] == nodes[0]["host_bandwidth_down"] == "400000000000 bit"
    assert [(e["source"], e["target"], e["latency"], e["jitter"]) for e in edges] == [
        (0, 3, "1000 ns", "0 ns"), (3, 2, "2000 ns", "0 ns"), (1, 4, "1000 ns", "0 ns"), (4, 2, "2000 ns", "0 ns"),
        (5, 2, "2000 ns", "0 ns")]


def test_the_command_line_prints_each_form(g, capsys):
    g.main(["--blocks", "1", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["name"] == "leaf-spine-1" and len(doc["blocks"][0]["devices"]) == 13
    assert [d["name"] for d in doc["fabric"]] == ["leaf0", "b0.cert", "ext0"]
    g.main(["--blocks", "1", "--outside", "2", "--json"])
    assert [d["name"] for d in json.loads(capsys.readouterr().out)["fabric"]][-2:] == ["ext0", "ext1"]
    g.main(["--blocks", "1", "--netns"])
    assert capsys.readouterr().out.splitlines()[0] == "ip netns add na-b0"
    g.main(["--blocks", "1", "--gml"])
    assert capsys.readouterr().out.startswith("graph [\n  directed 0\n")
    g.main(["--blocks", "1", "--fleet", "--netns"])
    assert capsys.readouterr().out.splitlines()[:2] == ["ip netns add na-east-b0", "ip netns add na-west-b0"]
