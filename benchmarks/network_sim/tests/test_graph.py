"""The device graph: the measured node and the leaf-spine generator as `datacenter` declarations, memory accounting's
values, coverage, and the two exports."""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import math
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def g():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location("network_sim_graph", HERE / "graph.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)  # dataclasses resolve string annotations through sys.modules
        spec.loader.exec_module(mod)
        yield mod


def test_a_node_is_one_block_of_eight_gpus_on_two_sockets_with_the_measured_pcie_tree(g):
    from datacenter import route

    dc = g.check(g.rtx_node("n"))
    (block,) = dc.blocks
    assert block.name == "n" and dc.fabric == [] and dc.boundary == []
    assert Counter(d.kind for d in block.devices) == {"gpu": 8, "cpu": 2, "switch": 2, "nic": 1}
    for k in range(8):
        assert [(e.a, e.b, e.kind) for e in dc.links if f"n.gpu{k}" in (e.a, e.b)] == \
            [(f"n.gpu{k}", f"n.pcie{k // 4}", "pcie")]  # its own root port, on its socket
    assert [(e.a, e.b) for e in dc.links if "n.nic0" in (e.a, e.b)] == [("n.nic0", "n.pcie1")]
    assert [(e.a, e.b) for e in route(dc, "n.gpu0", "n.nic0")] == [
        ("n.gpu0", "n.pcie0"), ("n.pcie0", "n.cpu0"), ("n.cpu0", "n.cpu1"), ("n.pcie1", "n.cpu1"), ("n.nic0", "n.pcie1")]


def test_leaf_spine_puts_each_node_behind_its_certifier(g):
    dc = g.check(g.leaf_spine(2))
    assert [b.name for b in dc.blocks] == ["b0", "b1"]
    assert [(d.name, d.kind, d.trusted) for d in dc.fabric] == [
        ("leaf0", "switch", False), ("b0.cert", "certifier", True), ("b1.cert", "certifier", True),
        ("ext0", "outside", False)]
    assert [(e.a, e.b) for e in dc.boundary] == [("b0.nic0", "b0.cert"), ("b1.nic0", "b1.cert")]
    assert [dc.certifier_of(e).name for e in dc.boundary] == ["b0.cert", "b1.cert"]
    wide = g.check(g.leaf_spine(4, leaves=2, spines=3, outside=2))
    trunks = sorted((e.a, e.b) for e in wide.links if e.b.startswith("spine"))
    assert trunks == [(f"leaf{j}", f"spine{k}") for j in range(2) for k in range(3)]
    assert [next(e.b for e in wide.links if e.a == f"b{i}.cert") for i in range(4)] == ["leaf0", "leaf1", "leaf0", "leaf1"]
    with pytest.raises(ValueError, match="spines to join"):
        g.leaf_spine(2, leaves=2)


def test_a_path_around_a_certifier_breaks_coverage_and_no_export_takes_it(g):
    from datacenter import Link

    dc = g.leaf_spine(2)
    dc.links.append(Link("b0.gpu3", "leaf0", g.PCIE_GPU))
    assert dc.validate() == ["boundary link b0.gpu3-leaf0 passes no certifier (coverage)"]
    for export in (g.check, g.netns_plan, g.shadow_gml):
        with pytest.raises(ValueError, match="passes no certifier"):
            export(dc)


def test_the_capacities_are_memory_accountings(g):
    """The node states remote bandwidth's per-device constants: one GPU's, and each of the host's two sockets' halves.
    What isn't measured yet is infinite, so a reading that takes it as an upper bound concludes nothing from it."""
    from datacenter.memory import read_time
    from verity.core.protocols.remote_bandwidth import timing

    dc = g.leaf_spine(1)
    gpu, cpu = dc.devices["b0.gpu0"], dc.devices["b0.cpu0"]
    assert gpu.on_chip.value == timing.GPU_CHIP_BYTES * 8 and gpu.memory.value == 96 * 2**30 * 8
    assert gpu.bandwidth.value == timing.GPU_READ_BYTES_PER_S * 8
    assert gpu.rate["oracle"].value == timing.GPU_SHA512_PER_S and gpu.rate["tc"].value == 5e14
    assert 1.597 * gpu.rate["square-1024"].value == timing.GPU_SQUARINGS_PER_S["p1024"]
    assert 1.567 * gpu.rate["square-512"].value == timing.GPU_SQUARINGS_PER_S["p512"]
    assert 2 * cpu.on_chip.value == timing.HOST_CHIP_BYTES * 8
    assert 2 * cpu.bandwidth.value == timing.HOST_READ_BYTES_PER_S * 8
    assert g.pending(dc) == ["host-dram/socket-capacity", "pcie/gpu", "pcie/nic", "pcie/socket-aggregate",
                             "rtx-pro-6000/cuda-ops", "socket/inter-socket"]
    named = [a for d in dc.devices.values() for a in [*d.rate.values(), d.memory, d.bandwidth, d.on_chip] if a]
    named += [e.capacity for e in dc.links]
    assert all(a.name and a.source in ("datasheet", "measured", "declared") for a in named)
    assert all(a == b for a in named for b in named if a.name == b.name)  # one name, one record
    bits = 8e9
    assert read_time(dc, "b0.gpu0", "b0.cpu0", bits) == bits / cpu.bandwidth.value  # the pending PCIe path costs nothing


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


def test_the_netns_plan_puts_each_block_behind_its_certifier(g):
    dc = g.leaf_spine(2)
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


def test_an_infinite_capacity_puts_no_rate_on_its_link(g):
    from datacenter import Assumption

    dc = g.leaf_spine(1)
    dc.links = [replace(e, capacity=Assumption("fabric/open", math.inf, "bit/s", "measured")) if e.a == "ext0" else e
                for e in dc.links]
    assert g.netns_plan(dc).namespace("ext0").interfaces[0].netem == ("delay", "2us")


def test_switches_route_every_fabric_subnet_through_a_two_tier_fabric(g):
    dc = g.leaf_spine(4, leaves=2, spines=2, outside=2)
    plan = g.netns_plan(dc)
    certs = [d.name for d in dc.fabric if d.kind == "certifier"]
    for src in ("na-ext0", "na-ext1", "na-b2-cert"):
        for c in certs:
            assert walk(plan, src, plan.fabric_address(c)) == g.netns_name(c)
        for b in ("b0", "b1", "b3"):
            assert walk(plan, src, plan.address(b, f"{b}.cert")) is None
    assert walk(plan, "na-b2-cert", plan.address("b2", "b2.cert")) == "na-b2"


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


def test_the_shadow_gml_has_a_node_per_network_node_and_an_edge_per_link(g):
    (key, graph), = parse_gml(g.shadow_gml(g.leaf_spine(2)))
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
    g.main(["--blocks", "1", "--netns"])
    assert capsys.readouterr().out.splitlines()[0] == "ip netns add na-b0"
    g.main(["--blocks", "1", "--gml"])
    assert capsys.readouterr().out.startswith("graph [\n  directed 0\n")
