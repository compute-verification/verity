"""The device graph: the generator's shapes, the validation (coverage and named capacities), what the Lean names derive,
memory accounting's values, and the two exports."""

from __future__ import annotations

import importlib.util
import ipaddress
import json
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
        spec = importlib.util.spec_from_file_location("network_sim_graph", HERE / "graph.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)  # dataclasses resolve string annotations through sys.modules
        spec.loader.exec_module(mod)
        yield mod


def with_link(hw, e):
    return replace(hw, links={**hw.links, e.id: e})


def test_a_node_is_a_block_of_eight_gpus_on_two_sockets_behind_its_certifier(g):
    hw = g.leaf_spine(2)
    assert hw.validate() == []
    assert sorted(hw.blocks()) == ["b0", "b1"] and hw.blocks()["b0"] == hw.units["b0"]
    assert Counter(hw.leaves[x].kind for x in hw.units["b0"]) == \
        {"gpu": 8, "hbm": 8, "cpu": 2, "host_dram": 2, "switch": 2, "nic": 1}
    assert hw.cert == hw.trusted == {"b0.cert", "b1.cert"}
    assert all(hw.leaves[c].block is None for c in hw.cert)
    for k in range(8):
        pcie = [e for e in hw.at(f"b0.gpu{k}") if e.kind == "pcie"]
        assert [e.ends for e in pcie] == [(f"b0.gpu{k}", f"b0.pcie{k // 4}")]  # its own root port, on its socket
        assert hw.units[f"b0.dev{k}"] == {f"b0.gpu{k}", f"b0.hbm{k}"}
        assert f"b0.gpu{k}" in hw.units[f"b0.s{k // 4}"]
    assert hw.links["b0.nic0.pcie"].ends == ("b0.nic0", "b0.pcie1") and "b0.nic0" in hw.units["b0.s1"]
    assert [e.id for e in hw.boundary(hw.units["b0"])] == ["b0.uplink"]
    assert hw.links["b0.uplink"].ends == ("b0.nic0", "b0.cert") and hw.links["b0.uplink"].kind == "nic_uplink"
    assert hw.covered(hw.units["b0"]) and hw.spatial_unit("b1")
    assert hw.link_capacity(hw.units["b0"]) == 400_000 and hw.uncertified_capacity(hw.units["b0"]) == 0


def test_more_leaves_take_spines_and_every_leaf_meets_every_spine(g):
    hw = g.leaf_spine(4, leaves=2, spines=3, outside=2)
    assert hw.validate() == []
    trunks = [e.ends for e in hw.links.values() if e.kind == "fabric" and e.ends[1].startswith("spine")]
    assert sorted(trunks) == [(f"leaf{j}", f"spine{k}") for j in range(2) for k in range(3)]
    assert [hw.links[f"b{i}.cert.fabric"].ends[1] for i in range(4)] == ["leaf0", "leaf1", "leaf0", "leaf1"]
    with pytest.raises(ValueError, match="spines to join"):
        g.leaf_spine(2, leaves=2)


def test_a_block_with_a_path_around_its_certifier_is_not_covered(g):
    hw = g.leaf_spine(2)
    up = hw.links["b0.uplink"]
    bypass = with_link(hw, replace(up, ends=("b0.nic0", "leaf0")))
    assert bypass.validate() == ["block b0: boundary link b0.uplink has no certifier at either end (coverage)"]
    assert bypass.uncertified_capacity(bypass.units["b0"]) == 400_000
    assert not bypass.spatial_unit("b0") and bypass.spatial_unit("b1")
    second = with_link(hw, replace(up, id="b0.side", ends=("b0.gpu3", "ext0"), kind="fabric"))
    assert second.validate() == ["block b0: boundary link b0.side has no certifier at either end (coverage)"]
    pending = with_link(hw, replace(up, id="b0.side", ends=("b0.gpu3", "ext0"), beta=g.PCIE_GPU))
    assert not pending.covered(pending.units["b0"]) and pending.uncertified_capacity(pending.units["b0"]) is None


def test_every_capacity_is_a_named_assumption(g):
    hw = g.leaf_spine(1)
    e = hw.links["b0.cert.fabric"]
    A = g.Assumption

    def errors(**kw):
        return with_link(hw, replace(e, **kw)).validate()

    assert errors(beta=A("", 1, "bits/tick", "declared")) == \
        ["b0.cert.fabric.beta: a capacity must be a named Assumption, not " + repr(A("", 1, "bits/tick", "declared"))]
    assert errors(beta=400_000)[0].startswith("b0.cert.fabric.beta: a capacity must be a named Assumption")
    assert errors(beta=A("x", 1, "bits/tick", "guessed")) == \
        ["b0.cert.fabric.beta: x's source 'guessed' is not one of ('datasheet', 'measured', 'declared')"]
    assert errors(ell=A("x", 1, "bits/tick", "declared")) == \
        ["b0.cert.fabric.ell: x is in 'bits/tick', where the field takes 'ticks'"]
    assert errors(beta=A("x", -1, "bits/tick", "declared")) == \
        ["b0.cert.fabric.beta: x's value -1 is not a whole number (or None, pending)"]
    twice = errors(beta=replace(e.beta, value=1))
    assert twice and all("fabric/uplink is stated twice" in x for x in twice)


def test_the_device_is_device_leans(g):
    hw = g.leaf_spine(1)
    d = hw.device("b0.gpu5")
    assert d.bw == hw.links["b0.hbm5.bus"].beta and d.mem == hw.leaves["b0.hbm5"].M
    assert d.chip == hw.leaves["b0.gpu5"].chip and set(d.rate) == set(g.OPS)
    assert list(d.cap) == ["b0.gpu5.pcie"] and d.cap["b0.gpu5.pcie"] == g.PCIE_GPU
    assert d.priced({"tc": 1}) == d.rate["tc"].value == 500_000_000
    assert d.priced({"tc": 1, "cuda": 1}) is None  # CUDA-core ops are pending
    assert hw.compute(hw.units["b0"]) == 8 * 500_000_000 and hw.compute(hw.units["b0.s0"]) == 4 * 500_000_000
    assert hw.M(hw.units["b0.dev0"]) == 96 * 2**30 * 8 and hw.M(hw.units["b0"]) is None  # host DRAM is pending
    with pytest.raises(ValueError, match="not a GPU"):
        hw.device("b0.cpu0")


def test_the_capacities_are_memory_accountings(g):
    """The generator states remote bandwidth's per-device constants at a 1 µs tick: one GPU's and each of the host's
    two sockets' halves."""
    from verity.core.protocols.remote_bandwidth import timing

    hw = g.leaf_spine(1)
    us = 1_000_000
    gpu, hbm, cpu, dram = (hw.leaves[x] for x in ("b0.gpu0", "b0.hbm0", "b0.cpu0", "b0.dram0"))
    assert gpu.chip.value == timing.GPU_CHIP_BYTES * 8
    assert hw.links["b0.hbm0.bus"].beta.value * us == timing.GPU_SPEC_BYTES_PER_S * 8
    assert hbm.read.value * us == timing.GPU_READ_BYTES_PER_S * 8
    assert gpu.rate["oracle"].value * us == timing.GPU_SHA512_PER_S
    assert 1.597 * (gpu.squarings["p1024"].value * us) == timing.GPU_SQUARINGS_PER_S["p1024"]
    assert 1.567 * (gpu.squarings["p512"].value * us) == timing.GPU_SQUARINGS_PER_S["p512"]
    assert 2 * cpu.chip.value == timing.HOST_CHIP_BYTES * 8
    assert 2 * dram.read.value * us == timing.HOST_READ_BYTES_PER_S * 8
    assert 2 * hw.links["b0.dram0.bus"].beta.value * us == timing.HOST_SPEC_BYTES_PER_S * 8
    assert {"pcie/gpu", "pcie/socket-aggregate"} <= set(hw.pending())
    assert hw.links["b0.pcie0.up"].beta.source == hw.links["b0.gpu0.pcie"].beta.source == "measured"


def test_a_tick_at_which_a_rate_isnt_whole_is_refused(g):
    with pytest.raises(ValueError, match="pick a tick at which every rate is a whole number"):
        g.leaf_spine(1, tick_ns=1)  # the host's 303.4 GB/s a socket is 2,427.2 bits a ns
    assert g.per_tick(7_000_000_000, 1_000) == 7_000
    with pytest.raises(ValueError, match="whole ticks"):
        g.leaf_spine(1, fabric_ell_ns=1_500)


def test_ell_is_the_least_latency_over_paths(g):
    hw = g.leaf_spine(2, fabric_ell_ns=3_000, uplink_ell_ns=1_000)
    assert hw.ell("b0.cert", "b1.cert") == 6 and hw.ell("b0.cert", "b0.cert") == 0
    assert hw.ell("b0.gpu0", "ext0") == 1 + 3 + 3  # inside the node, pending latencies count 0, a lower bound
    wide = g.leaf_spine(2, leaves=2, spines=1, fabric_ell_ns=3_000)
    assert wide.ell("b0.cert", "b1.cert") == 4 * 3


def test_the_locations_form_a_tree_with_no_trusted_leaf_in_a_block(g):
    hw = g.leaf_spine(1)
    across = replace(hw, units={**hw.units, "odd": frozenset({"b0.gpu0", "b0.gpu4"})}).validate()
    assert "locations b0.s0 and odd overlap without nesting: the locations form a tree" in across
    inside = replace(hw, leaves={**hw.leaves, "b0.cert": replace(hw.leaves["b0.cert"], block="b0")})
    assert "block b0: its location must be exactly the leaves that name it" in inside.validate()
    assert "block b0: holds trusted leaves ['b0.cert'], which run off the prover's machines" in inside.validate()
    assert "certifiers ['b0.cert'] are not trusted (cert ⊆ trusted)" in replace(hw, trusted=frozenset()).validate()


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
    plan = g.netns_plan(g.leaf_spine(2))
    assert [(n.name, n.role, n.forward) for n in plan.namespaces] == [
        ("na-b0", "block", False), ("na-b1", "block", False), ("na-leaf0", "switch", True),
        ("na-b0-cert", "certifier", False), ("na-b1-cert", "certifier", False), ("na-ext0", "host", False)]
    for b in ("b0", "b1"):
        (i,) = plan.namespace(b).interfaces
        assert i.peer == (f"na-{b}-cert", "eth1") and i.link == f"{b}.uplink" and i.netem == ()
        assert plan.namespace(b).routes == (("default", plan.address(f"{b}.cert", f"{b}.uplink")),)
    addresses = [i.address for n in plan.namespaces for i in n.interfaces]
    assert len(addresses) == len(set(addresses)) == 10
    for n in plan.namespaces:
        for i in n.interfaces:
            back = plan.namespace_of(i.peer[0]).interfaces
            peer = next(p for p in back if p.name == i.peer[1])
            assert peer.peer == (n.name, i.name) and peer.link == i.link
            assert ipaddress.ip_interface(peer.address).network == ipaddress.ip_interface(i.address).network
            assert bool(i.netem) == i.link.endswith("fabric")
    assert plan.namespace("ext0").interfaces[0].netem == ("delay", "2us", "rate", "400000000000bit")
    up = plan.up()
    veths = [c for c in up if c[:3] == ["ip", "link", "add"]]
    assert len(veths) == 5 and all(c[5] != "" and c.count("netns") == 2 for c in veths)  # never in the root namespace
    assert ["ip", "netns", "exec", "na-b0-cert", "iptables", "-A", "INPUT", "-i", "eth1", "-m", "conntrack", "!",
            "--ctstate", "ESTABLISHED,RELATED", "-j", "DROP"] in up
    assert plan.down() == [["ip", "netns", "del", n.name] for n in plan.namespaces]
    assert walk(plan, "na-ext0", plan.address("b0.cert", "b0.cert.fabric")) == "na-b0-cert"
    assert walk(plan, "na-ext0", plan.address("b0", "b0.uplink")) is None
    assert walk(plan, "na-b0", plan.address("ext0", "ext0.fabric")) is None


def test_switches_route_every_fabric_subnet_through_a_two_tier_fabric(g):
    hw = g.leaf_spine(4, leaves=2, spines=2, outside=2)
    plan = g.netns_plan(hw)
    for src in ("na-ext0", "na-ext1", "na-b2-cert"):
        for c in sorted(hw.cert):
            assert walk(plan, src, plan.address(c, f"{c}.fabric")) == g.netns_name(c)
        for b in ("b0", "b1", "b3"):
            assert walk(plan, src, plan.address(b, f"{b}.uplink")) is None
    assert walk(plan, "na-b2-cert", plan.address("b2", "b2.uplink")) == "na-b2"


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
        (4, "b1.cert", "certifier"), (5, "ext0", "host")]
    assert nodes[0]["host_bandwidth_up"] == nodes[0]["host_bandwidth_down"] == "400000000000 bit"
    assert [(e["source"], e["target"], e["latency"], e["jitter"]) for e in edges] == [
        (0, 3, "1000 ns", "0 ns"), (3, 2, "2000 ns", "0 ns"), (1, 4, "1000 ns", "0 ns"), (4, 2, "2000 ns", "0 ns"),
        (5, 2, "2000 ns", "0 ns")]


def test_the_command_line_prints_each_form(g, capsys):
    g.main(["--blocks", "1", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["cert"] == ["b0.cert"] and doc["bucket_ticks"] == 100_000 and len(doc["leaves"]) == 23 + 3
    g.main(["--blocks", "1", "--netns"])
    assert capsys.readouterr().out.splitlines()[0] == "ip netns add na-b0"
    g.main(["--blocks", "1", "--gml"])
    assert capsys.readouterr().out.startswith("graph [\n  directed 0\n")
