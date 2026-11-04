"""The declaration library: the measured node, a BlueField-3 behind its NIC, the leaf-spine generator and the demo's rack."""
from __future__ import annotations

from collections import Counter

import pytest
from verity.core.protocols.remote_bandwidth import ledger

from datacenter import Cadence, Port, route
from datacenter.examples import GB, leaf_spine, rack, rtx_node, two_sites, with_dpu


def test_a_node_is_one_block_of_eight_gpus_on_two_sockets_with_the_measured_pcie_tree():
    """Its devices add up to PoUB's node, `ledger.Node`: `Accounting.Device`'s one Python mirror."""
    dc, node = rtx_node("n"), ledger.Node.stated()
    ds = dc.devices.values()
    assert sum(float(d.on_chip) for d in ds if d.on_chip) == node.chip_bytes * 8
    assert sum(float(d.bandwidth) for d in ds if d.bandwidth) == pytest.approx(node.read_bytes_per_s * 8)
    assert sum(float(d.write) for d in ds if d.write) == pytest.approx(
        (8 * node.gpu_write_bytes_per_s + node.host_write_bytes_per_s) * 8)
    for w, rate in node.squarings_per_s.items():
        assert sum(float(d.rate[f"square-{w[1:]}"]) for d in ds if d.rate) == pytest.approx(rate)
    assert dc.validate() == []
    (block,) = dc.blocks
    assert block.name == "n" and dc.fabric == [] and dc.boundary == []
    assert Counter(d.kind for d in block.devices) == {"gpu": 8, "cpu": 2, "switch": 2, "nic": 1}
    for k in range(8):
        assert [(e.a, e.b, e.kind) for e in dc.links if f"n.gpu{k}" in (e.a, e.b)] == \
            [(f"n.gpu{k}", f"n.pcie{k // 4}", "pcie")]  # its own root port, on its socket
    assert [(e.a, e.b) for e in dc.links if "n.nic0" in (e.a, e.b)] == [("n.nic0", "n.pcie1")]
    assert [(e.a, e.b) for e in route(dc, "n.gpu0", "n.nic0")] == [
        ("n.gpu0", "n.pcie0"), ("n.pcie0", "n.cpu0"), ("n.cpu0", "n.cpu1"), ("n.pcie1", "n.cpu1"), ("n.nic0", "n.pcie1")]
    assert all(e.latency is None and e.delay == 0.0 for e in dc.links)  # not measured: a delay that claims nothing
    gpu = dc.devices["n.gpu0"]
    assert set(gpu.rate) == {"mac:bf16", "mac:fp8", "mac:fp4", "cuda:op", "cuda:gate", "oracle", "square-1024",
                             "square-512"}
    assert gpu.rate["cuda:gate"].value == 2.6e13 * 32


def test_a_dpu_sits_between_the_nic_and_the_way_out_with_its_bmc_behind_it():
    dc = with_dpu(rtx_node("n"))
    assert dc.validate() == []
    dpu, bmc = dc.devices["n.dpu"], dc.devices["n.bmc"]
    assert dc.block_of("n.dpu").name == dc.block_of("n.bmc").name == "n"
    assert [(e.a, e.b) for e in dc.links if "n.dpu" in (e.a, e.b)] == [("n.nic0", "n.dpu"), ("n.dpu", "n.bmc")]
    assert dpu.memory.value == 200 * GB and dpu.residual.value == pytest.approx(6.48 * GB)
    assert dpu.write is bmc.write is None                             # no write ceiling stated: a wipe fills at its links
    assert bmc.memory.value == pytest.approx(1.32 * GB) and bmc.residual.value == pytest.approx(0.442 * GB)
    behind = leaf_spine(1, dpus=True)
    assert [(e.a, e.b) for e in behind.boundary] == [("b0.dpu", "b0.cert")]


def test_leaf_spine_puts_each_node_behind_its_certifier():
    dc = leaf_spine(2)
    assert dc.validate() == []
    assert [b.name for b in dc.blocks] == ["b0", "b1"]
    assert [(d.name, d.kind, d.trusted) for d in dc.fabric] == [
        ("leaf0", "switch", False), ("b0.cert", "certifier", True), ("b1.cert", "certifier", True),
        ("ext0", "outside", False)]
    assert [(e.a, e.b) for e in dc.boundary] == [("b0.nic0", "b0.cert"), ("b1.nic0", "b1.cert")]
    assert [repr(p) for p in dc.ports] == ["b0.cert:in", "b0.cert:out", "b1.cert:in", "b1.cert:out"]
    assert dc.port("b0.nic0", "b0.cert") == Port("b0.cert", "egress") and dc.port("b0.cert", "b0.nic0").direction == "ingress"
    assert {(e.latency.name, e.latency.value) for e in dc.links if e.latency} == {
        ("fabric/uplink-latency", 1e-6), ("fabric/hop-latency", 2e-6)}
    assert dc.cadence == Cadence() and leaf_spine(1, cadence=Cadence(buckets=60)).cadence.buckets == 60
    wide = leaf_spine(4, leaves=2, spines=3, outside=("ext0", "ext1"))
    assert wide.validate() == []
    trunks = sorted((e.a, e.b) for e in wide.links if e.b.startswith("spine"))
    assert trunks == [(f"leaf{j}", f"spine{k}") for j in range(2) for k in range(3)]
    assert [next(e.b for e in wide.links if e.a == f"b{i}.cert") for i in range(4)] == ["leaf0", "leaf1", "leaf0", "leaf1"]
    assert sorted((e.a, e.b) for e in wide.uncovered_links()) == sorted(trunks + [("ext0", "leaf0"), ("ext1", "leaf1")])
    assert [(e.a, e.b) for e in dc.uncovered_links()] == [("ext0", "leaf0")]
    with pytest.raises(ValueError, match="spines to join"):
        leaf_spine(2, leaves=2)


def test_the_rack_is_nodes_with_dpus_on_one_leaf_with_a_challenger_and_the_world_over_a_wan():
    dc = rack(nodes=2)
    assert dc.validate() == []
    assert dc.name == "rack" and [b.name for b in dc.blocks] == ["n0", "n1"]
    assert [(d.name, d.kind, d.trusted) for d in dc.fabric] == [
        ("leaf0", "switch", False), ("n0.cert", "certifier", True), ("n1.cert", "certifier", True),
        ("world", "outside", False), ("challenger", "challenger", True)]
    assert [(e.a, e.b) for e in dc.boundary] == [("n0.dpu", "n0.cert"), ("n1.dpu", "n1.cert")]
    world = dc.link("world", "leaf0")
    assert world.kind == "wan" and world.delay == 1e-3 and dc.link("challenger", "leaf0").delay == 1e-6
    c = dc.cadence
    assert (c.frame_bits, c.bucket_ticks, c.buckets, c.jitter_classes, c.tick.value) == (32, 1, 4, 128, 1e-4)
    named = [a for d in dc.devices.values() for a in [*d.rate.values(), d.memory, d.bandwidth, d.on_chip, d.residual,
                                                      d.write] if a]
    named += [a for e in dc.links for a in (e.capacity, e.latency) if a] + [c.tick]
    assert all(a.name and a.source in ("datasheet", "measured", "declared") for a in named)
    assert all(a == b for a in named for b in named if a.name == b.name)  # one name, one record
    assert [b.name for b in rack(nodes=1, dpus=False).blocks] == ["n0"] and "n0.dpu" not in rack(1, False).devices


def test_two_sites_are_two_leaf_spines_joined_by_a_wan_between_their_spines():
    """The WAN's capacity and latency are declared assumptions; it is fabric between the two spines, with no certifier
    and no port, since every block is behind its own certifier; and an east block's way out to the one outside host, on
    the west site, crosses it."""
    fleet = two_sites()
    assert fleet.validate() == []
    dc = fleet.merged()
    assert dc.validate() == [] and dc.name == "east+west"
    assert [b.name for b in dc.blocks] == ["east.b0", "east.b1", "west.b0", "west.b1"]
    assert [repr(p) for p in dc.ports] == [f"{s}.b{i}.cert:{d}" for s in ("east", "west") for i in range(2)
                                           for d in ("in", "out")]
    assert sorted(d.name for d in dc.fabric if d.kind == "certifier") == [f"{s}.b{i}.cert" for s in ("east", "west")
                                                                           for i in range(2)]
    (wan,) = fleet.links
    assert (wan.a, wan.b, wan.kind, wan.delay) == ("east.spine0", "west.spine0", "wan", 20e-3)
    assert [repr(a) for a in (wan.capacity, wan.latency)] == ["wan/east-west=1e+11 bit/s (declared)",
                                                             "wan/east-west-latency=0.02 s (declared)"]
    assert [d.name for d in dc.fabric if d.kind == "outside"] == ["ext0"]
    assert [(e.a, e.b) for e in route(dc, "east.b0.nic0", "ext0")] == [
        ("east.b0.nic0", "east.b0.cert"), ("east.b0.cert", "east.leaf0"), ("east.leaf0", "east.spine0"),
        ("east.spine0", "west.spine0"), ("west.leaf0", "west.spine0"), ("ext0", "west.leaf0")]
    named = [a for e in dc.links for a in (e.capacity, e.latency) if a]
    assert all(a == b for a in named for b in named if a.name == b.name)
    flat = two_sites(spines=0, leaves=1)
    assert flat.validate() == [] and [(e.a, e.b) for e in flat.links] == [("east.leaf0", "west.leaf0")]
