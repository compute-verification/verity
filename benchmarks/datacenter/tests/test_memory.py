"""Timed audits bind bits to a device. The honest holder answers a challenge in time, and a store kept in host DRAM or on
another node answers late. A deadline too loose to tell the tiers apart locates the store to no single device, and so
does a small sample, which a deadline can locate only by latency. Labels the prover could rebuild in time locate
nowhere. A read window's bits count as local only beyond what every path into the registered device could carry."""
from __future__ import annotations

import pytest

from datacenter import (Assumption, Block, Cadence, Challenge, Datacenter, Deviation, Device, Held, Input, Interval, Link,
                        Plan, Read, ReadWindow, Refused, Store, account, audit, place, run, trace)

GB = 8e9
US = Assumption("hop", 1e-6, "s", "measured")
KV = Store("kv", 1 * GB, "a.gpu")            # 1 GB of KV pages, registered on a.gpu
PCIE, NET = 64e9 * 8, 400e9


def node(n: str) -> Block:
    gpu = Device(f"{n}.gpu", "gpu", rate={"square-1024": Assumption("rtx-pro-6000/squarings-p1024", 1.499e10, "1/s")},
                 memory=Assumption(f"{n}.gpu.hbm", 96 * GB, "bits"),
                 bandwidth=Assumption(f"{n}.gpu.hbm_bw", 1.6e12 * 8, "bit/s"))
    dram = Device(f"{n}.dram", "dram", memory=Assumption(f"{n}.dram.size", 512 * GB, "bits"),
                  bandwidth=Assumption(f"{n}.dram.bw", 200e9 * 8, "bit/s"))
    return Block(n, [gpu, dram])


def datacenter() -> Datacenter:
    net = Assumption("uplink", NET, "bit/s", "declared")
    pcie = Assumption("pcie5x16", PCIE, "bit/s")
    fabric = [Device("a.cert", "certifier"), Device("b.cert", "certifier"), Device("leaf", "switch"),
              Device("challenger", "challenger"), Device("world", "outside")]
    links = [Link("a.gpu", "a.dram", pcie, kind="pcie"), Link("b.gpu", "b.dram", pcie, kind="pcie"),
             Link("a.gpu", "a.cert", net, US), Link("a.cert", "leaf", net, US), Link("b.gpu", "b.cert", net, US),
             Link("b.cert", "leaf", net, US), Link("challenger", "leaf", net, US), Link("world", "leaf", net, US)]
    tick = Assumption("tick", 1e-4, "s", "declared")                # 100 µs ticks
    return Datacenter("dc0", [node("a"), node("b")], fabric, links, Cadence(buckets=4, tick=tick))


def setup(plan: Plan, stores=(KV,)):
    c = trace(lambda x: x ^ 1, x=Input(8))
    p = place(c, datacenter(), "a.gpu", deliver="world", stores=stores, x="world")
    return p, plan, p.public(plan)


def challenged(deadline: int):
    return setup(Plan(Interval(0, 1000), challenges=[Challenge("kv", at=0, deadline=deadline)]))


def test_the_honest_holder_answers_in_time_and_only_it_could_have():
    p, plan, profile = challenged(deadline=8)   # 6 µs of round trip + 1 GB at 1.6 TB/s ≈ 0.63 ms ≈ 7 ticks
    claims = audit(run(p, {"x": 3}, plan), profile)
    assert Held("kv", 1 * GB, Interval(0, 8)) in claims and not [c for c in claims if isinstance(c, Refused)]
    assert account(claims, profile).located == {"kv": ["a.gpu"]}


@pytest.mark.parametrize("elsewhere", ["a.dram", "b.gpu"])
def test_host_dram_passing_for_hbm_and_a_store_on_another_node_answer_late(elsewhere):
    p, plan, profile = challenged(deadline=8)
    refused = [c for c in audit(run(p, {"x": 3}, plan, Deviation(moved={"kv": elsewhere})), profile)
               if isinstance(c, Refused)]
    assert len(refused) == 1 and "past its deadline" in refused[0].why


def test_a_deadline_too_loose_locates_the_store_to_no_single_device():
    p, plan, profile = challenged(deadline=300)  # 30 ms: host DRAM over PCIe (5 ms read, 16 ms over PCIe) fits too
    acc = account(audit(run(p, {"x": 3}, plan), profile), profile)
    assert acc.sound and "a.gpu" in acc.located["kv"] and "a.dram" in acc.located["kv"]


def test_a_small_sample_locates_by_latency_and_labels_rebuilt_in_time_locate_nowhere():
    """PoUS answers s sampled blocks, not the store. One 4 KiB block within a 100 µs tick is answered as fast from host
    DRAM or the next node as from HBM: only latency could tell them apart, and the tick is coarser than the path. And
    labels the prover can rebuild from their seed within the deadline locate nowhere, whatever answered (μ_rebuild's
    twin: 10 squarings a bit on a 1.5·10^10/s squarer rebuilds a block in 22 µs)."""
    block = 4096 * 8
    p, plan, profile = setup(Plan(Interval(0, 1000), challenges=[Challenge("kv", 0, 1, sample=block)]))
    assert {"a.gpu", "a.dram", "b.gpu"} <= set(account(audit(run(p, {"x": 3}, plan), profile), profile).located["kv"])
    for regen, located in ((0.0, True), (1000.0, True), (10.0, False)):
        labels = Store("labels", 1 * GB, "a.gpu", "labels", regen)
        p, plan, profile = setup(Plan(Interval(0, 1000), challenges=[Challenge("labels", 0, 1, sample=block)]), (labels,))
        assert bool(account(audit(run(p, {"x": 3}, plan), profile), profile).located["labels"]) == located, regen


def test_a_read_windows_bits_are_its_output_and_local_beyond_every_path_into_the_device():
    """The bandwidth audits: the prover reads the KV store from tick 0 to the boundary at 10 (1 ms); its roots are due
    by 12. HBM reads all 8 Gbit in that time. Every path into a.gpu (its 400 Gbit/s uplink and host DRAM's 512 Gbit/s
    PCIe link) could have carried 0.912 Gbit of it, so at least 7.088 Gbit were read on the GPU. A store kept in host
    DRAM or on the next node reads at most its path's rate, and certifies nothing local."""
    p, plan, profile = setup(Plan(Interval(0, 16), reads=[ReadWindow("kv", start=0, boundary=10, deadline=12)]))
    claims = audit(run(p, {"x": 3}, plan), profile)
    assert Read("kv", 1 * GB, Interval(0, 10)) in claims
    a = account(claims, profile).blocks[0]
    assert a.reads == 1 * GB and a.reads_local_at_least == pytest.approx(1 * GB - (NET + PCIE) * 1e-3)
    for elsewhere, rate in (("a.dram", PCIE), ("b.gpu", NET)):
        moved = account(audit(run(p, {"x": 3}, plan, Deviation(moved={"kv": elsewhere})), profile), profile).blocks[0]
        assert moved.reads == pytest.approx(rate * 1e-3) and moved.reads_local_at_least == 0
    late = Plan(Interval(0, 16), reads=[ReadWindow("kv", start=0, boundary=10, deadline=10)])
    assert [c.why for c in audit(run(p, {"x": 3}, late), p.public(late)) if isinstance(c, Refused)] == [
        "the roots arrived after the deadline"]
