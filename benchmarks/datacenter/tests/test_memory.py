"""Timed audits bind bits to a device: the honest holder answers a challenge in time, a store kept in host DRAM or on
another node answers late, a deadline too loose to tell the tiers apart locates the store to no single device, and a
bandwidth audit's reads count as local only beyond what the block's boundary could have carried in."""
from __future__ import annotations

import pytest

from datacenter import (Assumption, Block, Challenge, Datacenter, Deviation, Device, Grid, Held, Input, Interval, Link,
                        Plan, Read, Reads, Refused, Store, account, audit, place, run, trace)

GB = 8e9
US = Assumption("hop", 1e-6, "s", "measured")
KV = Store("kv", 1 * GB, "a.gpu")            # 1 GB of KV pages, registered on a.gpu


def node(n: str) -> Block:
    gpu = Device(f"{n}.gpu", "gpu", memory=Assumption(f"{n}.gpu.hbm", 96 * GB, "bits"),
                 bandwidth=Assumption(f"{n}.gpu.hbm_bw", 1.6e12 * 8, "bit/s"))
    dram = Device(f"{n}.dram", "dram", memory=Assumption(f"{n}.dram.size", 512 * GB, "bits"),
                  bandwidth=Assumption(f"{n}.dram.bw", 200e9 * 8, "bit/s"))
    return Block(n, [gpu, dram])


def datacenter() -> Datacenter:
    net = Assumption("uplink", 400e9, "bit/s", "declared")
    pcie = Assumption("pcie5x16", 64e9 * 8, "bit/s")
    fabric = [Device("a.cert", "certifier"), Device("b.cert", "certifier"), Device("leaf", "switch"),
              Device("challenger", "challenger"), Device("world", "outside")]
    links = [Link("a.gpu", "a.dram", pcie, kind="pcie"), Link("b.gpu", "b.dram", pcie, kind="pcie"),
             Link("a.gpu", "a.cert", net, US), Link("a.cert", "leaf", net, US), Link("b.gpu", "b.cert", net, US),
             Link("b.cert", "leaf", net, US), Link("challenger", "leaf", net, US), Link("world", "leaf", net, US)]
    tick = Assumption("tick", 1e-4, "s", "declared")                # 100 µs ticks
    return Datacenter("dc0", [node("a"), node("b")], fabric, links, Grid(window=4, tick=tick))


def setup(plan: Plan):
    c = trace(lambda x: x ^ 1, x=Input(8))
    p = place(c, datacenter(), "a.gpu", deliver="world", stores=(KV,), x="world")
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


def test_certified_reads_are_local_only_beyond_what_the_boundary_could_carry_in():
    """The bandwidth audits: 4 Gbit read from the KV store within 1 ms. The block's 400 Gbit/s boundary could have
    carried in 0.4 Gbit of it in that time, so at least 3.6 Gbit were read on the block."""
    p, plan, profile = setup(Plan(Interval(0, 16), reads=[Reads("kv", 0.5 * GB, Interval(0, 10))]))
    acc = account(audit(run(p, {"x": 3}, plan), profile), profile)
    a = acc.blocks[0]
    assert acc.sound and a.reads == 0.5 * GB and a.reads_local_at_least == pytest.approx(0.5 * GB - 400e9 * 1e-3)
    late = audit(run(p, {"x": 3}, plan, Deviation(moved={"kv": "b.gpu"})), profile)
    assert not any(isinstance(c, Read) for c in late)
    assert [c.why for c in late if isinstance(c, Refused)] == ["the reads finished after the window"]
