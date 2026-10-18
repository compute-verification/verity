"""Timed challenges bind held bits to a device: the honest holder answers in time, a store kept in host DRAM or on
another node answers late, and a deadline too loose to tell the tiers apart locates the store to no single device."""
from __future__ import annotations

from datacenter import (Answered, Assumption, Block, Challenge, Clock, Datacenter, Device, Input, Interval, Link,
                        Refused, Store, account, audit, occupancy, place, run, trace)

GB = 8e9


def node(n: str) -> Block:
    gpu = Device(f"{n}.gpu", "gpu", rate={"scalar": Assumption(f"{n}.gpu.rate", 1e6, "ops/s")},
                 memory=Assumption(f"{n}.gpu.hbm", 96 * GB, "bits"),
                 bandwidth=Assumption(f"{n}.gpu.hbm_bw", 1.6e12 * 8, "bit/s"))
    dram = Device(f"{n}.dram", "dram", memory=Assumption(f"{n}.dram.size", 512 * GB, "bits"),
                  bandwidth=Assumption(f"{n}.dram.bw", 200e9 * 8, "bit/s"))
    return Block(n, [gpu, dram])


def datacenter() -> Datacenter:
    net = Assumption("uplink", 400e9, "bit/s", "declared")
    pcie = Assumption("pcie5x16", 64e9 * 8, "bit/s")
    fabric = [Device("a.cert", "certifier", trusted=True), Device("b.cert", "certifier", trusted=True),
              Device("leaf", "switch"), Device("challenger", "challenger", trusted=True), Device("world", "outside")]
    links = [Link("a.gpu", "a.dram", pcie, kind="pcie"), Link("b.gpu", "b.dram", pcie, kind="pcie"),
             Link("a.gpu", "a.cert", net, 1e-6), Link("a.cert", "leaf", net, 1e-6), Link("b.gpu", "b.cert", net, 1e-6),
             Link("b.cert", "leaf", net, 1e-6), Link("challenger", "leaf", net, 1e-6), Link("world", "leaf", net, 1e-6)]
    return Datacenter("dc0", [node("a"), node("b")], fabric, links)


CLOCK = Clock(tick=1e-4)                     # 100 µs ticks
KV = Store("kv", 1 * GB, "a.gpu")            # 1 GB of KV pages, registered on a.gpu


def setup(deadline: int):
    c = trace(lambda x: x ^ 1, x=Input(8))
    p = place(c, datacenter(), {"x": "world", "*": "a.gpu"}, deliver="world", stores=(KV,))
    return p, [Challenge("kv", at=0, deadline=deadline)]


def test_the_honest_holder_answers_in_time_and_only_it_could_have():
    p, challenges = setup(deadline=8)        # honest: 6 µs of round trip + 1 GB at 1.6 TB/s ≈ 0.63 ms ≈ 7 ticks
    t = run(p, {"x": 3}, window=4, challenges=challenges, clock=CLOCK)
    claims = audit(t, occupancy(p, 32, 1, window=4))
    assert [c for c in claims if isinstance(c, Answered)] and not [c for c in claims if isinstance(c, Refused)]
    acc = account(claims, p, CLOCK, Interval(0, 100))
    assert acc.located == {"kv": ["a.gpu"]}


def test_host_dram_passing_for_hbm_and_a_store_on_another_node_answer_late():
    from datacenter import Deviation
    p, challenges = setup(deadline=8)
    for elsewhere in ("a.dram", "b.gpu"):
        t = run(p, {"x": 3}, window=4, challenges=challenges, clock=CLOCK, deviation=Deviation(moved={"kv": elsewhere}))
        refused = [c for c in audit(t, occupancy(p, 32, 1, window=4)) if isinstance(c, Refused)]
        assert refused and "past its deadline" in refused[0].why, elsewhere


def test_a_deadline_too_loose_locates_the_store_to_no_single_device():
    p, challenges = setup(deadline=300)      # 30 ms: host DRAM over PCIe (5 ms read + 16 ms PCIe) fits too
    t = run(p, {"x": 3}, window=4, challenges=challenges, clock=CLOCK)
    acc = account(audit(t, occupancy(p, 32, 1, window=4)), p, CLOCK, Interval(0, 1000))
    assert "a.gpu" in acc.located["kv"] and "a.dram" in acc.located["kv"]
