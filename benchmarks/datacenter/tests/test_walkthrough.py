"""The walkthrough's code, section by section, so every snippet the guide shows runs as shown.

The guide is the Notion page "Accounting for a datacenter, from a circuit up"; each function here is one of its sections,
in order, and `python tests/test_walkthrough.py` prints what the guide prints.
"""
from __future__ import annotations

from datacenter import (FLEET_CLOSED, Append, Assumption, Block, Cadence, Carried, Challenge, Credit, Datacenter,
                        Deviation, Device, Fleet, Input, Interval, Ledger, Link, Maintenance, Plan, Port, ReadWindow,
                        Refused, Release, Store, Wipe, account, audit, commit, crossings, digest, double_credit,
                        occupancy, place, run, trace)
from datacenter.examples import GB

INPUTS = {"x": 7, "w1": 3, "w2": 5}
IMAGE = 0xC0DE


# §1 A computation is a circuit
def layer(x, w1, w2):
    h = ((x * w1) ^ 0x5A).named("h")
    return (h * w2 + x).named("y")


def section_1():
    c = trace(layer, x=Input(16), w1=Input(16), w2=Input(16))
    print(c)
    print(c.evaluate({"x": 7, "w1": 3, "w2": 5})[c.outputs[0]])


# §2 Inputs are everything the circuit doesn't determine
def section_2():
    c = trace(layer, x=Input(16, "outside"),
              w1=Input(16, "registered", digest(3, 16)),
              w2=Input(16, "registered", digest(5, 16)))
    for v in c.inputs:
        print(v, v.source)
    return c


# §3 Values live under commitments
def section_3(c):
    values = c.evaluate({"x": 7, "w1": 3, "w2": 5})
    commitment, salt = commit(c, values)
    print(commitment.count, len(commitment.digest), len(salt))


# §4 Hardware is a declaration
def gpu(name):
    return Device(name, "gpu",
                  rate={"gate": Assumption(f"{name}.gates", 1e6, "gate/s")},
                  memory=Assumption(f"{name}.hbm", 96 * GB, "bits"),
                  residual=Assumption(f"{name}.residual", 8e6, "bits", "measured"))


def section_4():
    up = Assumption("uplink", 400e9, "bit/s", "declared")
    dc = Datacenter(
        "dc0",
        blocks=[Block("a", [gpu("a.gpu")]), Block("b", [gpu("b.gpu")])],
        fabric=[Device("a.cert", "certifier"), Device("b.cert", "certifier"), Device("leaf", "switch"),
                Device("challenger", "challenger"), Device("world", "outside")],
        links=[Link("a.gpu", "a.cert", up), Link("a.cert", "leaf", up),
               Link("b.gpu", "b.cert", up), Link("b.cert", "leaf", up),
               Link("challenger", "leaf", up), Link("world", "leaf", up)],
        cadence=Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                        tick=Assumption("tick", 1e-3, "s", "declared")))
    print(dc.validate())
    dc.links.append(Link("a.gpu", "leaf", up))
    print(dc.validate())
    dc.links.pop()
    print([d for d, device in dc.devices.items() if device.trusted])
    return dc


# §5 Placing the circuit
def section_5(c, dc):
    p = place(c, dc, "a.gpu", deliver="world", x="world", h="a.gpu", w2="b.gpu", y="b.gpu")
    for x in crossings(p):
        print(x.value, x.src, "->", x.dst, "read at tick", x.at)
    return p


# §6 What the network may carry
def section_6(p):
    for port, row in occupancy(p).items():
        print(port, row)


# §7 Time, and the verifier's plan
def section_7(p):
    plan = Plan(Interval(0, 16), wipes=[Wipe("b.gpu", 96 * GB - 8e6, Interval(0, 2))])
    profile = p.public(plan)
    print(profile.work, list(profile.windows))
    return plan, profile


# §8 A run, and what the trusted devices record
def section_8(p, plan):
    t = run(p, INPUTS, plan)
    for r in t.records:
        if any(r.counts):
            print(r.port, r.window, r.counts)
    print(len(t.records), "records in all")
    return t


# §9 The audit: the logical layer
def section_9(p, plan, profile, t):
    claims = audit(t, profile)
    carried = [x for x in claims if isinstance(x, Carried)]
    print(len(carried), "port-windows carried only committed crossings, at", carried[0].bits, "bits of timing advice each")
    for x in claims:
        if not isinstance(x, Carried):
            print(x)
    cheat = run(p, INPUTS, plan, Deviation(extra={(Port("a.cert", "egress"), 2): 1}))
    for x in audit(cheat, profile):
        if isinstance(x, Refused):
            print(x)
    return claims


# §10 The account: the physical layer
def section_10(claims, profile):
    acc = account(claims, profile)
    print(acc)
    print(account(claims, profile, assume=(FLEET_CLOSED,)).blocks[0].local_at_least)
    return acc


# §11 Where are the bits?
def section_11():
    us = Assumption("hop", 1e-6, "s", "measured")
    net = Assumption("uplink", 400e9, "bit/s", "declared")
    hbm = Device("a.gpu", "gpu", memory=Assumption("a.gpu.hbm", 96 * GB, "bits"),
                 bandwidth=Assumption("a.gpu.hbm_bw", 1.6e12 * 8, "bit/s"))
    dram = Device("a.dram", "dram", memory=Assumption("a.dram.size", 512 * GB, "bits"),
                  bandwidth=Assumption("a.dram.bw", 200e9 * 8, "bit/s"))
    node = Datacenter("dc1", [Block("a", [hbm, dram])],
                      [Device("a.cert", "certifier"), Device("challenger", "challenger"), Device("leaf", "switch"),
                       Device("world", "outside")],
                      [Link("a.gpu", "a.dram", Assumption("pcie5x16", 64e9 * 8, "bit/s"), kind="pcie"),
                       Link("a.gpu", "a.cert", net, us), Link("a.cert", "leaf", net, us),
                       Link("challenger", "leaf", net, us), Link("world", "leaf", net, us)],
                      Cadence(buckets=4, tick=Assumption("tick", 1e-4, "s", "declared")))
    kv = Store("kv", 1 * GB, "a.gpu")
    q = place(trace(lambda x: x ^ 1, x=Input(8)), node, "a.gpu", deliver="world", stores=(kv,), x="world")
    for deadline in (8, 300):
        plan = Plan(Interval(0, 1000), challenges=[Challenge("kv", at=0, deadline=deadline)])
        print(deadline, account(audit(run(q, {"x": 3}, plan), q.public(plan)), q.public(plan)).located)
    plan = Plan(Interval(0, 1000), challenges=[Challenge("kv", at=0, deadline=8)])
    late = run(q, {"x": 3}, plan, Deviation(moved={"kv": "a.dram"}))
    print([x for x in audit(late, q.public(plan)) if isinstance(x, Refused)])
    reads = Plan(Interval(0, 16), reads=[ReadWindow("kv", start=0, boundary=10, deadline=12)])
    a = account(audit(run(q, {"x": 3}, reads), q.public(reads)), q.public(reads)).blocks[0]
    print(a.reads, a.reads_local_at_least)
    sampled = Plan(Interval(0, 1000), challenges=[Challenge("kv", at=0, deadline=1, sample=4096 * 8)])
    print(account(audit(run(q, {"x": 3}, sampled), q.public(sampled)), q.public(sampled)).located)
    changes = [Append("kv", at=0, first=0, count=1000, coin_at=1, resident_at=4), Release("kv", first=0, count=1000, at=12)]
    grows = Plan(Interval(0, 16), changes=changes, lifetime=3, deferred=True)
    print(grows.bits(kv, 2), grows.bits(kv, 6), grows.held(kv, Interval(0, 16)))


# §12 Telemetry and container images
def served(x, w1, w2, image, temperature):
    h = ((x * w1) ^ image).named("h")
    return (h * w2 + x).named("y"), temperature


def section_12(dc):
    c2 = trace(served, x=Input(16), w1=Input(16, "registered", digest(3, 16)), w2=Input(16, "registered", digest(5, 16)),
               image=Input(16, "registered", digest(IMAGE, 16)), temperature=Input(16, "inside"))
    p2 = place(c2, dc, "a.gpu", deliver="world", x="world", image="world", temperature="b.gpu",
               h="a.gpu", w2="b.gpu", y="b.gpu")
    plan = Plan(Interval(0, 16))
    acc = account(audit(run(p2, INPUTS | {"image": IMAGE, "temperature": 45}, plan), p2.public(plan)), p2.public(plan))
    print(acc.input_bits, acc.covert_bits)
    swapped = audit(run(p2, INPUTS | {"image": 0xBAD0, "temperature": 45}, plan), p2.public(plan))
    print([x for x in swapped if isinstance(x, Refused)])
    return acc


# §13 Two audits, one device
def section_13():
    capacity = {("a.gpu", "compute"): 1e6}
    pouw = Credit("a.gpu", "compute", Interval(0, 10), 8e6)
    other = Credit("a.gpu", "compute", Interval(2, 8), 4e6)
    print(double_credit([pouw, other], capacity))


# §14 Many nights
def section_14(p):
    ledger = Ledger()
    for night in range(3):
        wipes = [Wipe(d, 96 * GB - 8e6, Interval(0, 2)) for d in ("a.gpu", "b.gpu")] if night else []
        plan = Plan(Interval(0, 16), wipes=wipes)
        ledger.add(account(audit(run(p, INPUTS, plan), p.public(plan)), p.public(plan)))
    print(ledger)
    serviced = Plan(Interval(0, 16), maintenance=[Maintenance("b", Interval(12, 16))])
    off = Deviation(down={(Port("b.cert", "egress"), 3), (Port("b.cert", "ingress"), 3)})
    acc = account(audit(run(p, INPUTS, serviced, off), p.public(serviced)), p.public(serviced))
    print(acc.sound, acc.serviced, acc.blocks[1].carried)
    return ledger


# §15 Many datacenters
def section_15(c, dc):
    wan = Link("east.a.cert", "west.a.cert", Assumption("wan", 100e9, "bit/s", "declared"), kind="wan")
    fleet = Fleet([dc.prefixed("east"), dc.prefixed("west")], [wan])
    print(fleet.validate())
    p = place(c, fleet.merged(), "west.a.gpu", deliver="world", x="world", h="west.a.gpu", w2="east.b.gpu", y="east.b.gpu")
    print(", ".join(f"{l.a}-{l.b}" for x in crossings(p) if x.value.name == "h" for l in x.path))


# §16 A night on a real rack
def section_16():
    from datacenter.demo import main
    main()


def walkthrough(between=lambda n: None):
    between(1); section_1()
    between(2); c = section_2()
    between(3); section_3(c)
    between(4); dc = section_4()
    between(5); p = section_5(c, dc)
    between(6); section_6(p)
    between(7); plan, profile = section_7(p)
    between(8); t = section_8(p, plan)
    between(9); claims = section_9(p, plan, profile, t)
    between(10); acc = section_10(claims, profile)
    between(11); section_11()
    between(12); acc12 = section_12(dc)
    between(13); section_13()
    between(14); ledger = section_14(p)
    between(15); section_15(c, dc)
    between(16); section_16()
    return c, acc, acc12, ledger


def test_the_walkthrough_runs_as_shown(capsys):
    c, acc, acc12, ledger = walkthrough()
    assert acc.sound and sum(b.credited for b in acc.blocks) == c.work
    assert all(b.credited > 0 for b in acc.blocks)        # each stage is credited to the block it ran on
    assert acc12.sound and acc12.input_bits == {"x": 16, "temperature": 16}
    assert ledger.sound and len(ledger.epochs) == 3
    out = capsys.readouterr().out
    for shown in ("passes no certifier", "occupancy differs", "past its deadline", "registered digest", "REFUSED"):
        assert shown in out, shown


if __name__ == "__main__":
    walkthrough(lambda n: print(f"--- §{n}"))
