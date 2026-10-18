"""The walkthrough's code, section by section, so every snippet the guide shows runs as shown.

The guide is the Notion page "Accounting for a datacenter, from a circuit up"; each test here is one of its sections,
in order, and `python tests/test_walkthrough.py` prints what the guide prints.
"""
from __future__ import annotations

from datacenter import (Assumption, Block, Clock, Credit, Datacenter, Deviation, Device, Fleet, Input, Interval, Link,
                        Offramp, Refused, Wipe, account, audit, commit, crossings, occupancy, place, run, schedulable,
                        trace)


INPUTS = {"x": 7, "w1": 3, "w2": 5}
#: stage h on block a, stage y on block b; the request comes from the world
PIPELINE = {"x": "world", "w1": "a.gpu", "h": "a.gpu", "w2": "b.gpu", "y": "b.gpu", "*": "a.gpu"}


# §1 A computation is a circuit
def layer(x, w1, w2):
    h = ((x * w1) ^ 0x5A).named("h")
    return (h * w2 + x).named("y")


def section_1():
    c = trace(layer, x=Input(16), w1=Input(16), w2=Input(16))
    print(c)
    print(c.evaluate({"x": 7, "w1": 3, "w2": 5})[c.outputs[0]])
    return c


# §2 Inputs are everything the circuit doesn't determine
def section_2():
    c = trace(layer, x=Input(16, source="outside"), w1=Input(16, source="registered"), w2=Input(16, source="registered"))
    for v in c.inputs:
        print(v, c.gates[v.id].input.source)
    return c


# §3 Values live under commitments
def section_3(c):
    values = c.evaluate({"x": 7, "w1": 3, "w2": 5})
    commitment, salt = commit([values[i] for i in range(len(values))], [g.bits for g in c.gates])
    print(commitment.digest.hex()[:32], commitment.count)


# §4 Hardware is a declaration
def gpu(name):
    return Device(name, "gpu",
                  rate={"scalar": Assumption(f"{name}.rate", 1e6, "ops/s")},
                  memory=Assumption(f"{name}.hbm", 96 * 8e9, "bits"),
                  residual=Assumption(f"{name}.residual", 8e6, "bits", "measured"))


def section_4():
    up = Assumption("uplink", 400e9, "bit/s", "declared")
    dc = Datacenter(
        "dc0",
        blocks=[Block("a", [gpu("a.gpu")]), Block("b", [gpu("b.gpu")])],
        fabric=[Device("a.cert", "certifier", trusted=True), Device("b.cert", "certifier", trusted=True),
                Device("leaf", "switch"), Device("world", "outside")],
        links=[Link("a.gpu", "a.cert", up), Link("a.cert", "leaf", up),
               Link("b.gpu", "b.cert", up), Link("b.cert", "leaf", up),
               Link("world", "leaf", up)])
    print(dc.validate())
    dc.links.append(Link("a.gpu", "leaf", up))
    print(dc.validate())
    dc.links.pop()
    return dc


# §5 Placing the circuit
def section_5(c, dc):
    p = place(c, dc, PIPELINE, deliver="world")
    print(p.leaf)
    print(p.step)
    for x in crossings(p):
        print(x.value, x.src, "->", x.dst, [f"{l.a}-{l.b}" for l in x.path])
    return p


# §6 What the network may carry
def section_6(p):
    profile = occupancy(p, frame_bits=32, bucket=1, window=4)
    print(profile)
    return profile


# §7–8 Time, and a run
def section_8(p):
    t = run(p, INPUTS, window=4, wipes=[Wipe("b.gpu", Interval(0, 2))])
    for r in t.records:
        print(r.link, r.window, r.counts)
    return t


# §9 The audit: the logical layer
def section_9(t, profile, p):
    claims = audit(t, profile, syncs=128)
    for x in claims:
        print(x)
    cheat = run(p, INPUTS, window=4, deviation=Deviation(extra={(("a.gpu", "a.cert"), 2): 1}))
    print([x for x in audit(cheat, profile, syncs=128) if isinstance(x, Refused)])
    return claims


# §10 The account: the physical layer
def section_10(claims, p):
    acc = account(claims, p, Clock(tick=1e-3), Interval(0, 16))
    print(acc)
    return acc


# §11 Telemetry on an off-ramp
def section_11(c, dc):
    temps = Offramp("telemetry", "b.gpu", "world", bits=64, windows=2)
    p = place(c, dc, PIPELINE, deliver="world", offramps=(temps,))
    t = run(p, INPUTS, window=4)
    acc = account(audit(t, occupancy(p, 32, 1, window=4), syncs=128), p, Clock(1e-3), Interval(0, 16))
    print(acc.offramp_bits, acc.covert_bits)
    return acc


# §12 Two audits, one device
def section_12():
    capacity = {("a.gpu", "compute"): 1e6}
    pouw = Credit("a.gpu", "compute", Interval(0, 10), 8e6)
    other = Credit("a.gpu", "compute", Interval(2, 8), 4e6)
    print(schedulable([pouw, other], capacity))


# §13 Many datacenters
def section_13(dc):
    def renamed(prefix):
        r = lambda s: f"{prefix}.{s}"  # noqa: E731
        return Datacenter(prefix,
                          [Block(r(b.name), [gpu(r(d.name)) for d in b.devices]) for b in dc.blocks],
                          [Device(r(d.name), d.kind, trusted=d.trusted) for d in dc.fabric if d.kind != "outside"],
                          [Link(r(l.a), r(l.b), l.capacity) for l in dc.links if "world" not in (l.a, l.b)])
    east, west = renamed("east"), renamed("west")
    west.fabric.append(Device("world", "outside"))
    west.links.append(Link("world", "west.leaf", Assumption("isp", 100e9, "bit/s")))
    wan = Link("east.a.cert", "west.a.cert", Assumption("wan", 100e9, "bit/s", "declared"), kind="wan")
    fleet = Fleet([east, west], [wan])
    print(fleet.validate())
    c = trace(layer, x=Input(16), w1=Input(16, source="registered"), w2=Input(16, source="registered"))
    p = place(c, fleet.merged(), {"x": "world", "w1": "west.a.gpu", "h": "west.a.gpu", "w2": "east.b.gpu", "y": "east.b.gpu", "*": "west.a.gpu"}, deliver="world")
    print([f"{l.a}-{l.b}" for x in crossings(p) if x.value.name == "h" for l in x.path])


def test_the_walkthrough_runs_as_shown(capsys):
    c = section_1()
    section_2()
    c = trace(layer, x=Input(16, source="outside"), w1=Input(16, source="registered"), w2=Input(16, source="registered"))
    section_3(c)
    dc = section_4()
    p = section_5(c, dc)
    profile = section_6(p)
    t = section_8(p)
    claims = section_9(t, profile, p)
    acc = section_10(claims, p)
    assert acc.sound and sum(b.credited for b in acc.blocks) == c.boolean_size
    assert all(b.credited > 0 for b in acc.blocks)        # each stage is credited to the block it ran on
    acc11 = section_11(c, dc)
    assert acc11.offramp_bits == {"telemetry": 128}
    section_12()
    section_13(dc)
    out = capsys.readouterr().out
    assert "passes no certifier" in out and "occupancy differs" in out


if __name__ == "__main__":
    c = section_1(); print("---")
    section_2(); print("---")
    c = trace(layer, x=Input(16, source="outside"), w1=Input(16, source="registered"), w2=Input(16, source="registered"))
    section_3(c); print("---")
    dc = section_4(); print("---")
    p = section_5(c, dc); print("---")
    profile = section_6(p); print("---")
    t = section_8(p); print("---")
    claims = section_9(t, profile, p); print("---")
    section_10(claims, p); print("---")
    section_11(c, dc); print("---")
    section_12(); print("---")
    section_13(dc)
