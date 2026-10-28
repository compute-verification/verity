"""The walkthrough's code, section by section, so every snippet the guide shows runs as shown.

The guide is the Notion page "Accounting for a datacenter, from a circuit up"; each function here is one of its sections,
in order, and `python tests/test_walkthrough.py` prints what the guide prints.
"""
from __future__ import annotations

from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean.gemm import GEMM_AMPERE, Gemm
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple

from datacenter import (FLEET_CLOSED, Accounts, Append, Assumption, Cadence, Carried, Challenge, Credit, Datacenter,
                        Deviation, DeviceEntry, Fleet, Interval, LinkEntry, Maintenance, Plan, Port, ReadWindow,
                        Refused, Release, StoreEntry, Wipe, Zone, account, ands, audit, commit, crossings,
                        double_credit, evaluate, names, occupancy, place, read, register, run, values, work)
from datacenter.examples import GB
from datacenter.models import WEIGHTS, X, words

# §1 A computation is a Program
BF16 = Array(16, BIT)                                   # a BF16 word: its 16 bits, least significant first
VEC, MAT = Array(32, BF16), Array(32, Array(32, BF16))
LINEAR = bind(Gemm, K=32, N=32, **GEMM_AMPERE)          # x[32] @ W[32, 32]^T on the BF16 tensor-core step, on bits


def layer(B, S, x, w1, w2):
    h = B.call(LINEAR, x, w1, name="h")
    return B.call(LINEAR, h, w2, name="y")


LAYER = CompositeDefinition("Layer", 1, (), lambda S: ((("x", VEC), ("w1", MAT), ("w2", MAT)), VEC), layer,
                            register=False)


def section_1():
    program = Program(LAYER, registered=("w1", "w2"))
    print(program.gates, "gates:", ", ".join(f"{n} ({nd.gates})" for n, nd in zip(names(program), program.root.body.nodes)))
    for v in values(program):
        print(v, "read by", [names(program)[r] for r in v.reads], *(["and returned"] if v.output else []))
    print(work(program), ands(program))
    y = values(program)[-1]
    print([hex(w) for w in read(program, evaluate(program, {"x": X} | WEIGHTS), y)[:4]])
    return program


# §2 Inputs are everything the Program doesn't determine
def section_2(program):
    print(list(zip(names(program), program.bindings)))
    registration = register(program, WEIGHTS)
    print(len(registration.root), "bytes of registration root")
    return registration


# §3 Values live under commitments
def section_3(program):
    values_ = evaluate(program, {"x": X} | WEIGHTS)
    commitment = commit(program, values_)
    print(len(commitment.rows), "rows,", len(commitment.root), "bytes of root,", len(commitment.salts[0]), "bytes of salt each")


# §4 Hardware is a declaration
def gpu(name):
    return DeviceEntry(name, "gpu",
                       rate={"mac:bf16": Assumption(f"{name}.bf16", 1e9, "MAC/s"),
                             "cuda:gate": Assumption(f"{name}.gates", 1e10, "gate/s")},
                       memory=Assumption(f"{name}.hbm", 96 * GB, "bits"),
                       residual=Assumption(f"{name}.residual", 8e6, "bits", "measured"),
                       write=Assumption(f"{name}.write", 4e14, "bit/s"))


def section_4():
    up = Assumption("uplink", 400e9, "bit/s", "declared")
    dc = Datacenter(
        "dc0",
        blocks=[Zone("a", [gpu("a.gpu")]), Zone("b", [gpu("b.gpu")])],
        fabric=[DeviceEntry("a.cert", "certifier"), DeviceEntry("b.cert", "certifier"), DeviceEntry("leaf", "switch"),
                DeviceEntry("challenger", "challenger"), DeviceEntry("world", "outside")],
        links=[LinkEntry("a.gpu", "a.cert", up), LinkEntry("a.cert", "leaf", up),
               LinkEntry("b.gpu", "b.cert", up), LinkEntry("b.cert", "leaf", up),
               LinkEntry("challenger", "leaf", up), LinkEntry("world", "leaf", up)],
        cadence=Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                        tick=Assumption("tick", 1e-3, "s", "declared")))
    print(dc.validate())
    dc.links.append(LinkEntry("a.gpu", "leaf", up))
    print(dc.validate())
    dc.links.pop()
    print([d for d, device in dc.devices.items() if device.trusted])
    return dc


# §5 Placing the Program
def section_5(program, registration, dc):
    p = place(program, dc, "a.gpu", registration=registration, deliver="world", x="world", h="a.gpu", y="b.gpu")
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
    print(profile.work, profile.units, profile.partition, list(profile.windows))
    return plan, profile


# §8 A run, and what the trusted devices record
def section_8(p, plan):
    t = run(p, {"x": X}, plan)
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
    cheat = run(p, {"x": X}, plan, Deviation(extra={(Port("a.cert", "egress"), 2): 1}))
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
def section_11(program, registration):
    us = Assumption("hop", 1e-6, "s", "measured")
    net = Assumption("uplink", 400e9, "bit/s", "declared")
    hbm = DeviceEntry("a.gpu", "gpu", memory=Assumption("a.gpu.hbm", 96 * GB, "bits"),
                      bandwidth=Assumption("a.gpu.hbm_bw", 1.6e12 * 8, "bit/s"))
    dram = DeviceEntry("a.dram", "dram", memory=Assumption("a.dram.size", 512 * GB, "bits"),
                       bandwidth=Assumption("a.dram.bw", 200e9 * 8, "bit/s"))
    node = Datacenter("dc1", [Zone("a", [hbm, dram])],
                      [DeviceEntry("a.cert", "certifier"), DeviceEntry("challenger", "challenger"),
                       DeviceEntry("leaf", "switch"), DeviceEntry("world", "outside")],
                      [LinkEntry("a.gpu", "a.dram", Assumption("pcie5x16", 64e9 * 8, "bit/s"), kind="pcie"),
                       LinkEntry("a.gpu", "a.cert", net, us), LinkEntry("a.cert", "leaf", net, us),
                       LinkEntry("challenger", "leaf", net, us), LinkEntry("world", "leaf", net, us)],
                      Cadence(buckets=4, tick=Assumption("tick", 1e-4, "s", "declared")))
    kv = StoreEntry("kv", 1 * GB, "a.gpu")
    q = place(program, node, "a.gpu", registration=registration, deliver="world", stores=(kv,), x="world")
    for deadline in (8, 300):
        plan = Plan(Interval(0, 1000), challenges=[Challenge("kv", at=0, deadline=deadline)])
        print(deadline, account(audit(run(q, {"x": X}, plan), q.public(plan)), q.public(plan)).located)
    plan = Plan(Interval(0, 1000), challenges=[Challenge("kv", at=0, deadline=8)])
    late = run(q, {"x": X}, plan, Deviation(moved={"kv": "a.dram"}))
    print([x for x in audit(late, q.public(plan)) if isinstance(x, Refused)])
    reads = Plan(Interval(0, 16), reads=[ReadWindow("kv", start=0, boundary=10, deadline=12)])
    a = account(audit(run(q, {"x": X}, reads), q.public(reads)), q.public(reads)).blocks[0]
    print(a.reads, a.reads_local_at_least)
    sampled = Plan(Interval(0, 1000), challenges=[Challenge("kv", at=0, deadline=1, sample=4096 * 8)])
    print(account(audit(run(q, {"x": X}, sampled), q.public(sampled)), q.public(sampled)).located)
    changes = [Append("kv", at=0, first=0, count=1000, coin_at=1, resident_at=4), Release("kv", first=0, count=1000, at=12)]
    grows = Plan(Interval(0, 16), changes=changes, lifetime=3, deferred=True)
    print(grows.bits(kv, 2), grows.bits(kv, 6), grows.held(kv, Interval(0, 16)))


# §12 Telemetry and container images
def served(B, S, x, w1, w2, image, temperature):
    h = B.call(LINEAR, x, w1, name="h")
    return tuple_of(B.call(LINEAR, h, w2, name="y"), temperature)


SERVED = CompositeDefinition("Served", 1, (), lambda S: ((("x", VEC), ("w1", MAT), ("w2", MAT), ("image", VEC),
                                                         ("temperature", Array(16, BIT))), Tuple(VEC, Array(16, BIT))), served,
                             register=False)


def section_12(dc):
    program = Program(SERVED, registered=("w1", "w2", "image"))
    registration = register(program, WEIGHTS | {"image": words("image", 32)})
    p = place(program, dc, "a.gpu", registration=registration, deliver="world", x="world", image="a.gpu",
              temperature="b.gpu", h="a.gpu", y="b.gpu")
    plan = Plan(Interval(0, 16))
    acc = account(audit(run(p, {"x": X, "temperature": 45}, plan), p.public(plan)), p.public(plan))
    print(acc.input_bits, acc.covert_bits)
    swapped = audit(run(p, {"x": X, "temperature": 45}, plan, Deviation(swapped={"image": words("other", 32)})),
                    p.public(plan))
    print([x for x in swapped if isinstance(x, Refused)])
    return acc


# §13 Two audits, one device
def section_13():
    capacity = {("a.gpu", "compute"): 1e6}
    pouw = Credit("a.gpu", "compute", Interval(0, 10), 8e6)
    other = Credit("a.gpu", "compute", Interval(2, 8), 4e6)
    print(double_credit([pouw, other], capacity))


# §14 Many nights
def section_14(program, registration, dc, p):
    ledger = Accounts()
    for night in range(3):
        start = 16 * night
        tonight = place(program, dc, "a.gpu", registration=registration, deliver="world", x="world", h="a.gpu",
                        y="b.gpu", start=start)
        wipes = [Wipe(d, 96 * GB - 8e6, Interval(start, start + 2)) for d in ("a.gpu", "b.gpu")] if night else []
        plan = Plan(Interval(start, start + 16), wipes=wipes)
        ledger.add(account(audit(run(tonight, {"x": X}, plan), tonight.public(plan)), tonight.public(plan)))
    print(ledger)
    print(ledger.custody(sum(16 * len(w) for w in WEIGHTS.values())))
    print(Accounts([ledger.epochs[1], ledger.epochs[1]]).joint())
    serviced = Plan(Interval(0, 16), maintenance=[Maintenance("b", Interval(12, 16))])
    off = Deviation(down={(Port("b.cert", "egress"), 3), (Port("b.cert", "ingress"), 3)})
    acc = account(audit(run(p, {"x": X}, serviced, off), p.public(serviced)), p.public(serviced))
    print(acc.sound, acc.serviced, acc.blocks[1].carried)
    return ledger


# §15 Many datacenters
def section_15(program, registration, dc):
    wan = LinkEntry("east.leaf", "west.leaf", Assumption("wan", 100e9, "bit/s", "declared"), kind="wan")
    fleet = Fleet([dc.prefixed("east"), dc.prefixed("west")], [wan])
    print(fleet.validate())
    p = place(program, fleet.merged(), "west.a.gpu", registration=registration, deliver="world", x="world",
              h="west.a.gpu", y="east.b.gpu")
    print(", ".join(f"{l.a}-{l.b}" for x in crossings(p) if x.value.name == "h" for l in x.path))


# §16 A night on a real rack
def section_16():
    from datacenter.demo import main
    main()


def walkthrough(between=lambda n: None):
    between(1); program = section_1()
    between(2); registration = section_2(program)
    between(3); section_3(program)
    between(4); dc = section_4()
    between(5); p = section_5(program, registration, dc)
    between(6); section_6(p)
    between(7); plan, profile = section_7(p)
    between(8); t = section_8(p, plan)
    between(9); claims = section_9(p, plan, profile, t)
    between(10); acc = section_10(claims, profile)
    between(11); section_11(program, registration)
    between(12); acc12 = section_12(dc)
    between(13); section_13()
    between(14); ledger = section_14(program, registration, dc, p)
    between(15); section_15(program, registration, dc)
    between(16); section_16()
    return program, acc, acc12, ledger


def test_the_walkthrough_runs_as_shown(capsys):
    program, acc, acc12, ledger = walkthrough()
    assert acc.sound and sum(b.credited for b in acc.blocks) == sum(ands(program).values())
    assert all(b.credited > 0 for b in acc.blocks)        # each stage is credited to the block it ran on
    assert acc.export_bits == 1024 and acc.uncovered == ["world-leaf"]   # h and y leave their blocks, uncharged
    assert acc12.sound and acc12.input_bits == {"x": 512, "temperature": 16}
    assert ledger.sound and len(ledger.epochs) == 3 and ledger.custody(32768).share < 0.1
    out = capsys.readouterr().out
    for shown in ("passes no certifier", "occupancy differs", "past its deadline", "differ from the registration",
                  "REFUSED"):
        assert shown in out, shown


if __name__ == "__main__":
    walkthrough(lambda n: print(f"--- §{n}"))
