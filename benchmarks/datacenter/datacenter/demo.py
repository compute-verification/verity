"""A night on a rack: `python -m datacenter.demo`.

Two RTX PRO 6000 nodes, each with a BlueField-3 DPU, behind their own certifiers (`examples.rack`). They serve a
two-stage model, one stage a node, whose first stage also reads a registered container image's bytes, and whose BMC
telemetry is delivered out. Each node's certifier releases a salt into it, and each node computes a noised product on
it (PoUW's proof work, which never leaves the node). The verifier's plan wipes the first node's DPU and BMC,
challenges a KV store and opens a read window on it. The demo runs the night honestly and accounts for it, under
compute accounting's salt custody too, then runs it again with an adversary who tries five things at once, each of
which the audit refuses for its own reason.
"""
from __future__ import annotations

from . import (SALT_CUSTODY, Challenge, Deviation, Input, Interval, Plan, Port, ReadWindow, Store, Wipe, account, audit,
               digest, place, run, trace)
from .examples import GB, rack

IMAGE = 0xC0FFEE
INPUTS = {"x": 7, "w1": 3, "w2": 5, "temperature": 45, "s0": 2**511 + 3, "s1": 2**511 + 5}


def model(x, w1, w2, image, temperature, s0, s1):
    h = ((x * w1) ^ image).named("h")
    y = (h * w2 + x).named("y")
    (h * s0).named("p0")
    (y * s1).named("p1")
    return y, temperature


def night(deviation: Deviation | None = None, image: int = IMAGE, assume=()):
    dc = rack(nodes=2)
    c = trace(model, x=Input(32), w1=Input(32, "registered", digest(3, 32)), w2=Input(32, "registered", digest(5, 32)),
              image=Input(32, "registered", digest(IMAGE, 32)), temperature=Input(16, "inside"),
              s0=Input(512, "trusted"), s1=Input(512, "trusted"))
    kv = Store("kv", 1 * GB, "n1.gpu0")
    p = place(c, dc, "n0.gpu0", deliver="world", stores=(kv,), x="world", image="world", temperature="n1.bmc",
              w1="n0.gpu0", h="n0.gpu0", w2="n1.gpu0", y="n1.gpu0", s0="n0.cert", s1="n1.cert", p0="n0.gpu0",
              p1="n1.gpu0")
    plan = Plan(Interval(0, 16), profiles=8,
                wipes=[Wipe("n0.dpu", 195.2 * GB, Interval(0, 2)), Wipe("n0.bmc", 0.88 * GB, Interval(0, 2))],
                challenges=[Challenge("kv", at=0, deadline=8)],
                reads=[ReadWindow("kv", start=0, boundary=10, deadline=12)])
    profile = p.public(plan)
    t = run(p, INPUTS | {"image": image}, plan, deviation)
    return account(audit(t, profile), profile, assume)


def main() -> None:
    print("An honest night:\n")
    print(night())
    print("\nUnder compute accounting's salt custody (unproved):\n")
    for b in night(assume=(SALT_CUSTODY,)).blocks:
        print(f"  {b.block}: k = {b.custody:g} bits, so at least {b.local_at_least:g} of {b.credited:g} credited ran on "
              f"{b.block}")
    print("\nThe same night with an adversary:\n")
    adversary = Deviation(extra={(Port("n0.cert", "egress"), 5): 2},      # covert frames on n0's uplink
                          moved={"kv": "n1.cpu0"},                         # KV kept in host DRAM, not on the GPU
                          failed_wipes={"n0.bmc"},                         # state hidden in the BMC through the wipe
                          down={(Port("n1.cert", "ingress"), 2)})          # a record withheld as "downtime"
    acc = night(adversary, image=0xBADC0DE)                                # and a swapped container image
    for r in acc.refused:
        print(f"  REFUSED {r.on!r}: {r.why}")
    print(f"\n  covert budget with the refused windows charged at line rate: {acc.covert_bits:.3g} bits")
    print(f"  KV reads certified local to n1: {acc.blocks[1].reads_local_at_least:.3g} bits "
          f"(honest night: {night().blocks[1].reads_local_at_least:.3g})")


if __name__ == "__main__":
    main()
