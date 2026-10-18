"""A night on a rack: `python -m datacenter.demo`.

Two RTX PRO 6000 nodes, each with a BlueField-3 DPU, behind their own certifiers (`examples.rack`). They serve a
two-stage model, one stage a node. The night also pulls a registered image, emits telemetry, holds a KV store that the
challenger checks, and wipes the first node's DPU and BMC. The demo runs it honestly and accounts for it, then
runs it again with an adversary who tries five things at once, each of which the audit refuses for its own reason.
"""
from __future__ import annotations

from . import (Challenge, Clock, Deviation, Input, Interval, ReadAudit, Refused, Store, Stream, Wipe, account, audit,
               occupancy, place, run, trace)
from .examples import GB, rack


def model(x, w1, w2):
    h = ((x * w1) ^ 0x5A5A).named("h")
    return (h * w2 + x).named("y")


def night(deviation: Deviation | None = None):
    dc = rack(nodes=2)
    c = trace(model, x=Input(32), w1=Input(32, source="registered"), w2=Input(32, source="registered"))
    image = Stream("image", "world", "n0.gpu0", bits=4096, windows=1, digest=bytes(64))
    telemetry = Stream("telemetry", "n1.bmc", "world", bits=256, windows=4)
    kv = Store("kv", 1 * GB, "n1.gpu0")
    p = place(c, dc, {"x": "world", "w1": "n0.gpu0", "h": "n0.gpu0", "w2": "n1.gpu0", "y": "n1.gpu0", "*": "n0.gpu0"},
              deliver="world", streams=(image, telemetry), stores=(kv,))
    clock = Clock(tick=1e-4)
    t = run(p, {"x": 7, "w1": 3, "w2": 5}, window=4, clock=clock,
            wipes=[Wipe("n0.dpu", Interval(0, 2)), Wipe("n0.bmc", Interval(0, 2))],
            challenges=[Challenge("kv", at=0, deadline=8)],
            reads=[ReadAudit("kv", Interval(0, 10), 0.5 * GB)],
            deviation=deviation)
    claims = audit(t, occupancy(p, frame_bits=32, bucket=1, window=4), syncs=128, profiles=8)
    return account(claims, p, clock, Interval(0, 16), op="scalar")


def main() -> None:
    print("An honest night:\n")
    print(night())
    print("\nThe same night with an adversary:\n")
    adversary = Deviation(extra={(("n0.dpu", "n0.cert"), 1): 2},     # covert frames on n0's uplink
                          tampered={"image"},                         # a swapped container image
                          moved={"kv": "n1.dram"},                    # KV kept in host DRAM, not on the GPU
                          failed_wipes={"n0.bmc"},                    # state hidden in the BMC through the wipe
                          down={(("n1.dpu", "n1.cert"), 2)})          # a record withheld as "downtime"
    acc = night(adversary)
    for r in acc.refused:
        print(f"  REFUSED {r.what}: {r.why}")


if __name__ == "__main__":
    main()
