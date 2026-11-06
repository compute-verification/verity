"""A night on a rack: `python -m datacenter.demo`.

Two RTX PRO 6000 nodes, each with a BlueField-3 DPU, behind their own certifiers (`examples.rack`). They serve a
two-stage model, a BF16 linear a node (`models.LINEAR`), whose registered inputs are the weights and the container image
the first node runs, and whose BMC telemetry is delivered out. Each node's certifier releases a salt into it, and each
node computes a noised product on it (`models.DOT`, PoUW's proof work in miniature, which never leaves the node). The
registration also commits the occupancy profiles the night may follow, one for each of 8 ticks its schedule may start
at. The verifier's plan wipes the first node's DPU and BMC, as far as their links fill them, challenges a KV store and
opens a read window on it. The demo runs the night honestly and accounts for it, under the salt custody
premise too, then runs it again with an adversary who tries five things at once, each of which the audit refuses for
its own reason.
"""
from __future__ import annotations

from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Tuple

from . import (SALT_CUSTODY, Challenge, Deviation, Interval, Plan, Port, ReadWindow, StoreEntry, Wipe, account, audit,
               occupancy, place, register, run)
from .examples import GB, rack
from .models import DOT, LINEAR, MAT, VEC, WEIGHTS, X, program, word, words

IMAGE = words("image", 32)
S0, S1 = words("salt0", 32), words("salt1", 32)
KV = StoreEntry("kv", 1 * GB, "n1.gpu0")


def model(B, x, w1, w2, image, temperature, s0, s1):
    h = B.call(LINEAR, x, w1, name="h")
    y = B.call(LINEAR, h, w2, name="y")
    B.call(DOT, h, s0, name="p0")
    B.call(DOT, y, s1, name="p1")
    return tuple_of(y, temperature)


PROGRAM = program("RackNight", {"x": VEC, "w1": MAT, "w2": MAT, "image": VEC, "temperature": word(16), "s0": VEC,
                                "s1": VEC}, Tuple(VEC, word(16)), model, registered=("w1", "w2", "image"))
VALUES = WEIGHTS | {"image": IMAGE}


def placed(registration, start: int = 0):
    return place(PROGRAM, rack(nodes=2), "n0.gpu0", registration=registration, deliver="world", stores=(KV,),
                 start=start, x="world", image="n0.gpu0", temperature="n1.bmc", h="n0.gpu0", y="n1.gpu0", s0="n0.cert",
                 s1="n1.cert", p0="n0.gpu0", p1="n1.gpu0")


#: the night's schedule may start at any of 8 ticks: the registration commits the 8 occupancy profiles that gives
REGISTRATION = register(PROGRAM, VALUES, profiles=[occupancy(placed(register(PROGRAM, VALUES), s)) for s in range(8)])


def night(deviation: Deviation | None = None, assume=()):
    p = placed(REGISTRATION)
    plan = Plan(Interval(0, 16), profiles=REGISTRATION,
                wipes=[Wipe("n0.dpu", 195.2 * GB, Interval(0, 2)), Wipe("n0.bmc", 0.88 * GB, Interval(0, 2))],
                challenges=[Challenge("kv", at=0, deadline=8)],
                reads=[ReadWindow("kv", start=0, boundary=10, deadline=12)])
    profile = p.public(plan)
    t = run(p, {"x": X, "temperature": 45, "s0": S0, "s1": S1}, plan, deviation)
    return account(audit(t, profile), profile, assume)


def main() -> None:
    print("An honest night:\n")
    print(night())
    print("\nUnder the salt custody premise (unproved):\n")
    for b in night(assume=(SALT_CUSTODY,)).blocks:
        print(f"  {b.block}: k = {b.custody:g} bits, so at least {b.local_at_least:g} of {b.credited:g} credited ran on "
              f"{b.block}")
    print("\nThe same night with an adversary:\n")
    adversary = Deviation(extra={(Port("n0.cert", "egress"), 5): 2},      # covert frames on n0's uplink
                          moved={"kv": "n1.cpu0"},                         # KV kept in host DRAM, not on the GPU
                          failed_wipes={"n0.bmc"},                         # state hidden in the BMC through the wipe
                          down={(Port("n1.cert", "ingress"), 2)},          # a record withheld as "downtime"
                          swapped={"image": words("other image", 32)})    # and a swapped container image
    acc = night(adversary)
    for r in acc.refused:
        print(f"  REFUSED {r.on!r}: {r.why}")
    print(f"\n  covert budget with the refused windows charged at line rate: {acc.covert_bits:.3g} bits")
    print(f"  KV reads certified local to n1: {acc.blocks[1].reads_local_at_least:.3g} bits "
          f"(honest night: {night().blocks[1].reads_local_at_least:.3g})")


if __name__ == "__main__":
    main()
