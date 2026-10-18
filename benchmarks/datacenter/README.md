# datacenter

A datacenter as Verity accounts for it, in a small Python API: a circuit, the hardware it runs on, the binding between
them, a run with what the trusted devices record, the audit's claims, and the account.

The accounting layer has two layers. The **logical layer** is where the cryptography happens: `logical` (a circuit
traced from Python, its values and their commitment) and `claims` (`audit`: what records and commitments establish
about the prover system, with no device premise). The **physical layer** interprets those claims with declared
hardware, named assumptions and trusted devices: `physical` (devices, blocks, links, datacenters and fleets, every
capacity an `Assumption`), `time` (one trusted clock, intervals, wipes) and `account` (credit, headroom, the covert
budget, and `feasible`, the joint rule for credit from several audits). `binding` places a circuit on a datacenter and
derives its crossings and public occupancy profile; `observe` runs it.

~~~python
from datacenter import *

c = trace(lambda x, w: (x * w) ^ 0x5A, x=Input(16), w=Input(16, source="registered"))
p = place(c, dc, {"*": "a.gpu", "x": "world"}, deliver="world")
claims = audit(run(p, {"x": 7, "w": 3}), occupancy(p, frame_bits=32, bucket=1), syncs=128)
print(account(claims, p, Clock(1e-3), Interval(0, 16)))
~~~

The objects mirror the Lean physical model: `Datacenter` is `Accounting.Hardware`, `Device` is `Accounting.Device`,
and `Placement` is `Accounting.Requirement.Placement` (#2399). Design: `note:network-accounting/20261018T0615Z-draft-datacenter-network-simulation`.
