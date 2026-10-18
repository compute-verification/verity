# datacenter

A datacenter as Verity accounts for it, in a small Python API: a circuit, the hardware it runs on, the binding between
them, the verifier's plan, a run with what the trusted devices record, the audit's claims, and the account.

The accounting layer has two layers. The **logical layer** is where the cryptography happens: `logical` (a circuit
traced from Python, its inputs and where their bits come from, the commitment to its values) and `claims` (`audit`: what
the trusted devices' records and the commitment establish about the prover system, with no device premise). The
**physical layer** interprets those claims with declared hardware, named assumptions and trusted devices: `physical`
(devices, blocks, links, certifier ports, the certifiers' grid and trusted clock, datacenters and fleets, every number
an `Assumption`), `memory` (registered stores, challenges, read audits, wipes and the times to serve them) and `account`
(credit, attribution, where each store was, what could be carried in, the covert budget, the `Ledger` across epochs,
and `double_credit`, the joint rule for credit from several audits). `binding` places a circuit on a datacenter,
derives its crossings and publishes its `Profile`, the only part of the placement the physical layer reads. `plan` is
the verifier's public plan for an epoch, and `observe.run` is the world: it executes a placement and records what the
trusted devices see.

~~~python
from datacenter import *
from datacenter.examples import rack

dc = rack(nodes=2)
c = trace(lambda x, w: (x * w) ^ 0x5A, x=Input(16), w=Input(16, "registered", digest(3, 16)))
p = place(c, dc, "n0.gpu0", deliver="world", x="world")
plan = Plan(Interval(0, 16))
profile = p.public(plan)
print(account(audit(run(p, {"x": 7, "w": 3}, plan), profile), profile))
~~~

`python -m datacenter.demo` runs a night on a two-node rack, honestly and then with an adversary, and
`python tests/test_walkthrough.py` prints the walkthrough (the Notion page "Accounting for a datacenter, from a circuit
up") section by section.

The objects mirror the Lean physical model: `Datacenter` is `Accounting.Hardware`, `Block` is `Hardware.SpatialUnit`
with `Covered`, and `Placement` is `Accounting.Placed` and `Accounting.Requirement.Placement` (#2386). Design:
`note:network-accounting/20261018T0615Z-draft-datacenter-network-simulation`.
