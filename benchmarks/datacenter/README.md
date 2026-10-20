# datacenter

A datacenter as Verity accounts for it, in a small Python API: a Program, the hardware it runs on, the binding between
them, the verifier's plan, a run with what the trusted devices record, the audit's claims, and the account.

The accounting layer has two layers. The **logical layer** is where the cryptography happens: `logical` (a
`verity.ml.circuits` Program, the one circuit format the sampled proofs, PoUW's compiler and the census all read, at the
census's grain: its root nodes, the census values they pass each other and each Call's work by kind; its evaluation, a
Call at a time through `verity.ml.circuits.evaluation`'s kernels; and the hm96-sha512 commitments to its registered
inputs, with the occupancy profiles the developer may choose from, and to a run's census values, a row per element, so
that inputs and rows are words in the word view whether the Program is Boolean or not) and `claims` (`audit`: what the
trusted devices' records and the commitments establish about the prover system, with no device premise). The **physical
layer** interprets those claims with declared hardware, named assumptions and trusted devices: `physical` (devices,
blocks, links, certifier ports, the certifiers' cadence and trusted clock, datacenters and fleets, every number an
`Assumption`), `memory` (registered stores, challenges, read audits, wipes and the times to serve them) and `account`
(credit, attribution, where each store was, what could be carried in, the covert budget, the export, the `Ledger` across
epochs with its custody check, and `double_credit`, the joint rule for credit from several audits). `binding` places a
Program's root nodes on a datacenter (where an input enters says where its bits come from), derives its crossings and
publishes its `Profile`, the only part of the placement the physical layer reads. `plan` is the verifier's public plan
for an epoch, and `observe.run` is the world: it executes a placement and records what the trusted devices see.

The covert budget is only what goes beyond the committed computation. Beside it, the account reports the export, λ_out
(`Account.export_bits`): the committed bits that crossed a certified boundary out of a block, by egress port, from the
declared crossings (`Profile.exported`). It counts as novel information read by its receivers once the declaration's
certifiers cover every link, so that `Datacenter.uncovered_links()`, the links no trusted device ends (a
switch-to-switch link, an outside host's link to a switch), is empty; until then it is reported as not applicable, with
those links named. The registration commits the occupancy profiles the developer may choose from
(`register(..., profiles=...)`), the plan reads their number from it (`Plan(..., profiles=registration)`), so the
choice is public advice of log2 of that number, and the audit refuses a declared occupancy that isn't one of them; with
none registered the one profile is the placement's own. `Ledger.custody(registered_bits)` is the share of the
registered weights that the ledger's epochs could have carried out: their covert budgets, their exports where charged,
and a side channel of σ a server over each epoch (`DEDICATED_RACKS`, 1 kbit/s, Daniel's premise that no untrusted
machine shares a block's rack). Weight custody over a horizon needs that share well below 1.

`examples` is the one place hardware is declared, every number with its source: `rtx_node` (one of our measured
nodes), `with_dpu` (a BlueField-3 between its NIC and its certifier, with its BMC behind it), `leaf_spine` (nodes behind
their certifiers on a leaf-spine fabric), `rack` (the demo's) and `two_sites` (a `Fleet` of two leaf-spines joined by
a WAN between their spines). A link between datacenters is fabric, as a switch is: coverage is per block, every block's
boundary is certified already, and the outside world is everything off every block. `benchmarks/network_sim/graph.py`
exports them as a network, a netns plan or a GML graph, and its tests pin their values against `verity` and the census.
`models` builds the demo's and the tests' small Programs from registered Boolean Definitions (`Gemm_v3`, the BF16
tensor-core step on bits, which `Gemm_v2`'s kernel evaluates through the word view).

~~~python
from datacenter import *
from datacenter.examples import rack
from datacenter.models import WEIGHTS, X, two_stage

program = two_stage()                                  # x -> h = x @ W1^T -> y = h @ W2^T, the weights registered
p = place(program, rack(nodes=2), "n0.gpu0", registration=register(program, WEIGHTS), deliver="world", x="world",
          h="n0.gpu0", y="n1.gpu0")
plan = Plan(Interval(0, 16))
profile = p.public(plan)
print(account(audit(run(p, {"x": X}, plan), profile), profile))
~~~

`python -m datacenter.demo` runs a night on a two-node rack, honestly and then with an adversary, and
`python tests/test_walkthrough.py` prints the walkthrough (the Notion page "Accounting for a datacenter, from a circuit
up") section by section. `benchmarks/network_sim/workload/accounting.py` accounts for the network simulator's MLP and
verity_vllm's serving Program the same way.

The objects mirror the Lean physical model: `Datacenter` is `Accounting.Hardware`, `Block` is `Hardware.SpatialUnit`
with `Covered`, `Placement` is `Accounting.Placed` (and `Accounting.Requirement.Placement`, in #2399) at the census's
grain, a `Store` is `Accounting.Requirement.Store`, and an `Interval` is `Accounting.Window`. Design:
`note:network-accounting/20261018T0615Z-draft-datacenter-network-simulation`.
