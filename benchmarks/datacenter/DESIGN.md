# datacenter: design

The package's interface, its two layers in detail, how a result is presented, the hardware examples, and the Lean
model its objects mirror. [`README.md`](README.md) says what it is and gives one thing to try.

## Interface

- **Python**: the package `datacenter`, the distribution `verity-datacenter` (this directory's `pyproject.toml`; it
  depends on `verity` alone). It is not a member of the uv workspace, so nothing installs it: an importer puts this
  directory on `sys.path`.
- **Importers**: `tests/benchmarks/pouw/test_proved_ticks_placed.py` alone. No code under `verity/` imports it. What
  it imports:
  - from `datacenter`: `FLEET_CLOSED`, `Assumption`, `Cadence`, `Computed`, `Credit`, `Datacenter`, `DeviceEntry`,
    `Interval`, `LinkEntry`, `Plan`, `Zone`, `account`, `ands`, `crossings`, `names`, `place` and `register`;
  - from `datacenter.logical`: `census_of`;
  - from `datacenter.models`: `s8_grid`.
- **Command**: `python -m datacenter.demo`, run from this directory with the workspace's Python. There is no console
  script.
- **Lean**: none. `physical` mirrors the Lean physical model, and no Lake package is here.
- **Documents**: none. The package writes no file and defines no schema.
- **Loads from**: `verity.circuits`, `verity.commitments`, `verity_vllm.definitions`, the registration log
  (`verity.protocols.registration.verifier`), `verity.protocols.proofs.spot.verifier` and PoUB's `timing`
  and `codec` (`verity.protocols.resource_accounting.memory.poub.verifier`).

## Overview

A datacenter as Verity accounts for it, in a small Python API: a Program, the hardware it runs on, the binding between
them, the verifier's plan, a run with what the trusted devices record, the audit's claims, and the account.

The accounting layer has two layers. The **logical layer** is where the cryptography happens: `logical` (a
`verity.circuits` Program, the one circuit format SPOT, PoUW's compiler and the census all read, at the
census's grain: its root nodes, the census values they pass each other, each Call's work by kind and its ANDs, the
credit; its evaluation, a Call at a time through `verity.circuits.evaluation`'s kernels; and the hm96-sha512
commitments to its registered inputs, with the occupancy profiles the developer may choose from, and to a run's census
values, a row per element, so that inputs and rows are words in the word view whether the Program is Boolean or not) and
`claims` (`audit`: what the trusted devices' records and the commitments establish about the developer's hardware, with no
device premise). The **physical layer** interprets those claims with registered hardware, named assumptions and trusted
devices: `physical` (devices, zones, links, network attester ports, the network attesters' cadence and trusted clock, datacenters and
fleets, every number an `Assumption`), `memory` (registered stores, challenges, read audits, memory wipes and the times
to serve them) and `account` (credit in ANDs, with the MACs beside it, attribution, where each store was, what could be
carried in, the covert budget, the export, `Accounts` across epochs with its custody check, and `double_credit`, the
joint rule for credit from several audits). `binding` places a Program's root nodes on a datacenter (where an input
enters says where its bits come from), derives its crossings and publishes its `Profile`, the only part of the placement
the physical layer reads. `plan` is the verifier's public plan for an epoch, and `observe.run` is the world: it executes
a placement and records what the trusted devices see.

The covert budget is only what goes beyond the committed computation. Beside it, the account reports the export, λ_out
(`Account.export_bits`): the committed bits that crossed a certified boundary out of a zone, by egress port, from the
declared crossings (`Profile.exported`). It counts as fresh input read by its receivers once the declaration's
network attesters cover every link, so that `Datacenter.uncovered_links()`, the links no trusted device ends (a
switch-to-switch link, an outside host's link to a switch), is empty; until then it is reported as not applicable, with
those links named. With the network attesters' pair links sealed (`Datacenter(sealed=True)`, off by default, and modeled only: the network
attester has no sealed pair link), a crossing from one zone to another is ciphertext between the two zones'
network attesters, so the export counts only the crossings to a reader off every zone, the facility boundary
(`Placed._exported`). The registration commits the occupancy profiles the developer may choose from
(`register(..., profiles=...)`), the plan reads their number from it (`Plan(..., profiles=registration)`), so the
choice is public advice of log2 of that number, and the audit refuses a declared occupancy that isn't one of them; with
none registered the one profile is the placement's own. `Accounts.custody(registered_bits)` is the share of the
registered weights that its epochs could have carried out: their covert budgets, their exports where charged,
and a side channel of σ a server over each epoch (`DEDICATED_RACKS`, 1 kbit/s, on the premise that no untrusted
machine shares a zone's rack). Weight custody over a monitoring period needs that share well below 1.

A result here is about a monitoring period: the stretch of time it covers, not one epoch, execution phase or run. Present it
for a week, a month and a year, over all compute, the adversary resource model (`world-inference-2027` in
`benchmarks/census/adversary.json`), with the datacenter's figure beside it, even though the demo checks two nodes. Lead with
the pooled total over the period and its share of the compute it covers, at δ = 1% and 10⁻⁶, with the per-epoch figure
beside it, never alone. Each epoch flips its own live coins, so the pooled total stays about the same as the period
grows. The datacenter's size and the epoch's length are stated parameters (from `benchmarks/census`), and a figure
scaled up from the demo's machines says which parts are proved and which are scaled.

`examples` is the one place hardware is declared, every number with its source: `rtx_node` (one of our measured
nodes), `with_dpu` (a BlueField-3 between its NIC and its network attester, with its BMC behind it), `leaf_spine` (nodes behind
their network attesters on a leaf-spine fabric), `rack` (the demo's) and `two_sites` (a `Fleet` of two leaf-spines joined by
a WAN between their spines). A link between datacenters is fabric, as a switch is: coverage is per zone, every zone's
boundary is certified already, and the outside world is everything off every zone.
`models` builds the demo's and the tests' small Programs from registered Boolean Definitions (`Gemm_v3`, the BF16
tensor-core step on bits, which `Gemm_v2`'s kernel evaluates through the word view).

The objects mirror the Lean physical model: `Datacenter` is `ResourceAccounting.Hardware`, a `DeviceEntry` is
`ResourceAccounting.Device` (the node's constants are its one Python mirror's, `verity.protocols.resource_accounting.memory.poub.verifier.timing`),
`Zone` is `Hardware.SpatialUnit` with `Covered`, `Placed` is `ResourceAccounting.Placed` (and
`ResourceAccounting.Requirement.Placement`) at the census's grain, a `StoreEntry` is `ResourceAccounting.Requirement.Store`, and an
`Interval` is `ResourceAccounting.Window`.
