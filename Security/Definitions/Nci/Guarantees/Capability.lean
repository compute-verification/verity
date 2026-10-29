import Definitions.Nci.Guarantees.Capability.Budgets
import Definitions.Nci.Guarantees.Capability.Incidence
import Definitions.Nci.Guarantees.Capability.Dependency
import Definitions.Nci.Guarantees.Capability.Witnesses
import Definitions.Nci.Guarantees.Capability.Pointwise
import Definitions.Nci.Guarantees.Capability.PointwiseWitnesses

/-!
# Local bounds: the repertoire

The statements a local bound is assembled from, one module per step of the argument:

1. `Budgets`: information premise + a short view → what every block holds (`HeldBudget`, `StatHeldBudget`,
   `RecoveredBudget`).
2. `Incidence` and `Dependency`: a budget + a structure → a per-block cap on counted products (`IncidenceCap`,
   `OperandCapSeparate`, `OperandCapJoint`; `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`,
   `WeightedProductCap`).
3. Counting, in core: a per-block cap → a circuit floor, `|C| ≥ c Ω / q` (`Verity.Guarantees.Counting`).
4. Coverage, over tasks and runs (`Definitions.Accounting.Guarantees.Coverage`): a circuit floor → success below it is
   rare (`CapabilityFloor`).

`Pointwise` reads steps 3 and 4 one execution at a time, over a decomposition of its actual circuit into qualifying
regions and a remainder: the per-region cap, the remainder ceiling and utilization bound the weighted count a gate
carries (`PointwiseCount`, `GammaBelow`), so carrying `Ω_*` takes `L = Ω_*/Γ` gates (`PointwiseThreshold`), and success
below `min(L, N_max)` has probability at most `δ + η` (`PointwiseEvent`); `WholeCapLocal` gives the per-region cap from
step 2's bounds.

`Witnesses` states each bound's witnesses at the toy circuit `Definitions.Nci.Capability.Toy`: an honest one (its
premises hold together there) and a nontrivial one (its conclusion fails there when a premise is dropped);
`PointwiseWitnesses` states `Pointwise`'s, at `Definitions.Nci.Capability.PointwiseToy`.
`CapabilityFloor`'s are in `Coverage`, at `Definitions.Accounting.CoverageToy`.

A new premise, structure or kind of consumer is a new module or a new value of an existing type, consumed by the same
steps. Improving the structure never strengthens a premise; a stronger premise is a new hypothesis, named as one. Each
is proved (`Proofs.Nci.Capability`). The survey of all of it is `Security/Definitions/Nci/Capability/README.md`.
-/
