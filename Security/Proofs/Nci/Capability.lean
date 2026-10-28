import Proofs.Nci.Capability.Budgets
import Proofs.Nci.Capability.Incidence
import Proofs.Nci.Capability.Dependency

/-!
# Local bounds: the proofs

Each local bound of `Definitions.Nci.Guarantees.Capability`, proved over the vocabulary of `Definitions.Nci.Capability`:

* `Budgets`: `HeldBudget`, `StatHeldBudget`, `RecoveredBudget`;
* `Incidence`: `IncidenceCap`, `OperandCapSeparate`, `OperandCapJoint`;
* `Dependency`: `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`, `WeightedProductCap`;
* `Items`: what they share about what a block holds.

The floor under coverage, `CapabilityFloor`, is `Proofs.Accounting.Coverage`.
-/
