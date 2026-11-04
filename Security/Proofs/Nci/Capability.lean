import Proofs.Nci.Capability.Budgets
import Proofs.Nci.Capability.Incidence
import Proofs.Nci.Capability.Dependency
import Proofs.Nci.Capability.Witnesses

/-!
# Local bounds: the proofs

Each local bound of `Definitions.Nci.Guarantees.Capability`, proved over the vocabulary of `Definitions.Nci.Capability`:

* `Budgets`: `HeldBudget`, `StatHeldBudget`, `RecoveredBudget`;
* `Incidence`: `IncidenceCap`, `OperandCapSeparate`, `OperandCapJoint`;
* `Dependency`: `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`, `WeightedProductCap`;
* `Items`: what they share about what a block holds;
* `Witnesses`: each bound's honest and nontrivial witnesses at the toy circuit.

The floor under coverage, `CapabilityFloor`, and its witnesses are `Proofs.Accounting.Coverage`.
-/
