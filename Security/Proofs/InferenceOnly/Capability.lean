import Proofs.InferenceOnly.Capability.Budgets
import Proofs.InferenceOnly.Capability.Incidence
import Proofs.InferenceOnly.Capability.Dependency
import Proofs.InferenceOnly.Capability.Witnesses

/-!
# Local bounds: the proofs

Each local bound of `Definitions.InferenceOnly.Guarantees.Capability`, proved over the vocabulary of `Definitions.InferenceOnly.Capability`:

* `Budgets`: `HeldBudget`, `StatHeldBudget`, `RecoveredBudget`;
* `Incidence`: `IncidenceCap`, `OperandCapSeparate`, `OperandCapJoint`;
* `Dependency`: `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`, `WeightedProductCap`;
* `Items`: what they share about what a block holds;
* `Witnesses`: each bound's honest and nontrivial witnesses at the toy circuit.

The floor under coverage, `CapabilityFloor`, and its witnesses are `Proofs.Accounting.Coverage`.
-/
