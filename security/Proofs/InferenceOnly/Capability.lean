module

public import Proofs.InferenceOnly.Capability.Budgets
public import Proofs.InferenceOnly.Capability.Dependency
public import Proofs.InferenceOnly.Capability.Witnesses

/-!
# Local bounds: the proofs

Each local bound of `Definitions.InferenceOnly.Guarantees.Capability`, proved over the vocabulary of `Definitions.InferenceOnly.Capability`:

* `Budgets`: `HeldBudget`, `StatHeldBudget`, `RecoveredBudget`;
* `Dependency`: `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`;
* `Items`: what they share about what a block holds;
* `Witnesses`: each bound's honest and nontrivial witnesses at the toy circuit.

The floor under coverage, `CapabilityFloor`, and its witnesses are `Proofs.Accounting.Coverage`.
-/

@[expose] public section
