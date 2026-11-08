module

public import Proofs.Nci.Capability.Budgets
public import Proofs.Nci.Capability.Dependency
public import Proofs.Nci.Capability.Witnesses

/-!
# Local bounds: the proofs

Each local bound of `Definitions.Nci.Guarantees.Capability`, proved over the vocabulary of `Definitions.Nci.Capability`:

* `Budgets`: `HeldBudget`, `StatHeldBudget`, `RecoveredBudget`;
* `Dependency`: `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`;
* `Items`: what they share about what a block holds;
* `Witnesses`: each bound's honest and nontrivial witnesses at the toy circuit.

The floor under coverage, `CapabilityFloor`, and its witnesses are `Proofs.Accounting.Coverage`.
-/

@[expose] public section
