module

public import Proofs.Slowdown.Capability.Budgets
public import Proofs.Slowdown.Capability.Dependency
public import Proofs.Slowdown.Capability.Witnesses

/-!
# Local bounds: the proofs

Each local bound of `Definitions.Slowdown.Guarantees.Capability`, proved over the vocabulary of `Definitions.Slowdown.Capability`:

* `Budgets`: `HeldBudget`, `StatHeldBudget`, `RecoveredBudget`;
* `Dependency`: `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`;
* `Items`: what they share about what a block holds;
* `Witnesses`: each bound's honest and nontrivial witnesses at the toy circuit.

The floor under coverage, `CapabilityFloor`, and its witnesses are `Proofs.Slowdown.Coverage`.
-/

@[expose] public section
