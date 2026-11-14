module

public import Definitions.Slowdown.Guarantees.Capability.Budgets
public import Definitions.Slowdown.Guarantees.Capability.Dependency
public import Definitions.Slowdown.Guarantees.Capability.Witnesses

/-!
# Local bounds: the repertoire

The statements a local bound is assembled from, one module per step of the argument:

1. `Budgets`: information premise + a short view → what every block holds (`HeldBudget`, `StatHeldBudget`,
   `RecoveredBudget`).
2. `Dependency`: a budget + a structure → a per-block cap on what a block finishes (`ImportOrRecompute`,
   `RecomputeByGates`, `RecomputeByState`).
3. Counting, in core: a per-block cap → a circuit floor, `|C| ≥ c Ω / q` (`Verity.Guarantees.Counting`).
4. Coverage, over tasks and runs (`Definitions.Slowdown.Guarantees.Coverage`): a circuit floor → success below it is
   rare (`CapabilityFloor`).

`Witnesses` states each bound's witnesses at the toy circuit `Definitions.Slowdown.Capability.Toy`: an honest one (its
premises hold together there) and a nontrivial one (its conclusion fails there when a premise is dropped).
`CapabilityFloor`'s are in `Coverage`, at `Definitions.Slowdown.CoverageToy`.

A new premise, structure or kind of consumer is a new module or a new value of an existing type, consumed by the same
steps. Improving the structure never strengthens a premise; a stronger premise is a new hypothesis, named as one. Each
is proved (`Proofs.Slowdown.Capability`). The survey of all of it is `security/Definitions/Slowdown/Capability/README.md`.
-/

@[expose] public section
