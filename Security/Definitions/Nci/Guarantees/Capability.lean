import Definitions.Nci.Guarantees.Capability.Budgets
import Definitions.Nci.Guarantees.Capability.Incidence
import Definitions.Nci.Guarantees.Capability.Dependency
import Definitions.Nci.Guarantees.Capability.Fresh
import Definitions.Nci.Guarantees.Capability.Witnesses

/-!
# Local bounds: the repertoire

The statements a local bound is assembled from, one module per step of the argument:

1. `Budgets`: information premise + a short view → what every block holds (`HeldBudget`, `StatHeldBudget`,
   `RecoveredBudget`). `Fresh` puts an inference block's other fresh input into a budget's parameters: the advice into
   `aBlk` (`BlockDescriptionAdvice`), what faulty proof blocks inject into `k` (`FaultyInjection`).
2. `Incidence` and `Dependency`: a budget + a structure → a per-block cap on counted products (`IncidenceCap`,
   `OperandCapSeparate`, `OperandCapJoint`; `ImportOrRecompute`, `RecomputeByGates`, `RecomputeByState`,
   `WeightedProductCap`).
3. Counting, in core: a per-block cap → a circuit floor, `|C| ≥ c Ω / q` (`Verity.Guarantees.Counting`).
4. Coverage, over tasks and runs (`Definitions.Accounting.Guarantees.Coverage`): a circuit floor → success below it is
   rare (`CapabilityFloor`).

`Witnesses` states each bound's witnesses at the toy circuit `Definitions.Nci.Capability.Toy`: an honest one (its
premises hold together there) and a nontrivial one (its conclusion fails there when a premise is dropped).
`CapabilityFloor`'s are in `Coverage`, at `Definitions.Accounting.CoverageToy`.

A new premise, structure or kind of consumer is a new module or a new value of an existing type, consumed by the same
steps. Improving the structure never strengthens a premise; a stronger premise is a new hypothesis, named as one. Each
is proved (`Proofs.Nci.Capability`). The survey of all of it is `Security/Definitions/Nci/Capability/README.md`.
-/
