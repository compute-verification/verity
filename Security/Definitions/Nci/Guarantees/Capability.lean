import Definitions.Nci.Guarantees.Capability.Budgets
import Definitions.Nci.Guarantees.Capability.Incidence
import Definitions.Nci.Guarantees.Capability.Dependency

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

A new premise, structure or kind of consumer is a new module or a new value of an existing type, consumed by the same
steps. Improving the structure never strengthens a premise; a stronger premise is a new hypothesis, named as one. None
is proved yet: `LocalBoundsStated` holds their text in the lock. The survey of all of it is
`Security/Definitions/Nci/Capability/README.md`.
-/

namespace Nci.Guarantees

/-- **The local bounds are stated**: each of them, equal to itself, so the lock records their text and every definition
they read until they are proved. -/
def LocalBoundsStated : Prop :=
  let s := HeldBudget ∧ StatHeldBudget ∧ RecoveredBudget ∧ IncidenceCap ∧ OperandCapSeparate ∧ OperandCapJoint ∧
    ImportOrRecompute ∧ RecomputeByGates ∧ RecomputeByState ∧ WeightedProductCap
  s = s

end Nci.Guarantees
