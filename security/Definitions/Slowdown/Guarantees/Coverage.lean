module

public import Definitions.Slowdown.Assumptions.Coverage
public import Definitions.Slowdown.CoverageToy
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# The floor under coverage

Coverage of efficient methods turns a floor on covered computations (from a local bound on inference blocks and the
counting theorem) into a bound on success below the floor.
-/

@[expose] public section

namespace Slowdown.Guarantees

open scoped ENNReal

/-- **Covered success below the floor is rare.** Under coverage of efficient methods at cost range `Tmax` and error `η`,
if every covered computation in the class `𝒜` costs at least `Lc` (the smallest floor over the local arguments whose
premises hold, each from its cap and Theorem 3), then for every computation the probability of lying in `𝒜`, reaching
the target, and costing below `min Lc Tmax` is at most `η`. -/
def CapabilityFloor : Prop :=
  ∀ {In L : Type} {m : ℕ} (G : ResourceAccounting.Slowdown.Task In (Fin m → List Bool)) (𝒜 : ResourceAccounting.Slowdown.Class In L)
    (Core : (In → ResourceAccounting.Actual.Run L) → In → Prop) (cost : ResourceAccounting.Actual.Run L → ℕ) (Tmax Lc : ℕ)
    (η : ℝ≥0∞),
    Assumptions.CoversEfficientMethods G 𝒜 Core cost Tmax η →
    (∀ C x, (C, x) ∈ 𝒜 → Core C x → Lc ≤ cost (C x)) →
    ∀ (C : In → ResourceAccounting.Actual.Run L) (pos : Fin m → L × ℕ × ℕ),
      G.prob (fun x 𝒴 => (C, x) ∈ 𝒜 ∧ ResourceAccounting.Slowdown.Succeeds (C x) pos 𝒴 ∧ cost (C x) < min Lc Tmax) ≤ η

/-- **`CapabilityFloor`'s premises hold together** at the toy task (`ResourceAccounting.Slowdown.Toy`), with the covered
structure at least two gates, the cost a run's gate count, `Tmax = 10`, `η = 1/2` and `Lc = 2`: coverage holds, and
every covered computation costs at least `2`. Coverage is not empty: the guess, one gate, lacks the structure and
reaches the target on half the coins, so coverage fails at `η = 0`. The floor's hypothesis is the covered structure
written out, not `Toy.covered`: the lock's witnesses rule refuses an honest witness's hypothesis headed by a definition
outside an assumptions module. -/
def CapabilityFloorHonest : Prop :=
  Assumptions.CoversEfficientMethods ResourceAccounting.Slowdown.Toy.task Set.univ ResourceAccounting.Slowdown.Toy.covered
      ResourceAccounting.Slowdown.Toy.cost 10 2⁻¹ ∧
    (∀ (C : Unit → ResourceAccounting.Actual.Run Unit) x, (C, x) ∈ (Set.univ : ResourceAccounting.Slowdown.Class Unit Unit) →
      2 ≤ (C x).A.N → 2 ≤ ResourceAccounting.Slowdown.Toy.cost (C x)) ∧
    ¬ Assumptions.CoversEfficientMethods ResourceAccounting.Slowdown.Toy.task Set.univ ResourceAccounting.Slowdown.Toy.covered
      ResourceAccounting.Slowdown.Toy.cost 10 0

/-- **`CapabilityFloor`'s conclusion can fail**: at `η = 0`, the guess, costing one gate, below the floor `Lc = 2`,
reaches the target below the floor on half the coins. -/
def CapabilityFloorNontrivial : Prop :=
  ¬ ResourceAccounting.Slowdown.Toy.task.prob (fun x 𝒴 =>
      (ResourceAccounting.Slowdown.Toy.comp, x) ∈ (Set.univ : ResourceAccounting.Slowdown.Class Unit Unit) ∧
        ResourceAccounting.Slowdown.Succeeds (ResourceAccounting.Slowdown.Toy.comp x) ResourceAccounting.Slowdown.Toy.pos 𝒴 ∧
        ResourceAccounting.Slowdown.Toy.cost (ResourceAccounting.Slowdown.Toy.comp x) < min 2 10) ≤ 0

end Slowdown.Guarantees
