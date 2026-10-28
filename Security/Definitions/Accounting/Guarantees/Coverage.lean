import Definitions.Accounting.Assumptions.Coverage
import Definitions.Accounting.CoverageToy

/-!
# The floor under coverage

Coverage of efficient methods turns a floor on covered computations (from a local bound on inference blocks and the
counting theorem) into a bound on success below the floor.
-/

namespace Nci.Guarantees

open scoped ENNReal

/-- **Covered success below the floor is rare.** Under coverage of efficient methods at cost range `Tmax` and error `η`,
if every covered computation in the class `𝒜` costs at least `Lc` (the smallest floor over the local arguments whose
premises hold, each from its cap and Theorem 3), then for every computation the probability of lying in `𝒜`, reaching
the target, and costing below `min Lc Tmax` is at most `η`. -/
def CapabilityFloor : Prop :=
  ∀ {In L : Type} {m : ℕ} (G : Accounting.Overhead.Task In (Fin m → List Bool)) (𝒜 : Accounting.Overhead.Class In L)
    (Core : (In → Accounting.Actual.Run L) → In → Prop) (cost : Accounting.Actual.Run L → ℕ) (Tmax Lc : ℕ)
    (η : ℝ≥0∞),
    Assumptions.CoversEfficientMethods G 𝒜 Core cost Tmax η →
    (∀ C x, (C, x) ∈ 𝒜 → Core C x → Lc ≤ cost (C x)) →
    ∀ (C : In → Accounting.Actual.Run L) (pos : Fin m → L × ℕ × ℕ),
      G.prob (fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Accounting.Overhead.Succeeds (C x) pos 𝒴 ∧ cost (C x) < min Lc Tmax) ≤ η

/-- **`CapabilityFloor`'s premises hold together** at the toy task (`Accounting.Overhead.Toy`), with the covered
structure at least two gates, the cost a run's gate count, `Tmax = 10`, `η = 1/2` and `Lc = 2`: coverage holds, and
every covered computation costs at least `2`. Coverage is not empty: the guess, one gate, lacks the structure and
reaches the target on half the coins, so coverage fails at `η = 0`. -/
def CapabilityFloorHonest : Prop :=
  Assumptions.CoversEfficientMethods Accounting.Overhead.Toy.task Set.univ Accounting.Overhead.Toy.covered
      Accounting.Overhead.Toy.cost 10 2⁻¹ ∧
    (∀ C x, (C, x) ∈ (Set.univ : Accounting.Overhead.Class Unit Unit) → Accounting.Overhead.Toy.covered C x →
      2 ≤ Accounting.Overhead.Toy.cost (C x)) ∧
    ¬ Assumptions.CoversEfficientMethods Accounting.Overhead.Toy.task Set.univ Accounting.Overhead.Toy.covered
      Accounting.Overhead.Toy.cost 10 0

/-- **`CapabilityFloor`'s conclusion can fail**: at `η = 0`, the guess, costing one gate, below the floor `Lc = 2`,
reaches the target below the floor on half the coins. -/
def CapabilityFloorNontrivial : Prop :=
  ¬ Accounting.Overhead.Toy.task.prob (fun x 𝒴 =>
      (Accounting.Overhead.Toy.comp, x) ∈ (Set.univ : Accounting.Overhead.Class Unit Unit) ∧
        Accounting.Overhead.Succeeds (Accounting.Overhead.Toy.comp x) Accounting.Overhead.Toy.pos 𝒴 ∧
        Accounting.Overhead.Toy.cost (Accounting.Overhead.Toy.comp x) < min 2 10) ≤ 0

end Nci.Guarantees
