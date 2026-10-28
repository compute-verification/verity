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

/-- **`CapabilityFloor`'s premises hold together** at the toy task (`Accounting.Overhead.Toy`): every computation has
the covered structure, so coverage holds at `η = 0`, and every one costs `5 = Lc`. -/
def CapabilityFloorHonest : Prop :=
  Assumptions.CoversEfficientMethods Accounting.Overhead.Toy.task Set.univ Accounting.Overhead.Toy.covered
      Accounting.Overhead.Toy.cost 5 0 ∧
    ∀ C x, (C, x) ∈ (Set.univ : Accounting.Overhead.Class Unit Unit) → Accounting.Overhead.Toy.covered C x →
      5 ≤ Accounting.Overhead.Toy.cost (C x)

/-- **`CapabilityFloor`'s conclusion can fail**: when a computation costs `0`, below the floor `Lc = 5`, and reaches
the target on every input, success below the floor has probability `1`, above `η = 0`. -/
def CapabilityFloorNontrivial : Prop :=
  ¬ Accounting.Overhead.Toy.task.prob (fun x 𝒴 =>
      (Accounting.Overhead.Toy.comp, x) ∈ (Set.univ : Accounting.Overhead.Class Unit Unit) ∧
        Accounting.Overhead.Succeeds (Accounting.Overhead.Toy.comp x) Accounting.Overhead.Toy.pos 𝒴 ∧
        Accounting.Overhead.Toy.free (Accounting.Overhead.Toy.comp x) < min 5 5) ≤ 0

end Nci.Guarantees
