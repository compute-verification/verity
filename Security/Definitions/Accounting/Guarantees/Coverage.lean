import Definitions.Accounting.Assumptions.Coverage

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
  ∀ {In L : Type} {m : ℕ} (G : Accounting.Slowdown.Task In (Fin m → List Bool)) (𝒜 : Accounting.Slowdown.Class In L)
    (Core : (In → Accounting.Actual.Run L) → In → Prop) (cost : Accounting.Actual.Run L → ℕ) (Tmax Lc : ℕ)
    (η : ℝ≥0∞),
    Assumptions.CoversEfficientMethods G 𝒜 Core cost Tmax η →
    (∀ C x, (C, x) ∈ 𝒜 → Core C x → Lc ≤ cost (C x)) →
    ∀ (C : In → Accounting.Actual.Run L) (pos : Fin m → L × ℕ × ℕ),
      G.prob (fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Accounting.Slowdown.Succeeds (C x) pos 𝒴 ∧ cost (C x) < min Lc Tmax) ≤ η

/-- **The floor under coverage is stated**: `CapabilityFloor`, equal to itself, so the lock records its text and every
definition it reads until it is proved. -/
def CapabilityFloorStated : Prop :=
  let s := CapabilityFloor
  s = s

end Nci.Guarantees
