import Proofs.Nci.Cover.Area

/-!
# The cover theorem: the additive bound

Under `ConditionalUpdateHardness` a block's whole products pay for their uses inside the activation regions
(`whole_use_entry`, one gate of `A j` per use) and for their updated parameters outside them, so `#J + b #L ≤ k`, and
`4 b #J #L ≤ (#J + b #L)² ≤ k²`. The rest are at most `k`, as in the area bound.
-/

namespace Nci.Cover.SecurityProofs

/-- `ConditionalUpdateHardness` implies `UpdateHardness`. -/
theorem updateHardness_of_conditional {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s : ℕ} {b : ℝ} (h : Assumptions.ConditionalUpdateHardness M A s b) :
    Assumptions.UpdateHardness M s b := by
  sorry

theorem AdditiveIntensity : Guarantees.AdditiveIntensity := by
  sorry

theorem AdditiveCoverBound : Guarantees.AdditiveCoverBound := by
  sorry

end Nci.Cover.SecurityProofs
