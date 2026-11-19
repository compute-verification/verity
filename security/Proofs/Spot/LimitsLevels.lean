module

public import Definitions.Spot.LimitsLevels
public import Proofs.Core.LevelsCount

/-!
# The limits at a levels design, proved

`Spot.Limits.levelsCount`: `Levels.isCountBound_countAt` read on the sets of proof blocks `blk` places among the last
level's blocks. `Spot.Limits.takes_countAt`: at a levels design, `Takes` at any count gives `Takes` at
`Levels.countAt`, its count premise `levelsCount`'s.
-/

@[expose] public section

namespace Spot.Limits

open Verity Verity.Game
open scoped ENNReal

theorem levelsCount : LevelsCount := by
  intro G _ _ L n blk hblk δ B hδ
  have hsub : B.map blk ⊆ L.blocks := fun X hX => by
    obtain ⟨u, -, rfl⟩ := Finset.mem_map.1 hX
    exact hblk u
  simpa using L.isCountBound_countAt δ _ hsub hδ

/-- **`Takes` at a levels design, at the count over the last level's blocks.** -/
theorem takes_countAt {G : Type} [Fintype G] [DecidableEq G] {C : Circuit} {n : ℕ} (P : Partition C n)
    (L : Levels G) (blk : Fin n ↪ Finset G) (hblk : ∀ u, blk u ∈ L.blocks) {Reg : Type}
    {session : Finset (Finset G) → Reg → Game Bool} (A : Spot.Sampling.ExtractionAnalysis L.design Reg session)
    (tr : A.Tr → C.Trace) (rd : Strategy (Spot.Sampling.audit L.design Reg session) → (h w : Fin C.N) → BitVec (C.w w))
    (D : Description) (a₀ : C.Trace) (Dl : Finset (Fin C.N)) (δ εks εbind εw δin : ℝ≥0∞) (H k : ℕ)
    (h : Takes P L.design blk A tr rd D a₀ Dl δ εks εbind εw δin H k) :
    Takes P L.design blk A tr rd D a₀ Dl δ εks εbind εw δin H (L.countAt δ) := by
  obtain ⟨h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, -⟩ := h
  exact ⟨h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, levelsCount L blk hblk δ⟩

end Spot.Limits
