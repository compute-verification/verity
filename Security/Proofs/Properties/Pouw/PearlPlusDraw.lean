import Properties.Pouw.PearlPlusDraw
import Mathlib.Data.Real.Basic
import Mathlib.Data.ENNReal.Real

/-!
# The witnesses of Pearl+'s run draw miss bound at δ (`Properties.Pouw.PearlPlus.RunDrawMissesAtDelta`)

In the exclusion form: the honest witness is the bound's premise at an explicit run, one draw over one tile that holds
every ticket (`Heavy`, by evaluation in the kernel); the nontrivial one is, at that tile with no draw, the premise
failing (ε at no draw exceeds one) and the bound failing with it (the draw names no tile, so it misses the tile with
probability one).
-/

namespace Properties.Pouw.PearlPlus.SecurityProofs

open Properties.Pouw.PearlPlus PouwCheck.PearlPlus
open scoped ENNReal

/-- One tile, `(0, 0)`, holding the run's one ticket. -/
def oneTile : List ((ℕ × ℕ) × ℕ) := [((0, 0), 1)]

/-- The one tile, as a set of tiles. -/
def oneTileSet : Finset (ℕ × ℕ) := {(0, 0)}

/-- **`RunDrawMissesAtDelta`'s honest witness**: at one draw, the tile that holds every ticket is heavy. -/
theorem runDrawMissesAtDelta_honest : Heavy oneTile 1 oneTileSet := by
  unfold Heavy weightOf ticketTotal oneTile oneTileSet
  decide +kernel

/-- **`RunDrawMissesAtDelta`'s nontrivial witness**: at no draw, the tile that holds every ticket is not heavy, and the
draw misses it with probability one, above δ. -/
theorem runDrawMissesAtDelta_nontrivial :
    ¬ Heavy oneTile 0 oneTileSet ∧
      ¬ Verity.Game.prCoin (MissesOn oneTile 0 oneTileSet) ≤ ENNReal.ofReal (DELTA : ℝ) := by
  refine ⟨?_, ?_⟩
  · unfold Heavy weightOf ticketTotal oneTile oneTileSet
    decide +kernel
  · have hall : ∀ g, MissesOn oneTile 0 oneTileSet g := fun _ => ⟨[], rfl, fun _ _ h => by simp at h⟩
    have h1 : Verity.Game.prCoin (MissesOn oneTile 0 oneTileSet) = 1 := by
      unfold Verity.Game.prCoin Verity.Game.avg
      have h0 : (Fintype.card (Fin 0 → Fin (drawBytes oneTile) → Fin 256) : ℝ≥0∞) ≠ 0 := by
        exact_mod_cast Fintype.card_ne_zero
      simp only [hall, if_true, mul_one, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      exact ENNReal.mul_inv_cancel h0 (ENNReal.natCast_ne_top _)
    have hq : DELTA < 1 := by decide +kernel
    have hd : (DELTA : ℝ) < 1 := by exact_mod_cast hq
    rw [h1]
    exact not_le.2 (ENNReal.ofReal_lt_one.2 hd)

end Properties.Pouw.PearlPlus.SecurityProofs
