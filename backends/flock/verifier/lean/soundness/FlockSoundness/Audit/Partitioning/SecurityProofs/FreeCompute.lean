import FlockSoundness.Audit.Partitioning.Guarantees

/-!
# Optimal partitioning against free compute: the proofs

The proofs of the free-compute statements of `Partitioning.Guarantees`.
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs

open Finset Game
open scoped ENNReal

theorem incl_le_one {n : ℕ} (L : Law n) (u : Fin n) : L.incl u ≤ 1 := by
  classical
  unfold Law.incl prCoin
  refine (avg_mono fun ω => ?_).trans (avg_const (C := L.Ω) 1).le
  split_ifs <;> simp

theorem ConverseExact : Guarantees.ConverseExact := by
  sorry

theorem ConverseAsymptotic : Guarantees.ConverseAsymptotic := by
  sorry

theorem Achievability : Guarantees.Achievability := by
  sorry

theorem GranularityOptimum : Guarantees.GranularityOptimum := by
  sorry

theorem GranularityChoice : Guarantees.GranularityChoice := by
  sorry

theorem HeavyUnit : Guarantees.HeavyUnit := by
  intro n w C δ L hL u hu
  have h := hL {u} (by simpa using hu)
  rw [Law.escape_singleton, ENNReal.toReal_sub_of_le (incl_le_one L u) ENNReal.one_ne_top,
    ENNReal.toReal_one] at h
  linarith

theorem UnequalDraws : Guarantees.UnequalDraws := by
  sorry

theorem UnequalCost : Guarantees.UnequalCost := by
  sorry

end FlockSoundness.Audit.Partitioning.SecurityProofs
