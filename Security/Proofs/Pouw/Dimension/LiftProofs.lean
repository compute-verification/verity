import Definitions.Pouw.Dimension.Lift
import Models.Pouw.Assumptions.Dimension.Pipe
import Proofs.Pouw.Dimension.MinRankProofs

/-! # `Lift_MinRank` and MinRank ≥ ρ: the witnesses -/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.GameProofs Pouw.NCP.RelationWitness

/-- **`Lift_MinRank`'s witness.** -/
theorem liftMinRankWitness : LiftMinRankWitness := by
  intro ρ k R _ _ m n N A B U L P σ _ _
  rw [one_mul]
  exact_mod_cast Finite.card_subtype_le _

/-- **MinRank ≥ ρ's witness, jointly.** -/
theorem minRankGeWitness : MinRankGeWitness := fun k hk h16 =>
  ⟨shiftBy 24 k, shiftBy24_admissible hk h16, indep_of_fresh (fresh hk h16), minRank2_shift hk h16,
    fun _ a ha b hb hne => ⟨fun _ => a, fun _ => b, fun i j _ => Subsingleton.elim i j,
      fun i j _ => Subsingleton.elim i j, fun _ => ⟨ha, hb⟩, fun _ => hne, fun i l h => by
        have := i.2; have := l.2; have := Fin.lt_def.1 h; omega⟩⟩

private lemma epsL5e_base {ρ G Λ : ℕ} (h : Λ + G ≤ ρ + 1) :
    0 < (1 + (64 : ℝ) ^ ((Λ : ℤ) - ρ + G - 1)) / 64 ∧ (1 + (64 : ℝ) ^ ((Λ : ℤ) - ρ + G - 1)) / 64 ≤ 1 := by
  have he : (Λ : ℤ) - ρ + G - 1 ≤ 0 := by omega
  have h0 : 0 < (64 : ℝ) ^ ((Λ : ℤ) - ρ + G - 1) := zpow_pos (by norm_num) _
  have h1 : (64 : ℝ) ^ ((Λ : ℤ) - ρ + G - 1) ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) he
  constructor <;> [positivity; (rw [div_le_one (by norm_num)]; linarith)]

private lemma epsL5e_den_nonneg (G Λ k M n T : ℕ) :
    0 ≤ ((n * T + k : ℕ) : ℝ) / G + T + ((M * T : ℕ) : ℝ) / Λ := by positivity

/-- **`LiftMinRankL5e`'s witness.** -/
theorem liftMinRankL5eWitness : LiftMinRankL5eWitness := by
  refine ⟨fun k hk h16 => ?_, fun ρ G Λ k M n T d _ _ h hd => ?_, fun ρ G Λ k M n T d _ _ h hd => ?_⟩
  · obtain ⟨R, hA, hI, _, hM⟩ := minRankGeWitness k hk h16
    exact ⟨R, hA, hI, hM⟩
  · obtain ⟨hb0, hb1⟩ := epsL5e_base h
    refine one_le_zpow_of_nonpos₀ hb0 hb1 ?_
    exact Int.ceil_le.2 (by
      rw [Int.cast_zero]
      exact div_nonpos_of_nonpos_of_nonneg (by exact_mod_cast hd) (epsL5e_den_nonneg G Λ k M n T))
  · obtain ⟨hb0, hb1⟩ := epsL5e_base h
    refine zpow_le_one₀ hb0 hb1 ?_
    exact Int.ceil_nonneg (div_nonneg (by exact_mod_cast hd) (epsL5e_den_nonneg G Λ k M n T))

end Pouw.Dimension
