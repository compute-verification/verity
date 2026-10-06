import Proofs.Flock.Soundness.Audit.Partitioning.Model

/-!
# Proofs of three of `Partitioning.Witness`'s instances

`achievability_witness`, `granularity_witness` and `unequal_witness` have the statements of `Witness.achievability`,
`Witness.granularity` and `Witness.unequal`, word for word. This file imports only `Partitioning.Model`, so `Witness` can
import it. `Law.subset n n` draws every unit, so every set doing positive work escapes it with probability 0
(`meetsTarget_subset_self`).
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal

open Finset

theorem log_inv_half : Real.log (1 / 2 : ℝ)⁻¹ = Real.log 2 := by norm_num

theorem log_two_le : Real.log 2 ≤ 1 := by
  have := Real.log_le_sub_one_of_pos (two_pos : (0 : ℝ) < 2)
  linarith

/-- The law drawing every unit meets every positive target. -/
theorem meetsTarget_subset_self {n : ℕ} (w : Fin n → ℝ) {C δ : ℝ} (hC : 0 < C) (hδ : 0 ≤ δ) :
    MeetsTarget (Law.subset n n le_rfl) w C δ := fun B hB => by
  have hne : B.Nonempty := by
    rcases B.eq_empty_or_nonempty with h | h
    · rw [h, Finset.sum_empty] at hB
      linarith
    · exact h
  have hc := hne.card_pos
  have hcn : B.card ≤ n := by simpa using Finset.card_le_univ B
  rw [Law.subset_escape, Nat.choose_eq_zero_of_lt (by omega)]
  simpa using hδ

theorem sum_fin100 (a b : ℝ) : ∑ u : Fin 100, (if u.val < 50 then a else b) = 50 * a + 50 * b := by
  rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul]
  have h1 : (univ.filter fun u : Fin 100 => u.val < 50).card = 50 := by decide
  have h2 : (univ.filter fun u : Fin 100 => ¬ u.val < 50).card = 50 := by decide
  rw [h1, h2]
  norm_num

theorem achievability_witness : min 1 (Real.log (1 / 2 : ℝ)⁻¹ / 2 * 1) + 1 / 4 < 1 := by
  rw [log_inv_half]
  have := log_two_le
  have := min_le_right (1 : ℝ) (Real.log 2 / 2 * 1)
  linarith

theorem granularity_witness :
    MeetsTarget (Law.subset 1000 1000 le_rfl) (fun _ => 1) 10 (1 / 2) ∧
    MeetsTarget (Law.subset 500 500 le_rfl) (fun _ => 2) 10 (1 / 2) ∧
    0 < 1 - 1 / 10 - 10 / (1000 * 1) - 1 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) ∧
    0 < 1 - 2 / 10 - 10 / (500 * 2) - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10) := by
  rw [log_inv_half]
  have := log_two_le
  exact ⟨meetsTarget_subset_self _ (by norm_num) (by norm_num), meetsTarget_subset_self _ (by norm_num) (by norm_num),
    by linarith, by linarith⟩

theorem unequal_witness :
    (∀ u : Fin 100, 1 ≤ (if u.val < 50 then (1 : ℝ) else 2) ∧ (if u.val < 50 then (1 : ℝ) else 2) ≤ 2) ∧
    2 * Real.log (1 / 2 : ℝ)⁻¹ ≤ 10 ∧
    MeetsTarget (Law.subset 100 100 le_rfl) (fun u => if u.val < 50 then 1 else 2) 10 (1 / 2) ∧
    0 < Real.log (1 / 2 : ℝ)⁻¹ * (1 - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10)) *
      ((∑ u : Fin 100, if u.val < 50 then (1 : ℝ) else 2) / (2 * 10) * (1 - 2 / 10) -
        (⌊Real.logb 2 (2 / 1)⌋₊ + 1)) ∧
    0 < Real.log (1 / 2 : ℝ)⁻¹ * (1 - 2 * Real.log (1 / 2 : ℝ)⁻¹ / (2 * 10)) *
      ((∑ u : Fin 100, (if u.val < 50 then (1 : ℝ) else 2) * unitCost 0 1 (if u.val < 50 then 1 else 2)) /
          (2 ^ 2 * 10) * (1 - 2 / 10) - (⌊Real.logb 2 (2 / 1)⌋₊ + 1) * unitCost 0 1 2) := by
  have hl := log_two_le
  have hl0 : 0 < Real.log 2 := Real.log_pos one_lt_two
  have hJ : ⌊Real.logb 2 (2 / 1)⌋₊ = 1 := by
    rw [div_one, Real.logb_self_eq_one one_lt_two, Nat.floor_one]
  have hcost : ∑ u : Fin 100, (if u.val < 50 then (1 : ℝ) else 2) * unitCost 0 1 (if u.val < 50 then 1 else 2) =
      50 * 1 + 50 * 4 := by
    rw [← sum_fin100]
    exact Finset.sum_congr rfl fun u _ => by split_ifs <;> norm_num [unitCost]
  rw [log_inv_half, hJ, sum_fin100, hcost]
  have hβ : 0 < Real.log 2 * (1 - 2 * Real.log 2 / (2 * 10)) := mul_pos hl0 (by linarith)
  refine ⟨fun u => by split_ifs <;> norm_num, by linarith, meetsTarget_subset_self _ (by norm_num) (by norm_num),
    mul_pos hβ (by norm_num), mul_pos hβ (by norm_num [unitCost])⟩

end FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal
