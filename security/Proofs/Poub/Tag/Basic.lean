module

public import Mathlib.Probability.Distributions.Uniform
public import Definitions.Core.Prob

/-! Lemmas the tag proofs share: under a uniform sample, an event's probability is its count over the type's. -/

@[expose] public section

open Verity.Game renaming prPMF → prob

namespace ResourceAccounting.TagProofs

open scoped ENNReal

theorem prob_eq_sum {α : Type} [Fintype α] (μ : PMF α) (E : α → Prop) [DecidablePred E] :
    prob μ E = ∑ x, if E x then μ x else 0 := by
  unfold prob
  rw [PMF.toOuterMeasure_apply_fintype]
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases h : E x <;> simp [Set.indicator, h]

theorem prob_le_card_mul_iSup {α : Type} [Fintype α] (μ : PMF α) (E : α → Prop) [DecidablePred E] (n : ℕ)
    (h : (Finset.univ.filter E).card ≤ n) : prob μ E ≤ (n : ℝ≥0∞) * ⨆ x, μ x := by
  rw [prob_eq_sum, ← Finset.sum_filter]
  calc ∑ x ∈ Finset.univ.filter E, μ x ≤ ∑ _x ∈ Finset.univ.filter E, ⨆ y, μ y :=
        Finset.sum_le_sum fun x _ => le_iSup (fun y => μ y) x
    _ = ((Finset.univ.filter E).card : ℝ≥0∞) * ⨆ y, μ y := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (n : ℝ≥0∞) * ⨆ y, μ y := by gcongr

theorem iSup_uniform {α : Type} [Fintype α] [Nonempty α] :
    (⨆ x, PMF.uniformOfFintype α x) = (Fintype.card α : ℝ≥0∞)⁻¹ := by
  simp only [PMF.uniformOfFintype_apply, ciSup_const]

theorem prob_uniform_le {α : Type} [Fintype α] [Nonempty α] (E : α → Prop) [DecidablePred E] (n : ℕ)
    (h : (Finset.univ.filter E).card ≤ n) :
    prob (PMF.uniformOfFintype α) E ≤ (n : ℝ≥0∞) / Fintype.card α := by
  rw [div_eq_mul_inv, ← iSup_uniform]
  exact prob_le_card_mul_iSup _ E n h

end ResourceAccounting.TagProofs
