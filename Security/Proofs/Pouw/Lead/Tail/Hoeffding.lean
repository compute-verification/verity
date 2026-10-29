import Mathlib.MeasureTheory.Integral.Bochner.SumMeasure
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Probability.UniformOn
import Mathlib.Tactic.LinearCombination
import Proofs.Pouw.Lead.Tail.Product

/-!
# Hoeffding's inequality over coin blocks

A sum `∑ i, f i (ω i)` over a finite product of uniform blocks, each term reading only its block and lying in
`[0, c i]`, exceeds the sum of the blocks' means by `t ≥ 0` on at most an `exp (−2t² / ∑ i, c i²)` share of the coins
(`card_sum_ge_le_hoeffding`). Each block's factor is Mathlib's Hoeffding lemma (`hasSubgaussianMGF_of_mem_Icc`) under
the block's uniform measure (`sum_exp_le_hoeffding`), and `card_sum_ge_le_of_block` multiplies the blocks at
`l = 4t / ∑ i, c i²`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open MeasureTheory ProbabilityTheory Finset
open scoped NNReal

section Block

variable {Y : Type*} [Fintype Y]

/-- Under the uniform measure on a finite block, an integral is the block's mean. -/
theorem integral_uniformOn_univ [MeasurableSpace Y] [DiscreteMeasurableSpace Y] [Nonempty Y] (g : Y → ℝ) :
    ∫ y, g y ∂(uniformOn (Set.univ : Set Y)) = (∑ y, g y) / Fintype.card Y := by
  rw [integral_fintype (μ := uniformOn (Set.univ : Set Y)) (f := g) Integrable.of_finite]
  have hy : ∀ y : Y, (uniformOn (Set.univ : Set Y)).real {y} = 1 / Fintype.card Y := fun y => by
    rw [measureReal_def, uniformOn_univ, Measure.count_singleton, ENNReal.toReal_div]
    simp
  simp only [hy, smul_eq_mul, sum_div]
  exact sum_congr rfl fun y _ => by ring

/-- `sum_exp_le_hoeffding` on a nonempty block, from Hoeffding's lemma under the block's uniform measure. -/
theorem sum_exp_le_hoeffding_aux [mY : MeasurableSpace Y] [DiscreteMeasurableSpace Y] [Nonempty Y] (g : Y → ℝ)
    {c : ℝ} (hg : ∀ y, 0 ≤ g y ∧ g y ≤ c) (l : ℝ) :
    ∑ y, Real.exp (l * (g y - (∑ z, g z) / Fintype.card Y)) ≤
      Fintype.card Y * Real.exp (c ^ 2 * l ^ 2 / 8) := by
  have h := (hasSubgaussianMGF_of_mem_Icc (μ := uniformOn (Set.univ : Set Y)) (a := 0) (b := c)
    (AEMeasurable.of_discrete (f := g)) (ae_of_all _ hg)).mgf_le l
  rw [integral_uniformOn_univ, mgf, integral_uniformOn_univ] at h
  have hc : (((‖c - 0‖₊ / 2) ^ 2 : ℝ≥0) : ℝ) = c ^ 2 / 4 := by
    rw [NNReal.coe_pow, NNReal.coe_div, coe_nnnorm, sub_zero, Real.norm_eq_abs, div_pow, sq_abs]
    norm_num
  have key : (∑ y, Real.exp (l * (g y - (∑ z, g z) / Fintype.card Y))) / Fintype.card Y ≤
      Real.exp (c ^ 2 * l ^ 2 / 8) := by
    refine h.trans_eq ?_
    congr 1
    rw [hc]
    ring
  rw [div_le_iff₀ (Nat.cast_pos.2 Fintype.card_pos)] at key
  exact key.trans_eq (mul_comm _ _)

/-- **Hoeffding's lemma on one block**: if `g` lies in `[0, c]` on a finite block, then for every `l` the block's
total of `exp (l · (g − mean g))` is at most `#Y · exp (c² l² / 8)`. -/
theorem sum_exp_le_hoeffding (g : Y → ℝ) {c : ℝ} (hg : ∀ y, 0 ≤ g y ∧ g y ≤ c) (l : ℝ) :
    ∑ y, Real.exp (l * (g y - (∑ z, g z) / Fintype.card Y)) ≤
      Fintype.card Y * Real.exp (c ^ 2 * l ^ 2 / 8) := by
  rcases isEmpty_or_nonempty Y with hY | hY
  · simp
  · exact sum_exp_le_hoeffding_aux (mY := ⊤) g hg l

end Block

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ i, Fintype (X i)]

/-- **Hoeffding over coin blocks**: if each `f i` reads only block `i` and lies in `[0, c i]`, the coins on which
`∑ i, f i (ω i)` reaches the sum of the blocks' means plus `t ≥ 0` number at most `#coins · exp (−2t² / ∑ i, c i²)`. -/
theorem card_sum_ge_le_hoeffding (f : ∀ i, X i → ℝ) (c : ι → ℝ) (hf : ∀ i x, 0 ≤ f i x ∧ f i x ≤ c i) {t : ℝ}
    (ht : 0 ≤ t) :
    (#{ω : ∀ i, X i | ∑ i, (∑ x, f i x) / Fintype.card (X i) + t ≤ ∑ i, f i (ω i)} : ℝ) ≤
      Fintype.card (∀ i, X i) * Real.exp (-2 * t ^ 2 / ∑ i, c i ^ 2) := by
  obtain ⟨C, hC⟩ : ∃ C, ∑ i, c i ^ 2 = C := ⟨_, rfl⟩
  have hC0 : 0 ≤ C := by
    rw [← hC]
    exact sum_nonneg fun i _ => sq_nonneg (c i)
  rw [hC]
  rcases hC0.eq_or_lt with h0 | hpos
  · rw [← h0, div_zero, Real.exp_zero, mul_one]
    exact_mod_cast card_le_univ _
  · have hl : 0 ≤ 4 * t / C := div_nonneg (by linarith) hpos.le
    have hB := card_sum_ge_le_of_block (fun i x => f i x - (∑ y, f i y) / Fintype.card (X i)) hl
      (fun i => c i ^ 2 * (4 * t / C) ^ 2 / 8) (fun i => sum_exp_le_hoeffding (f i) (hf i) _) t
    calc (#{ω : ∀ i, X i | ∑ i, (∑ x, f i x) / Fintype.card (X i) + t ≤ ∑ i, f i (ω i)} : ℝ)
        ≤ #{ω : ∀ i, X i | t ≤ ∑ i, (f i (ω i) - (∑ x, f i x) / Fintype.card (X i))} := by
          refine Nat.cast_le.2 (card_le_card fun ω hω => ?_)
          simp only [mem_filter, mem_univ, true_and] at hω ⊢
          rw [sum_sub_distrib]
          linarith
      _ ≤ Fintype.card (∀ i, X i) * Real.exp (-(4 * t / C * t) + ∑ i, c i ^ 2 * (4 * t / C) ^ 2 / 8) := hB
      _ = Fintype.card (∀ i, X i) * Real.exp (-2 * t ^ 2 / C) := by
          rw [← sum_div, ← sum_mul, hC]
          have hCC : C * C⁻¹ = 1 := mul_inv_cancel₀ hpos.ne'
          congr 2
          linear_combination (2 * t ^ 2 * C⁻¹) * hCC

/-- Hoeffding against upper bounds `m i` on the blocks' means. -/
theorem card_sum_ge_le_hoeffding_of_mean_le (f : ∀ i, X i → ℝ) (c m : ι → ℝ)
    (hf : ∀ i x, 0 ≤ f i x ∧ f i x ≤ c i) (hm : ∀ i, (∑ x, f i x) / Fintype.card (X i) ≤ m i) {t : ℝ}
    (ht : 0 ≤ t) :
    (#{ω : ∀ i, X i | ∑ i, m i + t ≤ ∑ i, f i (ω i)} : ℝ) ≤
      Fintype.card (∀ i, X i) * Real.exp (-2 * t ^ 2 / ∑ i, c i ^ 2) := by
  have hsum : ∑ i, (∑ x, f i x) / Fintype.card (X i) ≤ ∑ i, m i := sum_le_sum fun i _ => hm i
  refine le_trans ?_ (card_sum_ge_le_hoeffding f c hf ht)
  refine Nat.cast_le.2 (card_le_card fun ω hω => ?_)
  simp only [mem_filter, mem_univ, true_and] at hω ⊢
  linarith

end Pouw.SecurityProofs.Lead.Tail
