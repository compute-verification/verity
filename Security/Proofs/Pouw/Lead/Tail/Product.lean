import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Data.Fintype.BigOperators

/-!
# Exponential moments over a product of coin blocks

The coins are a finite product `∀ i, X i` of blocks. A sum `∑ i, F i (ω i)` whose term `F i` reads only block `i`
reaches `a` on at most `exp (−l·a)` times `∏ i, ∑ x, exp (l · F i x)` of the coins, for any `l ≥ 0` (Markov's
inequality for `exp (l · ∑ F)`, whose total over the product factors block by block). A bound on each block's factor
then bounds the count: `card_sum_ge_le_of_block`, which the Hoeffding and Chernoff forms instantiate.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ i, Fintype (X i)]

/-- **Markov over a product**: for `l ≥ 0`, the coins on which `∑ i, F i (ω i)` reaches `a` number at most
`exp (−l·a) · ∏ i, ∑ x, exp (l · F i x)`. -/
theorem card_sum_ge_le_prod (F : ∀ i, X i → ℝ) {l : ℝ} (hl : 0 ≤ l) (a : ℝ) :
    (#{ω : ∀ i, X i | a ≤ ∑ i, F i (ω i)} : ℝ) ≤ Real.exp (-(l * a)) * ∏ i, ∑ x, Real.exp (l * F i x) := by
  have h : (#{ω : ∀ i, X i | a ≤ ∑ i, F i (ω i)} : ℝ) * Real.exp (l * a) ≤ ∏ i, ∑ x, Real.exp (l * F i x) :=
    calc (#{ω : ∀ i, X i | a ≤ ∑ i, F i (ω i)} : ℝ) * Real.exp (l * a)
        = ∑ ω ∈ {ω : ∀ i, X i | a ≤ ∑ i, F i (ω i)}, Real.exp (l * a) := by rw [sum_const, nsmul_eq_mul]
      _ ≤ ∑ ω ∈ {ω : ∀ i, X i | a ≤ ∑ i, F i (ω i)}, Real.exp (l * ∑ i, F i (ω i)) :=
          sum_le_sum fun ω hω => Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left (mem_filter.1 hω).2 hl)
      _ ≤ ∑ ω : ∀ i, X i, Real.exp (l * ∑ i, F i (ω i)) :=
          sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => (Real.exp_pos _).le
      _ = ∏ i, ∑ x, Real.exp (l * F i x) := by
          rw [prod_univ_sum, Fintype.piFinset_univ]
          refine sum_congr rfl fun ω _ => ?_
          rw [mul_sum, Real.exp_sum]
  rw [Real.exp_neg, ← div_eq_inv_mul, le_div_iff₀ (Real.exp_pos _)]
  exact h

/-- **Block bounds to a count**: if each block's factor is at most `#X_i · exp (B i)`, the coins on which
`∑ i, F i (ω i)` reaches `a` number at most `#coins · exp (−l·a + ∑ i, B i)`. -/
theorem card_sum_ge_le_of_block (F : ∀ i, X i → ℝ) {l : ℝ} (hl : 0 ≤ l) (B : ι → ℝ)
    (hB : ∀ i, ∑ x, Real.exp (l * F i x) ≤ Fintype.card (X i) * Real.exp (B i)) (a : ℝ) :
    (#{ω : ∀ i, X i | a ≤ ∑ i, F i (ω i)} : ℝ) ≤
      Fintype.card (∀ i, X i) * Real.exp (-(l * a) + ∑ i, B i) :=
  calc (#{ω : ∀ i, X i | a ≤ ∑ i, F i (ω i)} : ℝ)
      ≤ Real.exp (-(l * a)) * ∏ i, ∑ x, Real.exp (l * F i x) := card_sum_ge_le_prod F hl a
    _ ≤ Real.exp (-(l * a)) * ∏ i, ((Fintype.card (X i) : ℝ) * Real.exp (B i)) :=
        mul_le_mul_of_nonneg_left
          (prod_le_prod₀ (fun i _ => sum_nonneg fun x _ => (Real.exp_pos _).le) fun i _ => hB i)
          (Real.exp_pos _).le
    _ = Fintype.card (∀ i, X i) * Real.exp (-(l * a) + ∑ i, B i) := by
        rw [prod_mul_distrib, ← Real.exp_sum, Fintype.card_pi, Nat.cast_prod, Real.exp_add]
        ring

end Pouw.SecurityProofs.Lead.Tail
