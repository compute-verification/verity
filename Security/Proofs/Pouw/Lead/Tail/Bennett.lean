import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Tactic.LinearCombination
import Proofs.Pouw.Lead.Tail.Product

/-!
# Chernoff's and Bennett's bounds over independent coin blocks

For counts of rare events Hoeffding's slack, of order `c · √(n · ln (1/η))` over `n` blocks, is far above the mean;
Chernoff's bound pays a constant factor on the mean and `c · ln (1/η)` instead. If each `f i` reads only block `i`, lies
in `[0, c]` and averages at most `m i` over its block, then for every `l ≥ 0` the coins on which `∑ i, f i (ω i)`
reaches `a` are at most an `exp (−l·a + (e^{l c} − 1)/c · ∑ i, m i)` share (`card_sum_ge_le_chernoff`); at `l = 1/c`,
the sum reaches `(e − 1) · ∑ i, m i + s` on at most an `exp (−s/c)` share (`card_sum_ge_le_chernoff_one`); at
`l = ln (1 + s/M)/c`, `M = ∑ i, m i`, it reaches `M + s` on at most an `exp (−(M/c) · h (s/M))` share,
`h u = (1 + u) · ln (1 + u) − u` (`card_sum_ge_le_bennett`).

These need independent blocks; `Chernoff`'s `count_tail` needs only sequential domination.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Finset

/-- **The chord of `exp`**: on `[0, c]`, `x ↦ exp (l·x)` lies below its chord, `1 + (e^{l c} − 1) · x/c`. -/
theorem exp_mul_le_chord {l c x : ℝ} (hx0 : 0 ≤ x) (hxc : x ≤ c) :
    Real.exp (l * x) ≤ 1 + (Real.exp (l * c) - 1) * (x / c) := by
  rcases (hx0.trans hxc).eq_or_lt with hc | hc
  · subst hc
    obtain rfl : x = 0 := le_antisymm hxc hx0
    simp
  · have hs0 : 0 ≤ x / c := div_nonneg hx0 hc.le
    have hs1 : x / c ≤ 1 := (div_le_one₀ hc).2 hxc
    have h := convexOn_exp.2 (Set.mem_univ 0) (Set.mem_univ (l * c)) (sub_nonneg.2 hs1) hs0 (by ring)
    calc Real.exp (l * x) = Real.exp ((1 - x / c) • (0 : ℝ) + (x / c) • (l * c)) := by
          congr 1
          rw [smul_zero, zero_add, smul_eq_mul, div_mul_eq_mul_div, eq_div_iff hc.ne']
          ring
      _ ≤ (1 - x / c) • Real.exp 0 + (x / c) • Real.exp (l * c) := h
      _ = 1 + (Real.exp (l * c) - 1) * (x / c) := by
          rw [smul_eq_mul, smul_eq_mul, Real.exp_zero]
          ring

/-- A block's total is at most its size times any bound on its mean. -/
theorem sum_le_card_mul_of_mean_le {Y : Type*} [Fintype Y] {g : Y → ℝ} {m : ℝ}
    (h : (∑ y, g y) / Fintype.card Y ≤ m) : ∑ y, g y ≤ Fintype.card Y * m := by
  rcases isEmpty_or_nonempty Y with hY | hY
  · simp
  · exact (div_le_iff₀' (Nat.cast_pos.2 Fintype.card_pos)).1 h

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ i, Fintype (X i)]

/-- **Chernoff over coin blocks**: if each `f i` reads only block `i`, lies in `[0, c]` and averages at most `m i`
over its block, then for `l ≥ 0` the coins on which `∑ i, f i (ω i)` reaches `a` number at most
`#coins · exp (−l·a + (e^{l c} − 1)/c · ∑ i, m i)`. -/
theorem card_sum_ge_le_chernoff (f : ∀ i, X i → ℝ) {c : ℝ} (m : ι → ℝ) (hf : ∀ i x, 0 ≤ f i x ∧ f i x ≤ c)
    (hm : ∀ i, (∑ x, f i x) / Fintype.card (X i) ≤ m i) {l : ℝ} (hl : 0 ≤ l) (a : ℝ) :
    (#{ω : ∀ i, X i | a ≤ ∑ i, f i (ω i)} : ℝ) ≤
      Fintype.card (∀ i, X i) * Real.exp (-(l * a) + (Real.exp (l * c) - 1) / c * ∑ i, m i) := by
  have hK : 0 ≤ (Real.exp (l * c) - 1) / c := by
    rcases le_or_gt 0 c with hc | hc
    · exact div_nonneg (sub_nonneg.2 (Real.one_le_exp (mul_nonneg hl hc))) hc
    · exact div_nonneg_of_nonpos
        (sub_nonpos.2 (Real.exp_le_one_iff.2 (mul_nonpos_of_nonneg_of_nonpos hl hc.le))) hc.le
  have hB : ∀ i, ∑ x, Real.exp (l * f i x) ≤
      Fintype.card (X i) * Real.exp ((Real.exp (l * c) - 1) / c * m i) := fun i =>
    calc ∑ x, Real.exp (l * f i x) ≤ ∑ x, (1 + (Real.exp (l * c) - 1) * (f i x / c)) :=
          sum_le_sum fun x _ => exp_mul_le_chord (hf i x).1 (hf i x).2
      _ = Fintype.card (X i) + (Real.exp (l * c) - 1) / c * ∑ x, f i x := by
          rw [sum_add_distrib, sum_const, card_univ, nsmul_eq_mul, mul_one, mul_sum]
          congr 1
          exact sum_congr rfl fun x _ => by ring
      _ ≤ Fintype.card (X i) + (Real.exp (l * c) - 1) / c * (Fintype.card (X i) * m i) :=
          add_le_add le_rfl (mul_le_mul_of_nonneg_left (sum_le_card_mul_of_mean_le (hm i)) hK)
      _ = Fintype.card (X i) * ((Real.exp (l * c) - 1) / c * m i + 1) := by ring
      _ ≤ Fintype.card (X i) * Real.exp ((Real.exp (l * c) - 1) / c * m i) :=
          mul_le_mul_of_nonneg_left (Real.add_one_le_exp _) (Nat.cast_nonneg _)
  calc (#{ω : ∀ i, X i | a ≤ ∑ i, f i (ω i)} : ℝ)
      ≤ Fintype.card (∀ i, X i) * Real.exp (-(l * a) + ∑ i, (Real.exp (l * c) - 1) / c * m i) :=
        card_sum_ge_le_of_block f hl _ hB a
    _ = Fintype.card (∀ i, X i) * Real.exp (-(l * a) + (Real.exp (l * c) - 1) / c * ∑ i, m i) := by
        rw [mul_sum]

/-- **Chernoff at `l = 1/c`**: the coins on which `∑ i, f i (ω i)` reaches `(e − 1) · ∑ i, m i + s` number at most
`#coins · exp (−s/c)`. -/
theorem card_sum_ge_le_chernoff_one (f : ∀ i, X i → ℝ) {c : ℝ} (hc : 0 < c) (m : ι → ℝ)
    (hf : ∀ i x, 0 ≤ f i x ∧ f i x ≤ c) (hm : ∀ i, (∑ x, f i x) / Fintype.card (X i) ≤ m i) (s : ℝ) :
    (#{ω : ∀ i, X i | (Real.exp 1 - 1) * ∑ i, m i + s ≤ ∑ i, f i (ω i)} : ℝ) ≤
      Fintype.card (∀ i, X i) * Real.exp (-(s / c)) := by
  refine (card_sum_ge_le_chernoff f m hf hm (one_div_nonneg.2 hc.le) _).trans_eq ?_
  rw [one_div_mul_cancel hc.ne']
  congr 2
  ring

/-- **Bennett's form**, Chernoff at `l = ln (1 + s/M)/c` for `M = ∑ i, m i > 0`: the coins on which `∑ i, f i (ω i)`
reaches `M + s` (`s ≥ 0`) number at most `#coins · exp (−(M/c) · h (s/M))`, `h u = (1 + u) · ln (1 + u) − u`. -/
theorem card_sum_ge_le_bennett (f : ∀ i, X i → ℝ) {c : ℝ} (hc : 0 < c) (m : ι → ℝ)
    (hf : ∀ i x, 0 ≤ f i x ∧ f i x ≤ c) (hm : ∀ i, (∑ x, f i x) / Fintype.card (X i) ≤ m i)
    (hM : 0 < ∑ i, m i) {s : ℝ} (hs : 0 ≤ s) :
    (#{ω : ∀ i, X i | ∑ i, m i + s ≤ ∑ i, f i (ω i)} : ℝ) ≤
      Fintype.card (∀ i, X i) * Real.exp (-((∑ i, m i) / c *
        ((1 + s / ∑ i, m i) * Real.log (1 + s / ∑ i, m i) - s / ∑ i, m i))) := by
  obtain ⟨M, hMdef⟩ : ∃ M, ∑ i, m i = M := ⟨_, rfl⟩
  rw [hMdef] at hM ⊢
  have hu : 1 ≤ 1 + s / M := le_add_of_nonneg_right (div_nonneg hs hM.le)
  have hl : 0 ≤ Real.log (1 + s / M) / c := div_nonneg (Real.log_nonneg hu) hc.le
  refine (card_sum_ge_le_chernoff f m hf hm hl (M + s)).trans_eq ?_
  rw [hMdef, div_mul_cancel₀ _ hc.ne', Real.exp_log (zero_lt_one.trans_le hu)]
  have hMM : M * M⁻¹ = 1 := mul_inv_cancel₀ hM.ne'
  congr 2
  linear_combination (Real.log (1 + s / M) * s * c⁻¹) * hMM

end Pouw.SecurityProofs.Lead.Tail
