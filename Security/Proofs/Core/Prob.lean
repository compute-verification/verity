module

public import Definitions.Core.Prob
import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Tactic.Linarith

/-!
# The probability of an event under a distribution on a finite type

On a finite type, `prPMFReal μ E` (`Definitions.Core.Prob`) is the sum of the weights of `E`'s outcomes
(`prPMFReal_eq_sum`, `prPMFReal_eq_sum_mul`, and `prPMFReal_mem` for a set), and the weights sum to one
(`sum_toReal_pmf`). From these: it lies in `[0, 1]`, is monotone in the event (`prPMFReal_mono`), an event's and its
complement's sum to one (`prPMFReal_add_not`), and the union bound (`prPMFReal_or_le`).
-/

@[expose] public section

namespace Verity.Game

variable {α : Type}

theorem prPMFReal_nonneg (μ : PMF α) (E : α → Prop) : 0 ≤ prPMFReal μ E :=
  ENNReal.toReal_nonneg

theorem prPMFReal_congr (μ : PMF α) {E F : α → Prop} (h : ∀ x, E x ↔ F x) : prPMFReal μ E = prPMFReal μ F := by
  rw [show E = F from funext fun x => propext (h x)]

theorem prPMFReal_false (μ : PMF α) : prPMFReal μ (fun _ => False) = 0 := by
  simp [prPMFReal, prPMF]

variable [Fintype α]

theorem prPMFReal_eq_sum (μ : PMF α) (E : α → Prop) [DecidablePred E] :
    prPMFReal μ E = ∑ x, if E x then (μ x).toReal else 0 := by
  rw [prPMFReal, prPMF, PMF.toOuterMeasure_apply, tsum_fintype,
    ENNReal.toReal_sum fun x _ => ne_top_of_le_ne_top (μ.apply_ne_top x) (Set.indicator_le_self _ _ x)]
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases h : E x <;> simp [Set.indicator, h]

/-- `prPMFReal_eq_sum` as the expectation of the event's indicator. -/
theorem prPMFReal_eq_sum_mul (μ : PMF α) (E : α → Prop) [DecidablePred E] :
    prPMFReal μ E = ∑ x, (μ x).toReal * if E x then 1 else 0 := by
  rw [prPMFReal_eq_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases h : E x <;> simp [h]

/-- The weights sum to one. -/
theorem sum_toReal_pmf (μ : PMF α) : ∑ x, (μ x).toReal = 1 := by
  have h0 : ∑ x, μ x = 1 :=
    (tsum_fintype (L := SummationFilter.unconditional α) (fun x => μ x)).symm.trans μ.tsum_coe
  have h := congrArg ENNReal.toReal h0
  rwa [ENNReal.toReal_sum fun x _ => μ.apply_ne_top x, ENNReal.toReal_one] at h

/-- The probability of landing in a set is the set's weight. -/
theorem prPMFReal_mem (μ : PMF α) (s : Finset α) : prPMFReal μ (· ∈ s) = ∑ x ∈ s, (μ x).toReal := by
  classical
  rw [prPMFReal_eq_sum, ← Finset.sum_filter, Finset.filter_mem_eq_inter, Finset.univ_inter]

theorem prPMFReal_le_one (μ : PMF α) (E : α → Prop) : prPMFReal μ E ≤ 1 := by
  classical
  rw [prPMFReal_eq_sum, ← sum_toReal_pmf μ]
  exact Finset.sum_le_sum fun x _ => by split_ifs <;> simp

theorem prPMFReal_mono (μ : PMF α) {E F : α → Prop} (h : ∀ x, E x → F x) : prPMFReal μ E ≤ prPMFReal μ F := by
  classical
  simp only [prPMFReal_eq_sum]
  refine Finset.sum_le_sum fun x _ => ?_
  by_cases hE : E x
  · simp [hE, h x hE]
  · by_cases hF : F x <;> simp [hE, hF]

theorem prPMFReal_true (μ : PMF α) : prPMFReal μ (fun _ => True) = 1 := by
  classical
  simp [prPMFReal_eq_sum, sum_toReal_pmf]

/-- An event's probability and its complement's sum to one. -/
theorem prPMFReal_add_not (μ : PMF α) (E : α → Prop) : prPMFReal μ E + prPMFReal μ (fun x => ¬ E x) = 1 := by
  classical
  simp only [prPMFReal_eq_sum]
  rw [← Finset.sum_add_distrib, ← sum_toReal_pmf μ]
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases h : E x <;> simp [h]

/-- The union bound: `P(E ∨ F) ≤ P(E) + P(F)`. -/
theorem prPMFReal_or_le (μ : PMF α) (E F : α → Prop) :
    prPMFReal μ (fun x => E x ∨ F x) ≤ prPMFReal μ E + prPMFReal μ F := by
  classical
  simp only [prPMFReal_eq_sum]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun x _ => ?_
  have hq : 0 ≤ (μ x).toReal := ENNReal.toReal_nonneg
  by_cases hE : E x <;> by_cases hF : F x <;> simp [hE, hF] <;> linarith

end Verity.Game
