import Definitions.Pouw.Basic.Prob
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Logic.Equiv.Prod
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Data.Fintype.BigOperators

/-!
# Lemmas about `pr`
-/

namespace Pouw

open Finset

variable {α β : Type} [Fintype α] [Fintype β]

theorem pr_eq_card (P : α → Prop) (s : Finset α) (hs : ∀ x, x ∈ s ↔ P x) :
    pr P = (s.card : ℝ) / Fintype.card α := by
  unfold pr
  congr 3
  ext x
  simp [hs]

theorem pr_nonneg (P : α → Prop) : 0 ≤ pr P := by
  unfold pr; positivity

theorem pr_le_one (P : α → Prop) : pr P ≤ 1 := by
  classical
  unfold pr
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast (Finset.card_filter_le _ _).trans (by simp)

theorem pr_mono {P P' : α → Prop} (h : ∀ x, P x → P' x) : pr P ≤ pr P' := by
  classical
  unfold pr
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact_mod_cast Finset.card_le_card (Finset.monotone_filter_right _ (fun x _ => h x))

theorem pr_eq_zero {P : α → Prop} (h : ∀ x, ¬ P x) : pr P = 0 := by
  classical
  unfold pr
  simp [Finset.filter_false_of_mem (fun x _ => h x)]

theorem pr_eq_one [Nonempty α] {P : α → Prop} (h : ∀ x, P x) : pr P = 1 := by
  classical
  unfold pr
  rw [Finset.filter_true_of_mem (fun x _ => h x), Finset.card_univ]
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  field_simp

theorem pr_or_le (P P' : α → Prop) : pr (fun x => P x ∨ P' x) ≤ pr P + pr P' := by
  classical
  unfold pr
  rw [← add_div]
  apply div_le_div_of_nonneg_right _ (by positivity)
  norm_cast
  refine (Finset.card_le_card ?_).trans (Finset.card_union_le _ _)
  intro x hx
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at hx ⊢
  exact hx

theorem pr_equiv (e : α ≃ β) (P : β → Prop) : pr (fun x => P (e x)) = pr P := by
  classical
  unfold pr
  rw [Fintype.card_congr e]
  congr 2
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact_mod_cast Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl)

/-- An event that reads only the first component has at most its marginal probability. -/
theorem pr_fst_le (P : α → Prop) : pr (fun x : α × β => P x.1) ≤ pr P := by
  classical
  unfold pr
  have hf : (univ.filter fun x : α × β => P x.1) = (univ.filter P) ×ˢ (univ : Finset β) := by
    ext x; simp
  rw [hf, Finset.card_product, Fintype.card_prod, Finset.card_univ]
  rcases Nat.eq_zero_or_pos (Fintype.card β) with h | h
  · simp [h]; positivity
  · push_cast
    rw [mul_div_mul_right _ _ (by exact_mod_cast h.ne')]

/-- An event on `α × β × γ` that reads only the first two components. -/
theorem pr_fst2_le {γ : Type} [Fintype γ] (P : α × β → Prop) :
    pr (fun x : α × β × γ => P (x.1, x.2.1)) ≤ pr P :=
  (pr_equiv (Equiv.prodAssoc α β γ).symm (fun y => P y.1)).le.trans (pr_fst_le P)

/-- If every section `x.1 = a` has probability at most `p ≥ 0`, so has the whole event. -/
theorem pr_prod_le (P : α × β → Prop) (p : ℝ) (hp : 0 ≤ p) (h : ∀ a, pr (fun b => P (a, b)) ≤ p) :
    pr P ≤ p := by
  classical
  unfold pr at *
  rcases Nat.eq_zero_or_pos (Fintype.card β) with hb | hb
  · simp [Fintype.card_prod, hb, hp]
  rcases Nat.eq_zero_or_pos (Fintype.card α) with ha | ha
  · simp [Fintype.card_prod, ha, hp]
  have key : ∀ a, ((univ.filter fun b => P (a, b)).card : ℝ) ≤ p * Fintype.card β := by
    intro a
    have := h a
    rwa [div_le_iff₀ (by exact_mod_cast hb)] at this
  have hsum : (univ.filter P).card = ∑ a, (univ.filter fun b => P (a, b)).card := by
    simp only [Finset.card_filter]
    rw [Fintype.sum_prod_type]
  rw [div_le_iff₀ (by rw [Fintype.card_prod]; exact_mod_cast Nat.mul_pos ha hb), hsum, Fintype.card_prod]
  push_cast
  calc ∑ a, ((univ.filter fun b => P (a, b)).card : ℝ) ≤ ∑ _a : α, p * Fintype.card β :=
        Finset.sum_le_sum fun a _ => key a
    _ = p * (Fintype.card α * Fintype.card β) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

end Pouw
