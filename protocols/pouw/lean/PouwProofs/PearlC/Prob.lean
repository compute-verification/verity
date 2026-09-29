import Pouw.PearlC.Basic

/-! Monotonicity of `pr`, and `pr` of an event that never happens. -/

namespace Pouw.PearlC

theorem pr_mono {α : Type} [Fintype α] {P P' : α → Prop} (h : ∀ x, P x → P' x) : pr P ≤ pr P' := by
  classical
  unfold pr
  have hsub : (Finset.univ.filter P) ⊆ (Finset.univ.filter P') := by
    intro x hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
    exact h x hx
  have hc : ((Finset.univ.filter P).card : ℝ) ≤ (Finset.univ.filter P').card := by
    exact_mod_cast Finset.card_le_card hsub
  exact div_le_div_of_nonneg_right hc (Nat.cast_nonneg _)

theorem pr_eq_zero {α : Type} [Fintype α] {P : α → Prop} (h : ∀ x, ¬ P x) : pr P = 0 := by
  classical
  unfold pr
  have hempty : Finset.univ.filter P = ∅ := Finset.filter_false_of_mem (fun x _ => h x)
  simp [hempty]

end Pouw.PearlC
