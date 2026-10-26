import Properties.Accounting.JointBridge.WorkOfCreditedWords
import Mathlib.Tactic.Linarith

/-!
# Proof of the joint period's PoUW guarantee from a credit in words

At each value `ξ` of the other audits' coins, PoUW's coordinate of the coins is the source game's (`coin`), so the
joint event, PoUW accepting while the compute account did less than `W`, maps into the source's (`pr_le_pr_of_equiv`):
the verdict is the source's acceptance (`accepts_iff`), and on an accepted period whose fixed words reach the credit,
`W ≤ m·p(tc)·credit − u·p(tc)·K/b_unit ≤ m·p(tc)·fixed − u·p(tc)·help ≤ work` (`cert_le`, `HelpWithinCertifier`,
`WordCostsMacs`), since `m·p(tc) ≥ 0` and `u·p(tc) ≥ 0`.
-/

namespace Accounting.SecurityProofs.Bridge

/-- The uniform probability of `E` on `A` is at most that of `F` on `X` when `e : A ≃ X` maps `E` into `F`. -/
theorem pr_le_pr_of_equiv {A X : Type} [Fintype A] [Fintype X] (e : A ≃ X) {E : A → Prop} {F : X → Prop}
    (h : ∀ a, E a → F (e a)) : Pouw.pr E ≤ Pouw.pr F := by
  classical
  unfold Pouw.pr
  rw [Fintype.card_congr e]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hmap : ((Finset.univ.filter E).map e.toEmbedding) ⊆ Finset.univ.filter F := by
    intro x hx
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Equiv.coe_toEmbedding] at hx ⊢
    obtain ⟨a, ha, rfl⟩ := hx
    exact h a ha
  have := Finset.card_le_card hmap
  rw [Finset.card_map] at this
  convert (Nat.cast_le (α := ℝ)).2 this using 3

/-- **The joint period's PoUW guarantee from a credit in words, at any price**
(`Properties.Accounting.JointBridge.WorkOfCreditedWords`). -/
theorem WorkOfCreditedWords : Properties.Accounting.JointBridge.WorkOfCreditedWords := by
  intro ι α I _ _ Ω _ d C r c p m u bUnit X _ acc fixed credit help δ real hcost hhelp hb ξ
  refine le_trans (pr_le_pr_of_equiv (F := fun x => acc ξ x ∧ fixed ξ x < credit ξ x) real.coin
    fun y ⟨hacc, hnot⟩ => ?_) (hb ξ)
  have hacc' := (real.accepts_iff ξ y).1 hacc
  refine ⟨hacc', lt_of_not_ge fun hle => hnot ?_⟩
  show c.W ≤ (r.work p (Function.update ξ (r.own .pouw) y) : ℝ)
  have hmp : (0 : ℝ) ≤ m * p .tc := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hup : (0 : ℝ) ≤ u * p .tc := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hcert := real.cert_le ξ y hacc'
  have hwork := hcost ξ y hacc'
  have hcred : (m : ℝ) * p .tc * credit ξ (real.coin y) ≤ m * p .tc * fixed ξ (real.coin y) :=
    mul_le_mul_of_nonneg_left hle hmp
  have hunits : (u : ℝ) * p .tc * help ξ (real.coin y) ≤ u * p .tc * c.helpUnits bUnit :=
    mul_le_mul_of_nonneg_left (hhelp ξ y hacc') hup
  linarith

end Accounting.SecurityProofs.Bridge
