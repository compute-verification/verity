import Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Proofs.Accounting.JointBridge.WorkOfCreditedWords
import Proofs.Pouw.Period

/-!
# Proof of the joint period's PoUW guarantee from H1's checked credit

`Accounting.SecurityProofs.Bridge.WorkOfCreditedWords` at H1's instance: at each value `ξ` of the other audits' coins,
`Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod` at the source's instance there bounds the play's
acceptance with `n` times its fixed words below the check's credit, which is the bound the bridge takes, read through
`Pouw.pr` (`pr_le_prob_of_equiv` at the identity). The H200's instance is it at `Accounting.h200S8Price`,
`receiptWordMacs R C` multiply-adds a word and `receiptUnitMacs K R C` a unit.
-/

namespace Accounting.SecurityProofs.Bridge

open Properties.Pouw.Period

/-- The uniform probability of `E` on `A` is at most that of `F` on `X` when `e : A ≃ X` maps `E` into `F`. -/
theorem pr_le_prob_of_equiv {A X : Type} [Fintype A] [Fintype X] (e : A ≃ X) {E : A → Prop} {F : X → Prop}
    (h : ∀ a, E a → F (e a)) : Pouw.pr E ≤ Pouw.RelativeSecurity.prob F := by
  classical
  unfold Pouw.pr Pouw.RelativeSecurity.prob
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

/-- `WorkOfH1CheckedCreditPeriodAt p macs u b_unit` from `H1CheckedCreditPeriod`, at the source's instance at each
value of the coins. -/
theorem workOfH1CheckedCreditPeriodAt_of (hsrc : H1CheckedCreditPeriod) (p : Price) (macs : Compose.Kind → ℕ)
    (u bUnit : ℕ) : Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriodAt p macs u bUnit := by
  intro ι α I _ _ Ω _ d C r c ι' K T L U Cn Ξ V _ _ _ _ _ _ _ Sc Cd n hn hN hr hM hK prm enc staged flat δs δbind
    δks ws opened salt Wp Pl folded help real hcost hhelp hScope hρ hw hluck hgame hdead hpre hbind hreg hdraw hsvc
    hsalt hpins hmu hrb hguess hfix
  refine WorkOfCreditedWords ι α I Ω d C r c p _ u bUnit _ _ (fun ξ x => n * ((Pl ξ).run x).fixed) _ help
    (δs + n * (δbind + δks)) real hcost hhelp fun ξ => ?_
  have hs := hsrc Sc Cd hn hN hr hM hK prm (ws ξ) (opened ξ) (salt ξ) enc staged flat (Wp ξ) (Pl ξ) (folded ξ) δs
    δbind δks hScope hρ hw hluck (hgame ξ) (hdead ξ) (hpre ξ) (hbind ξ) (hreg ξ) (hdraw ξ) (hsvc ξ) (hsalt ξ) hpins
    (hmu ξ) (hrb ξ) (hguess ξ) (hfix ξ)
  refine le_trans ?_ hs
  exact pr_le_prob_of_equiv (Equiv.refl _) fun _ h => h

/-- **The joint period's PoUW guarantee from H1's checked credit**, at every price, per-kind multiply-add count, unit
count and unit width (`Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod`). -/
theorem WorkOfH1CheckedCreditPeriod : Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod :=
  workOfH1CheckedCreditPeriodAt_of Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod

/-- **The joint period's PoUW guarantee from H1's checked credit on the H200, at INT8**
(`Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriodH200`). -/
theorem WorkOfH1CheckedCreditPeriodH200 : Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriodH200 :=
  fun sha fetch K R C bUnit =>
    WorkOfH1CheckedCreditPeriod (h200S8Price sha fetch) (fun _ => receiptWordMacs R C) (receiptUnitMacs K R C) bUnit

end Accounting.SecurityProofs.Bridge
