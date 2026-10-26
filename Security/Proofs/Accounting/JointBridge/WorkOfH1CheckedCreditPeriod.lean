import Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod
import Proofs.Pouw.Period

/-!
# Proof of the joint period's PoUW guarantee from H1's checked credit

At each value `ξ` of the other audits' coins, `Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod` at the
source's instance there bounds the play's acceptance with its fixed words below the check's credit. PoUW's coordinate of
the coins is the source's coins (`coin`), so the joint event, PoUW accepting while the compute account did less than
`W`, maps into the source's: the verdict is the play's (`accepts_iff`), and `W ≤ credit ≤ fixed ≤ work` on an accepted
period (`cert_le`, `work_of_fixed`) unless the fixed words fall below the credit (`pr_le_prob_of_equiv`).
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

/-- `WorkOfH1CheckedCreditPeriod` from `H1CheckedCreditPeriod`, at the source's instance at each value of the coins. -/
theorem workOfH1CheckedCreditPeriod_of (hsrc : H1CheckedCreditPeriod) :
    Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod := by
  intro ι α I _ _ Ω _ d C r c p ι' K T L U Cn Ξ V _ _ _ _ _ _ _ Sc Cd n hn hN hr hM hK prm enc staged flat δs δbind
    δks ws opened salt Wp Pl folded real hScope hρ hw hluck hgame hdead hpre hbind hreg hdraw hsvc hsalt hpins hmu hrb
    hguess hfix ξ
  have hb := hsrc Sc Cd hn hN hr hM hK prm (ws ξ) (opened ξ) (salt ξ) enc staged flat (Wp ξ) (Pl ξ) (folded ξ) δs
    δbind δks hScope hρ hw hluck (hgame ξ) (hdead ξ) (hpre ξ) (hbind ξ) (hreg ξ) (hdraw ξ) (hsvc ξ) (hsalt ξ) hpins
    (hmu ξ) (hrb ξ) (hguess ξ) (hfix ξ)
  refine le_trans (pr_le_prob_of_equiv real.coin fun y ⟨hacc, hwork⟩ => ?_) hb
  have hacc' := (real.accepts_iff ξ y).1 hacc
  refine ⟨hacc', lt_of_not_ge fun hle => hwork ?_⟩
  exact (real.cert_le ξ y hacc').trans (hle.trans (real.work_of_fixed ξ y hacc'))

/-- **The joint period's PoUW guarantee from H1's checked credit**
(`Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod`). -/
theorem WorkOfH1CheckedCreditPeriod : Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod :=
  workOfH1CheckedCreditPeriod_of Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod

end Accounting.SecurityProofs.Bridge
