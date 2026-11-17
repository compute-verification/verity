module

public import Proofs.Flock.Soundness.Audit.FlockPublic
public import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkedL
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-! On the constant and the statement's zero the committed transcript is the statement's (`Xpub`), not the prover's, given `ConstCols` and `ZeroCols`. -/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

theorem linkSoundE_monoL {decode : DecoderB (C := C) (Reg := Reg) plan tab}
    {Xc Xc' : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)} {k : ℕ} (hk : 1 ≤ k)
    (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode)
    (hsub : ∀ R τ S u, u ∈ S → ∀ T, P.LinkEvB plan tab decode S R u (Xc' R τ) T →
      P.LinkEvB plan tab decode S R u (Xc R τ) T)
    {εbind : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → ℝ≥0∞}
    (h : (analysisBEL P H E Lf plan tab decode Xc k hk hr hlow).BindSound εbind) :
    (analysisBEL P H E Lf plan tab decode Xc' k hk hr hlow).BindSound εbind := by
  classical
  intro R τ tgt
  refine le_trans (avg_mono fun ω => ?_) (h R τ tgt)
  cases htg : tgt ω with
  | none => simp only [atTgt, Option.elim_none, le_refl]
  | some u =>
    simp only [atTgt, Option.elim_some]
    split_ifs with hu
    · show ENNReal.ofReal (accProbL H E Lf plan (τ ω) *
          extractProbAccBL execArith H E Lf (plan (L.draw ω) R) (tab (L.draw ω) R u) k (batchStratL H E Lf plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (Xc' R τ)) /
          (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k)) ≤
        ENNReal.ofReal (accProbL H E Lf plan (τ ω) *
          extractProbAccBL execArith H E Lf (plan (L.draw ω) R) (tab (L.draw ω) R u) k (batchStratL H E Lf plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (Xc R τ)) /
          (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k))
      apply ENNReal.ofReal_le_ofReal
      have hr1 : 0 ≤ 1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k := by
        linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
      refine div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left ?_ (expect_nonneg (ind_nonneg _) _ _)) hr1
      unfold extractProbAccBL
      refine expectAcc_mono _ _ _ _ fun bs => ?_
      unfold ind
      split_ifs with h1 h2
      · exact le_refl _
      · exact absurd (hsub R τ _ u hu _ h1) h2
      · exact zero_le_one
      · exact le_refl _
    · exact le_refl _

variable {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : P.ValueBinding plan tab decode Hc)

theorem flock_batched_linkSoundE_pubL (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode) {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ}
    (ht : 0 ≤ t') :
    (analysisBEL P (L := L) H E Lf plan tab decode (fun R τ => Xpub ones zeros (XplurL P H E Lf plan tab vb R τ k Rw)) k hk
        (fun S R j => (hr S R j).trans_lt hρ) hlow).BindSound
      (linkBoundCRL P H E Lf plan tab vb k Rw M ρ t') :=
  linkSoundE_monoL P H E Lf plan tab hk _ hlow (fun _ _ _ _ hu _ h => P.linkEvB_of_pub plan tab hpub hu h)
    (flock_batched_linkSoundEL P H E Lf plan tab vb hlow hk hRw hM hρ hr ht)

theorem flock_batched_count_publicL (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧
        K ≤ (P.incorrect (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (flock_batched_countEL P H E Lf plan tab decode (fun R τ => Xpub ones zeros (XplurL P H E Lf plan tab vb R τ k Rw)) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (flock_batched_linkSoundE_pubL P H E Lf plan tab vb ones zeros hpub hlow hk hRw hM hρ hr ht) K σ).trans_eq
    (by rw [linkBoundCR_eqL P H E Lf plan tab vb hCR])

theorem flock_batched_sampled_publicL (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧
        ¬ Disjoint (P.incorrect (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (flock_batched_sampledEL P H E Lf plan tab decode (fun R τ => Xpub ones zeros (XplurL P H E Lf plan tab vb R τ k Rw)) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (flock_batched_linkSoundE_pubL P H E Lf plan tab vb ones zeros hpub hlow hk hRw hM hρ hr ht) σ).trans_eq
    (by rw [linkBoundCR_eqL P H E Lf plan tab vb hCR])

end FlockSoundness.Discharge.Exec
