module

public import Proofs.Flock.Soundness.Discharge.ZkLink.Public
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Audit.Circuit
import Proofs.Flock.Soundness.Game

/-! `RightRegZc`: right on the registration with canonical openings, so fewer units are right than under `RightRegZ`. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) (canon : Fin C.N → vb.Opening → Prop)

/-- **Unit `u` is right on the registration `R` with canonical openings**: `RightRegZ`, with each wire's opening
canonical at the wire (`canon g`). -/
def RightRegZc (R : Reg) (Xv : Fin C.N → Bool) (u : Fin n) : Prop :=
  P.Correct Xv u ∧ ∃ op : vb.Pos → vb.Opening, ∀ g ∈ P.io u,
    vb.com (op (vb.pos g)) = vb.cs R (vb.pos g) ∧ vb.bit (vb.val (op (vb.pos g))) g = C.eval Xv g ∧
      canon g (op (vb.pos g))

open Classical in
/-- **The units incorrect on the registration `R` with canonical openings**: those not right on it (`RightRegZc`). -/
noncomputable def incorrectRegZc (R : Reg) (Xv : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ RightRegZc A P planZ tab vb canon R Xv u

omit [Fintype K] [DecidableEq K] in
theorem mem_incorrectRegZc {R : Reg} {Xv : Fin C.N → Bool} {u : Fin n} :
    u ∈ incorrectRegZc A P planZ tab vb canon R Xv ↔ ¬ RightRegZc A P planZ tab vb canon R Xv u := by
  classical
  simp [incorrectRegZc]

omit [Fintype K] [DecidableEq K] in
theorem rightRegZ_of_rightRegZc {R : Reg} {Xv : Fin C.N → Bool} {u : Fin n}
    (h : RightRegZc A P planZ tab vb canon R Xv u) : RightRegZ A P planZ tab vb R Xv u :=
  ⟨h.1, h.2.imp fun _ hop g hg => ⟨(hop g hg).1, (hop g hg).2.1⟩⟩

omit [Fintype K] [DecidableEq K] in
theorem incorrectRegZ_subset_incorrectRegZc (R : Reg) (Xv : Fin C.N → Bool) :
    incorrectRegZ A P planZ tab vb R Xv ⊆ incorrectRegZc A P planZ tab vb canon R Xv :=
  fun _ hu => (mem_incorrectRegZc A P planZ tab vb canon).2 fun h =>
    (mem_incorrectRegZ A P planZ tab vb).1 hu (rightRegZ_of_rightRegZc A P planZ tab vb canon h)

def OpeningsCanon : Prop :=
  ∀ S R u, u ∈ S → ∀ M, (tabOfZ planZ tab S R u).S.Satisfies
      (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
    ∀ g ∈ P.io u, canon g (vb.opening S R u M (vb.pos g))

variable {L : SamplingDesign n}

open Classical in
noncomputable def analysisZCRegC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) (k : ℕ)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u) (hcan : OpeningsCanon A P planZ tab vb canon)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (hj : JointZC A Hs planZ k a) :
    ExtractionAnalysis L Reg (batchedSessionZC A Hs planZ) where
  Tr := Reg × (Fin C.N → Bool)
  incorrect Xt := incorrectRegZc A P planZ tab vb canon Xt.1 Xt.2
  committed R τ := (R, Xc R τ)
  ks S R τ u := ENNReal.ofReal (a S R (zcStrat A Hs planZ τ) (tab S R u) / (1 - rateZ (planZ S R) (tab S R u) k))
  link S R τ u Xt := if Xt.1 = R then linkTermZC A Hs P planZ tab decode k S R τ u Xt.2 else ⊤
  cover S R τ u Xt hS hu := by
    obtain ⟨R', Xt⟩ := Xt
    by_cases hR : R' = R
    swap
    · simp only [hR, ↓reduceIte, add_top, le_top]
    subst hR
    simp only [↓reduceIte]
    have hG : ∀ T, (tabOfZ planZ tab S R' u).Committed A T → LinkEvZK A P planZ tab decode S R' u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        push Not at hno
        have hcX : P.Correct Xt u := P.correct_congr (fun g hg => (hno g hg).symm) (hlow S R' u hS M hsat)
        refine (mem_incorrectRegZc A P planZ tab vb canon).1 hu ⟨hcX, fun c => vb.opening S R' u M c, fun g hg => ?_⟩
        obtain ⟨h1, h2⟩ := vb.opening_spec S R' u hS M hsat g hg
        exact ⟨h1, h2.trans ((hno g hg).trans (P.correct_eval (hown u) hcX g hg)), hcan S R' u hS M hsat g hg⟩⟩
    set ε := accProbZC A Hs (planZ S R') (zcStrat A Hs planZ τ)
    set r := rateZ (planZ S R') (tab S R' u) k
    set a' := a S R' (zcStrat A Hs planZ τ) (tab S R' u)
    set Lp := extractProbAccZ A Hs (planZ S R') (tab S R' u) k (zcStrat A Hs planZ τ)
      (LinkEvZK A P planZ tab decode S R' u Xt)
    have hr1 : 0 < 1 - r := by linarith [hr S R' (tab S R' u)]
    have ha : 0 ≤ a' := (hj S R' _ (tab S R' u)).1
    have hL : 0 ≤ Lp := extractProbAccZ_nonneg _ _ _ _ _ _ _
    have hε0 : 0 ≤ ε := accProbZC_nonneg _ _ _ _
    have key : ε ≤ a' / (1 - r) + ε * Lp / (1 - r) := by
      rcases hε0.eq_or_lt with h0 | hpos
      · rw [← h0]
        positivity
      · have h1 := one_le_fail_add_accZ A Hs (planZ S R') (tab S R' u) k (zcStrat A Hs planZ τ) _ hG hpos
        have h2 := (hj S R' (zcStrat A Hs planZ τ) (tab S R' u)).2
        rw [← add_div, le_div_iff₀ hr1]
        nlinarith
    show prob (· = true) (batchedSessionZC A Hs planZ S R') τ ≤
      ENNReal.ofReal (a' / (1 - r)) + ENNReal.ofReal (ε * Lp / (1 - r))
    rw [prob_batchedSessionZC_eq, ← ENNReal.ofReal_add (div_nonneg ha hr1.le)
      (div_nonneg (mul_nonneg hε0 hL) hr1.le)]
    exact ENNReal.ofReal_le_ofReal key

omit [Fintype K] in
theorem zc_knowledgeSound_regC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u) (hcan : OpeningsCanon A P planZ tab vb canon)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZCRegC A Hs P planZ tab vb canon Xc k hr hlow hown hcan a hj).KnowledgeSound
      (ksAvgZC A Hs planZ k a) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (planZ (L.draw ω) R).length =>
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k)))
    (tab (L.draw ω) R u)

theorem zk_linkSoundC_regC
    (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) (hcan : OpeningsCanon A P planZ tab vb canon)
    {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') :
    (analysisZCRegC (L := L) A Hs P planZ tab vb canon (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan a hj).BindSound
      (linkBoundZC A Hs P planZ tab vb k Rw M ρ t') := by
  intro R τ tgt
  refine le_of_eq_of_le (congrArg avg (funext fun ω => ?_))
    (zk_linkSoundC A Hs P planZ tab vb hlow hRw hM hρ hr hj ht R τ tgt)
  simp only [analysisZCRegC, analysisZC, ↓reduceIte]

theorem zk_flock_countC_regC (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u)
    (hcan : OpeningsCanon A P planZ tab vb canon) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        K ≤ (incorrectRegZc A P planZ tab vb canon (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  extraction_audit_count (analysisZCRegC A Hs P planZ tab vb canon (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan a hj)
    (zc_knowledgeSound_regC A Hs P planZ tab vb canon _ _ hlow hown hcan hj)
    (zk_linkSoundC_regC A Hs P planZ tab vb canon hlow hown hcan hRw hM hρ hr hj ht) K σ

theorem zk_flock_escapeC_regC (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u)
    (hcan : OpeningsCanon A P planZ tab vb canon) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (W : Finset (Fin n))
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        W ⊆ incorrectRegZc A P planZ tab vb canon (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.missProb W + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) := by
  refine (extraction_audit_le (analysisZCRegC A Hs P planZ tab vb canon
      (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan a hj)
    (zc_knowledgeSound_regC A Hs P planZ tab vb canon _ _ hlow hown hcan hj)
    (zk_linkSoundC_regC A Hs P planZ tab vb canon hlow hown hcan hRw hM hρ hr hj ht)
    (fun (rx : Reg × (Fin C.N → Bool)) _ => W ⊆ incorrectRegZc A P planZ tab vb canon rx.1 rx.2) σ).trans
    (add_le_add (add_le_add ?_ le_rfl) le_rfl)
  exact prCoin_mono fun ω h => h.2.mono_right h.1

end FlockSoundness.Discharge.ZkLink
