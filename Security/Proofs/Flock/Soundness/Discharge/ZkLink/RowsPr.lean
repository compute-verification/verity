import Proofs.Flock.Soundness.Discharge.ZkLink.RowsSound

/-!
# The rows link at the finders' collision probabilities

`zk_rowsLinkSoundC` bounds `analysisZCRows`' link term in closed form, `δ_link + δ_rows` each at `δ_link`'s closed form
under A2 for the link and rows finders. This file bounds it at the finders' collision probabilities, as
`zk_linkSoundPr` bounds `analysisZC`'s: the link term `linkPrZC` plus the rows term `rowsPrZC`, the audit extractor's
mass at the registered reads where it reads a value other than the registrant's.

* `rowsMassZC`, `rowsPrZC`: the rows mass at a commit string, and the rows term;
* `zk_rowsLinkSoundPr`: `analysisZCRows`' link term is at most `linkPrZC + rowsPrZC`, with no hash premise;
* `rowsPrZC_noReads`: with no registered reads the rows term is `0`, so the bound is `linkPrZC`;
* `rowsPrZC_le`: under A2 for the rows finders (`RowsCRZC`), at openings whose commitments are the registered ones, the
  rows term is at most `δ_rows`' closed form, `π·(t'(1 + k)/2^256 + k/(eR_w))/(1 − ρ)`.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc)
  (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening)

/-- **The rows mass at commit string `c`**: the audit extractor's reading, at `c` a registered read, of a value other
than the registrant's (`psiRowsZC`), jointly with acceptance and averaged over the draw. -/
noncomputable def rowsMassZC (k : ℕ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) (c : vb.Pos) : ℝ :=
  (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
      expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
        (psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω)) / Fintype.card L.Ω

/-- **The rows term, prover by prover**: `Σ_c rowsMassZC c/(1 − ρ)`. -/
noncomputable def rowsPrZC (k : ℕ) (ρ : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, rowsMassZC A Hs P planZ tab vb isRead wo k R τ c) / (1 - ρ))

/-- **The rows link at the finders' collision probabilities**: `analysisZCRows`' link term is at most
`linkPrZC + rowsPrZC`, for every prover. No hash premise. -/
theorem zk_rowsLinkSoundPr (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k : ℕ}
    {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZCRows (L := L) A Hs P planZ tab vb isRead wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).LinkSound
      (fun R τ => linkPrZC A Hs P planZ tab vb k Rw M ρ R τ + rowsPrZC A Hs P planZ tab vb isRead wo k ρ R τ) := by
  intro R τ tgt
  show _ ≤ linkPrZC A Hs P planZ tab vb k Rw M ρ R τ + rowsPrZC A Hs P planZ tab vb isRead wo k ρ R τ
  have hρ1 : 0 < 1 - ρ := by linarith
  set f : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (fun bs => psiZC A Hs P planZ tab vb R τ k Rw c ω bs + psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs)
    with hf
  set Λ : L.Ω → ℝ := fun ω => (∑ c, f c ω) / (1 - ρ) with hΛ
  have hf0 : ∀ c ω, 0 ≤ f c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ fun bs => add_nonneg (psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω bs)
      (psiRows_nonnegZC A Hs P planZ tab vb isRead wo R τ c ω bs))
  have hΛ0 : ∀ ω, 0 ≤ Λ ω := fun ω =>
    div_nonneg (Finset.sum_nonneg fun c _ => hf0 c ω) hρ1.le
  have step1 : ∀ ω, atTgt (L.draw ω) (tgt ω) (fun u =>
      (analysisZCRows (L := L) A Hs P planZ tab vb isRead wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).link (L.draw ω) R (τ ω) u
        ((analysisZCRows (L := L) A Hs P planZ tab vb isRead wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
          (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).committed R τ)) ≤ ENNReal.ofReal (Λ ω) := by
    intro ω
    refine atTgt_le (fun u hu => ?_) (tgt ω)
    simp only [analysisZCRows, ↓reduceIte]
    apply ENNReal.ofReal_le_ofReal
    have hε0 : 0 ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) :=
      expect_nonneg (ind_nonneg _) _ _
    have hLp0 := extractProbAccZ_nonneg A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
      (zcStrat A Hs planZ (τ ω))
      (RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw))
    have hLp : extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
        (zcStrat A Hs planZ (τ ω))
        (RowsLinkEvZK A P planZ tab vb isRead wo (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)) ≤
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => ∑ c, (psiZC A Hs P planZ tab vb R τ k Rw c ω bs +
              psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs)) :=
      expectAcc_mono _ _ _ _ fun bs => rowsLinkEv_le_psiZC A Hs P planZ tab vb isRead wo R τ k Rw hu bs
    have hrr : 1 - ρ ≤ 1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k := by
      linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
    have hsum : expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (fun bs => ∑ c, (psiZC A Hs P planZ tab vb R τ k Rw c ω bs +
            psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω bs)) = ∑ c, f c ω := by
      rw [expectAcc_sum, Finset.mul_sum]
    refine (div_le_div_of_nonneg_left (mul_nonneg hε0 hLp0) hρ1 hrr).trans ?_
    show _ ≤ (∑ c, f c ω) / (1 - ρ)
    rw [← hsum]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hLp hε0) hρ1.le
  refine (avg_mono step1).trans ?_
  rw [avg_ofReal Λ hΛ0]
  unfold linkPrZC rowsPrZC
  refine le_trans (ENNReal.ofReal_le_ofReal ?_) ENNReal.ofReal_add_le
  have hc : ∀ c, (∑ ω, f c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
          (linkFinderPrZC A Hs P planZ tab vb R τ c k Rw M + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) +
        rowsMassZC A Hs P planZ tab vb isRead wo k R τ c := fun c => by
    have hsplit : (∑ ω, f c ω) / Fintype.card L.Ω =
        (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (psiZC A Hs P planZ tab vb R τ k Rw c ω)) / Fintype.card L.Ω +
        rowsMassZC A Hs P planZ tab vb isRead wo k R τ c := by
      unfold rowsMassZC
      rw [← add_div, ← Finset.sum_add_distrib]
      congr 1
      refine Finset.sum_congr rfl fun ω _ => ?_
      rw [← mul_add, ← expectAcc_add]
    rw [hsplit]
    exact add_le_add (ideal_mass_le_prZC A Hs P planZ tab vb R τ k Rw M hRw hM c) le_rfl
  calc (∑ ω, Λ ω) / Fintype.card L.Ω
      = (∑ c, (∑ ω, f c ω) / Fintype.card L.Ω) / (1 - ρ) := by
        simp only [hΛ, div_eq_mul_inv, ← Finset.sum_mul]
        rw [Finset.sum_comm]
        ring
    _ ≤ (∑ c, ((Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
          (linkFinderPrZC A Hs P planZ tab vb R τ c k Rw M + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) +
        rowsMassZC A Hs P planZ tab vb isRead wo k R τ c)) / (1 - ρ) :=
        div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc c) hρ1.le
    _ = _ := by rw [Finset.sum_add_distrib, add_div]

/-- **With no registered reads the rows term is `0`**: `psiRowsZC` vanishes at every commit string. -/
theorem rowsPrZC_noReads (k : ℕ) (ρ : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) :
    rowsPrZC A Hs P planZ tab vb (fun _ => False) wo k ρ R τ = 0 := by
  have h0 : ∀ c, rowsMassZC A Hs P planZ tab vb (fun _ => False) wo k R τ c = 0 := fun c => by
    unfold rowsMassZC
    have hp : ∀ ω, psiRowsZC A Hs P planZ tab vb (fun _ => False) wo R τ c ω = fun _ => 0 := fun ω =>
      funext fun bs => by simp [psiRowsZC]
    simp only [hp, expectAcc_zero, mul_zero, Finset.sum_const_zero, zero_div]
  unfold rowsPrZC
  simp only [h0, Finset.sum_const_zero, zero_div, ENNReal.ofReal_zero]

/-- **The rows term in closed form**: under A2 for the rows finders (`RowsCRZC`), at openings `wo` whose commitments
at the registered reads are the registered ones, it is at most `π·(t'(1 + k)/2^256 + k/(eR_w))/(1 − ρ)`
(`ideal_rows_leZC`). -/
theorem rowsPrZC_le {k Rw : ℕ} (hRw : 1 ≤ Rw) {ρ : ℝ} (hρ : ρ < 1) {t' : ℝ} (ht : 0 ≤ t') (R : Reg)
    (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) (hon : ∀ c, isRead c → vb.com (wo c) = vb.cs R c)
    (hCR : RowsCRZC A Hs P planZ tab vb R τ isRead wo k Rw t') :
    rowsPrZC A Hs P planZ tab vb isRead wo k ρ R τ ≤
      ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (t' * (1 + k) / 2 ^ (256 : ℝ) + k / (Real.exp 1 * Rw))) / (1 - ρ)) := by
  have hρ1 : 0 < 1 - ρ := by linarith
  unfold rowsPrZC
  exact ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ =>
    ideal_rows_leZC A Hs P planZ tab vb isRead wo R τ k Rw hRw ht c (hon c) (hCR c)) hρ1.le)

end FlockSoundness.Discharge.ZkLink
