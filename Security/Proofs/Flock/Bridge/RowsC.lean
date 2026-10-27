import Proofs.Flock.Bridge.RightRegZc
import Proofs.Flock.Soundness.Discharge.ZkLink.RowsPr

/-!
# The rows link with canonical openings

`analysisZCRows` counts a drawn unit wrong when it is wrong on the registration (`wrongRegZ`) or off the registrant's
rows (`rowsWrongZ`). `analysisZCRowsC` counts it wrong when it is wrong on the registration with canonical openings
(`wrongRegZc`, under `OpeningsCanon`) or off the registrant's rows. Its knowledge and link terms are `analysisZCRows`',
and its `cover` is the two arguments joined: the unit's openings from a satisfying message are canonical, so a unit
wrong with canonical openings has a close satisfying message that decodes off the committed transcript, and a unit off
the registrant's rows has one that opens a registered read to another row.

* `analysisZCRowsC`, `zc_knowledgeSound_rowsC`, `zk_rowsLinkSoundPr_c`: the analysis, and its terms, `ksAvgZC` and
  `linkPrZC + rowsPrZC`, the finders' collision probabilities (`zk_rowsLinkSoundPr`), with no hash premise;
* `zk_flock_countPr_rowsC`: the count curve (`Audit.extraction_audit_count`) at it.
* `rowsFinderPrZC`, `ideal_rows_le_prZC`, `rowsPrZC_le_pr`: the rows term at the rows finders' collision probabilities,
  at openings `wo` whose commitments at the registered reads are the registered ones (as `rowsPrZC_le` is under A2).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) (canon : Fin C.N → vb.Opening → Prop)
  (isRead : vb.Pos → Prop) (wo : vb.Pos → vb.Opening)

variable {L : Law n}

open Classical in
/-- **`analysisZCRows` with canonical openings**: "wrong" is `wrongRegZc canon ∪ rowsWrongZ`, under `OpeningsCanon`;
its knowledge and link terms are `analysisZCRows`'. -/
noncomputable def analysisZCRowsC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) (k : ℕ)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u) (hcan : OpeningsCanon A P planZ tab vb canon)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (hj : JointZC A Hs planZ k a) :
    ExtractionAnalysis L Reg (batchedSessionZC A Hs planZ) where
  Tr := Reg × (Fin C.N → Bool)
  wrong Xt := wrongRegZc A P planZ tab vb canon Xt.1 Xt.2 ∪ rowsWrongZ A P planZ tab vb isRead wo Xt.2
  committed R τ := (R, Xc R τ)
  ks S R τ u := ENNReal.ofReal (a S R (zcStrat A Hs planZ τ) (tab S R u) / (1 - rateZ (planZ S R) (tab S R u) k))
  link S R τ u Xt := if Xt.1 = R then
      ENNReal.ofReal (accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ) *
        extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ)
          (RowsLinkEvZK A P planZ tab vb isRead wo S R u Xt.2) / (1 - rateZ (planZ S R) (tab S R u) k))
    else ⊤
  cover S R τ u Xt hS hu := by
    obtain ⟨R', Xt⟩ := Xt
    by_cases hR : R' = R
    swap
    · simp only [hR, ↓reduceIte, add_top, le_top]
    subst hR
    simp only [↓reduceIte]
    have hG : ∀ T, (tabOfZ planZ tab S R' u).Committed A T →
        RowsLinkEvZK A P planZ tab vb isRead wo S R' u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        have hno1 : ∀ g ∈ P.io u, decode S R' u M g = Xt g := fun g hg => by
          by_contra h; exact hno (Or.inl ⟨g, hg, h⟩)
        have hno2 : ∀ g ∈ P.io u, isRead (vb.pos g) →
            vb.val (vb.opening S R' u M (vb.pos g)) = vb.val (wo (vb.pos g)) := fun g hg hrd => by
          by_contra h; exact hno (Or.inr ⟨g, hg, hrd, h⟩)
        have hcX : P.Correct Xt u := P.correct_congr (fun g hg => (hno1 g hg).symm) (hlow S R' u hS M hsat)
        rcases Finset.mem_union.1 hu with hw | hw
        · refine (mem_wrongRegZc A P planZ tab vb canon).1 hw ⟨hcX, fun c => vb.opening S R' u M c, fun g hg => ?_⟩
          obtain ⟨h1, h2⟩ := vb.opening_spec S R' u hS M hsat g hg
          exact ⟨h1, h2.trans ((hno1 g hg).trans (P.correct_eval (hown u) hcX g hg)), hcan S R' u hS M hsat g hg⟩
        · obtain ⟨g, hg, hrd, hne⟩ := (mem_rowsWrongZ A P planZ tab vb isRead wo).1 hw
          exact hne ((hno1 g hg).symm.trans ((vb.opening_spec S R' u hS M hsat g hg).2.symm.trans
            (congrArg (fun v => vb.bit v g) (hno2 g hg hrd))))⟩
    set ε := accProbZC A Hs (planZ S R') (zcStrat A Hs planZ τ)
    set r := rateZ (planZ S R') (tab S R' u) k
    set a' := a S R' (zcStrat A Hs planZ τ) (tab S R' u)
    set Lp := extractProbAccZ A Hs (planZ S R') (tab S R' u) k (zcStrat A Hs planZ τ)
      (RowsLinkEvZK A P planZ tab vb isRead wo S R' u Xt)
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
/-- `analysisZCRowsC`'s knowledge term is `analysisZC`'s, `ksAvgZC`. -/
theorem zc_knowledgeSound_rowsC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u) (hcan : OpeningsCanon A P planZ tab vb canon)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZCRowsC A Hs P planZ tab vb canon isRead wo Xc k hr hlow hown hcan a hj).KnowledgeSound
      (ksAvgZC A Hs planZ k a) :=
  zc_knowledgeSound_rows A Hs P planZ tab vb isRead wo Xc hr hlow hown hj

/-- `analysisZCRowsC`'s link term is `analysisZCRows`', so the rows link at the finders' collision probabilities
(`zk_rowsLinkSoundPr`) bounds it. -/
theorem zk_rowsLinkSoundPr_c (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u)
    (hcan : OpeningsCanon A P planZ tab vb canon) {k : ℕ}
    {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZCRowsC (L := L) A Hs P planZ tab vb canon isRead wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan a hj).LinkSound
      (fun R τ => linkPrZC A Hs P planZ tab vb k Rw M ρ R τ + rowsPrZC A Hs P planZ tab vb isRead wo k ρ R τ) :=
  zk_rowsLinkSoundPr A Hs P planZ tab vb isRead wo hlow hown hRw hM hρ hr hj

/-- **The compiled `--zk` count curve, against the registration with canonical openings and the registrant's rows**:
it accepts with at least `K` units in `wrongRegZc canon ∪ rowsWrongZ` with probability at most
`miss K + ε_ks + linkPrZC + rowsPrZC`. No hash premise. -/
theorem zk_flock_countPr_rowsC (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u)
    (hcan : OpeningsCanon A P planZ tab vb canon) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        K ≤ (wrongRegZc A P planZ tab vb canon (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw) ∪
          rowsWrongZ A P planZ tab vb isRead wo (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        (linkPrZC A Hs P planZ tab vb k Rw M ρ (reg σ) (cont σ) +
          rowsPrZC A Hs P planZ tab vb isRead wo k ρ (reg σ) (cont σ)) :=
  extraction_audit_count (analysisZCRowsC A Hs P planZ tab vb canon isRead wo
      (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan a hj)
    (zc_knowledgeSound_rowsC A Hs P planZ tab vb canon isRead wo _ _ hlow hown hcan hj)
    (zk_rowsLinkSoundPr_c A Hs P planZ tab vb canon isRead wo hlow hown hcan hRw hM hρ hr hj) K σ

section RowsFinder

variable (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

open Classical in
/-- **The rows finder's collision probability at `c`**: the probability that the finder that plays the trial at `c`
once and outputs `rowsOutZC` outputs a collision of `Hc` (`0` at a commit string no draw reads, which has no trial). -/
noncomputable def rowsFinderPrZC (c : vb.Pos) (k Rw : ℕ) : ℝ :=
  if _ : Nonempty (ΩcZ A P planZ tab vb L c) then
    (prob (fun e => ∃ x y, rowsOutZC A Hs P planZ tab vb R τ wo c e = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y)
      (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)).toReal
  else 0

theorem rowsFinderPrZC_nonneg (c : vb.Pos) (k Rw : ℕ) : 0 ≤ rowsFinderPrZC A Hs P planZ tab vb wo R τ c k Rw := by
  unfold rowsFinderPrZC
  split_ifs
  · exact ENNReal.toReal_nonneg
  · exact le_rfl

/-- **The trial's rows mass at `c`** at its finder's collision probability, where the registrant's commitment at `c` is
the registered one: a value the trial reads at `c` other than the registrant's comes with an opening of `c`'s
registered commitment (`opOfVZC_spec`), so the binding's extractor collides on it and the registrant's. -/
theorem trial_rows_le_prZC (k Rw : ℕ) (c : vb.Pos) [hne : Nonempty (ΩcZ A P planZ tab vb L c)]
    (hon : vb.com (wo c) = vb.cs R c) :
    expect (trialValZC planZ R (psiRowsZC A Hs P planZ tab vb isRead wo R τ c))
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤
      rowsFinderPrZC A Hs P planZ tab vb wo R τ c k Rw := by
  classical
  rw [show trialValZC planZ R (psiRowsZC A Hs P planZ tab vb isRead wo R τ c) =
      ind (fun e => isRead c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c)) from
    funext fun e => trialVal_psiRowsZC A Hs P planZ tab vb isRead wo R τ c e, ← prob_toReal]
  have hsub : prob (fun e => isRead c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c))
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤
      prob (fun e => ∃ x y, rowsOutZC A Hs P planZ tab vb R τ wo c e = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y)
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) := by
    refine prob_mono (fun e he => ?_) _ _
    obtain ⟨-, hex⟩ := he
    show ∃ x y, rowsOutZC A Hs P planZ tab vb R τ wo c e = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y
    unfold rowsOutZC
    rw [dite_eq_left_of_eq_true (eq_true hex)]
    obtain ⟨h1, hne'⟩ := Classical.choose_spec hex
    have s1 := opOfVZC_spec A Hs P planZ tab vb R τ h1
    exact ⟨_, _, rfl, vb.collide_spec _ _ (s1.1.trans hon.symm) (by rw [s1.2]; exact hne')⟩
  have hfin : prob (fun e => ∃ x y, rowsOutZC A Hs P planZ tab vb R τ wo c e = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y)
      (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  unfold rowsFinderPrZC
  rw [dif_pos hne]
  exact ENNReal.toReal_mono hfin hsub

/-- **The ideal rows mass at `c`** at the rows finder's collision probability (`ideal_le_trialZC` at
`rowsFinderPrZC`), where the registrant's commitment at `c`, if it is a registered read, is the registered one. -/
theorem ideal_rows_le_prZC (k Rw : ℕ) (hRw : 1 ≤ Rw) (c : vb.Pos) (hon : isRead c → vb.com (wo c) = vb.cs R c) :
    (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (psiRowsZC A Hs P planZ tab vb isRead wo R τ c ω)) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (rowsFinderPrZC A Hs P planZ tab vb wo R τ c k Rw + k / (Real.exp 1 * Rw)) := by
  classical
  refine ideal_le_trialZC A Hs P planZ tab vb R τ k Rw hRw c _
    (psiRows_nonnegZC A Hs P planZ tab vb isRead wo R τ c) (psiRows_le_oneZC A Hs P planZ tab vb isRead wo R τ c)
    (fun ω hω => funext fun bs => ?_) (rowsFinderPrZC_nonneg A Hs P planZ tab vb wo R τ c k Rw) (fun hne => ?_)
  · simp [psiRowsZC, readsAt_eq_emptyZC A Hs P planZ tab vb R τ hω bs]
  · by_cases hrc : isRead c
    · exact trial_rows_le_prZC A Hs P planZ tab vb isRead wo R τ k Rw c (hon hrc)
    · have h0 : trialValZC planZ R (psiRowsZC A Hs P planZ tab vb isRead wo R τ c) = fun _ => 0 :=
        funext fun e => by rw [trialVal_psiRowsZC]; simp [ind, hrc]
      rw [h0, expect_const]
      exact rowsFinderPrZC_nonneg A Hs P planZ tab vb wo R τ c k Rw

/-- **The rows term at the rows finders' collision probabilities**: where the registrant's commitments at the
registered reads are the registered ones, `rowsPrZC` is at most `Σ_c π c·(Pr[rows finder at c] + k/(eR_w))/(1 − ρ)`. -/
theorem rowsPrZC_le_pr {k Rw : ℕ} (hRw : 1 ≤ Rw) {ρ : ℝ} (hρ : ρ < 1)
    (hon : ∀ c, isRead c → vb.com (wo c) = vb.cs R c) :
    rowsPrZC A Hs P planZ tab vb isRead wo k ρ R τ ≤
      ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (rowsFinderPrZC A Hs P planZ tab vb wo R τ c k Rw + k / (Real.exp 1 * Rw))) / (1 - ρ)) := by
  have hρ1 : 0 < 1 - ρ := by linarith
  unfold rowsPrZC
  exact ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ =>
    ideal_rows_le_prZC A Hs P planZ tab vb isRead wo R τ k Rw hRw c (hon c)) hρ1.le)

end RowsFinder

end FlockSoundness.Discharge.ZkLink
