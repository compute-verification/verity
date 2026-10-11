import Proofs.Flock.Recursive.ZkReg.Vals
import Proofs.Flock.Recursive.Fork

/-!
# Recursion: a stage's committed values against a candidate, for the compiled `--zk` session

A stage's transfer (`recursive_sound_upto`'s `htr`) bounds, for every candidate value vector `v`, the outer session
accepting while the value layer it binds disagrees with `v` on the stage's commit strings, by a slack plus, at each
commit string, a weight times the probability that a trial reads that string off `v`. This is `zk_flock_regValsC_cr`
with the registrant's opening replaced by a candidate value `wv c` at each commit string `c` of the stage's set `Sk`,
and with what was a collision of `Hc` (the registrant's opening against an extracted one) left as the trial's own
probability of reading a value other than `wv c` at `c` (`candTrialZC`): the candidate is the stage's plurality
reference, which no finder at a budget computes, so `stage_bind` bounds that probability by rewinding instead.

* `CandBadZ`, `candWrongZ`, `LinkEvCandZK`, `analysisCandZC`: `RegBadZ`, `regWrongZ`, `LinkEvRegZK`, `analysisRegZC` at
  the candidate;
* `psiCandZC`, `candTrialZC`, `candTermZC`: the trace at `c`, the trial's mass of it, and the term
  `(Σ_c π_c·(candTrialZC c + k/(eR_w)))/(1 − ρ)`;
* `zk_flock_candValsC`: `Pr[accept ∧ a drawn unit reads a wire of Sk whose layer bit is not wv's] ≤ ε_ks + δ_link +
  candTermZC`;
* `openingsOf`, `trialFinderZC`, `candWeightZC`, `candSlackZC`, `candTermZC_le`: the term in `stage_bind`'s form, the
  value binding's openings, the trial for `c` as a finder of the openings of `c` it reads, and
  `candTermZC ≤ slack + Σ_c κ_c·Pr[the trial for c opens c's digest to a value other than wv c]`;
* `zk_flock_candTransferC`: `zk_flock_candValsC` in that form, `htr`'s shape at one history;
* `zk_session_candValsR`: `zk_flock_candValsC` for `flock-verify verify --zk --registered F` sessions of `J`
  hidden-output tables, at the committed values their simulation decodes (`XplurZC` at `simZHJ … τ`, the value binding
  `vbHJ`), as `zk_session_regValsR` is `zk_flock_regValsC_cr` there; `candTermZC_le` puts its bound in `htr`'s shape.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession ZkLink StrictCR ZkReg
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
variable {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc) (Sk : Finset vb.Pos) (wv : vb.Pos → vb.Val)

/-! ## The event and the analysis -/

/-- **Wire `g` disagrees with the candidate** in the transcript `Xt`: its commit string is one of `Sk`, and `Xt`'s bit on
`g` is not the candidate value's. -/
def CandBadZ (Xt : Fin C.N → Bool) (g : Fin C.N) : Prop :=
  vb.pos g ∈ Sk ∧ Xt g ≠ vb.bit (wv (vb.pos g)) g

open Classical in
/-- The units that read a wire disagreeing with the candidate. -/
noncomputable def candWrongZ (Xt : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ∃ g ∈ P.io u, CandBadZ A P planZ tab vb Sk wv Xt g

/-- **The candidate link event** at drawn unit `u`'s table, for the transcript `Xc`: the oracle `C₀` is committed, and
every close satisfying padded witness, on one of `u`'s wires, decodes differently from `Xc`, or, the wire's commit string
being one of `Sk`, from the candidate. -/
def LinkEvCandZK (S : Finset (Fin n)) (R : Reg) (u : Fin n) (Xc : Fin C.N → Bool) (C₀ : Oracle F) : Prop :=
  (tabOfZ planZ tab S R u).Committed A C₀ ∧
    ∀ M ∈ padWits A (tabOfZ planZ tab S R u).sch (tabOfZ planZ tab S R u).t (tabOfZ planZ tab S R u).c₀ C₀,
      (tabOfZ planZ tab S R u).S.Satisfies (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
        ∃ g ∈ P.io u, decode S R u M g ≠ Xc g ∨ (vb.pos g ∈ Sk ∧ decode S R u M g ≠ vb.bit (wv (vb.pos g)) g)

/-- **The candidate link term**: `linkTermZC` at `LinkEvCandZK`. -/
noncomputable def linkTermCandZC (k : ℕ) (S : Finset (Fin n)) (R : Reg)
    (τ : Strategy (batchedSessionZC A Hs planZ S R)) (u : Fin n) (Xc : Fin C.N → Bool) : ℝ≥0∞ :=
  ENNReal.ofReal (accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ) *
    extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ)
      (LinkEvCandZK A P planZ tab vb Sk wv S R u Xc) /
    (1 - rateZ (planZ S R) (tab S R u) k))

variable {L : Law n}

/-- **The compiled `--zk` audit's analysis against a candidate**: `analysisZC`, its wrong units also those that read a
wire disagreeing with the candidate (`candWrongZ`), and its link term at `LinkEvCandZK`. -/
noncomputable def analysisCandZC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) (k : ℕ)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (hj : JointZC A Hs planZ k a) :
    ExtractionAnalysis L Reg (batchedSessionZC A Hs planZ) where
  Tr := Fin C.N → Bool
  wrong t := P.wrong t ∪ candWrongZ A P planZ tab vb Sk wv t
  committed R τ := Xc R τ
  ks S R τ u := ENNReal.ofReal (a S R (zcStrat A Hs planZ τ) (tab S R u) / (1 - rateZ (planZ S R) (tab S R u) k))
  link := linkTermCandZC A Hs P planZ tab vb Sk wv k
  cover S R τ u t hS hu := by
    classical
    have hG : ∀ T, (tabOfZ planZ tab S R u).Committed A T → LinkEvCandZK A P planZ tab vb Sk wv S R u t T :=
      fun T hc => ⟨hc, fun M _ hsat => by
        rcases Finset.mem_union.1 hu with hw | hw
        · by_contra hno
          refine P.mem_wrong.1 hw (P.correct_congr (fun g hg => ?_) (hlow S R u hS M hsat))
          by_contra hne
          exact hno ⟨g, hg, Or.inl (Ne.symm hne)⟩
        · obtain ⟨g, hg, hpos, hne⟩ := (Finset.mem_filter.1 hw).2
          by_cases hd : decode S R u M g = t g
          · exact ⟨g, hg, Or.inr ⟨hpos, hd ▸ hne⟩⟩
          · exact ⟨g, hg, Or.inl hd⟩⟩
    set ε := accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ)
    set r := rateZ (planZ S R) (tab S R u) k
    set a' := a S R (zcStrat A Hs planZ τ) (tab S R u)
    set Lp := extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ)
      (LinkEvCandZK A P planZ tab vb Sk wv S R u t)
    have hr1 : 0 < 1 - r := by linarith [hr S R (tab S R u)]
    have ha : 0 ≤ a' := (hj S R _ (tab S R u)).1
    have hL : 0 ≤ Lp := extractProbAccZ_nonneg _ _ _ _ _ _ _
    have hε0 : 0 ≤ ε := accProbZC_nonneg _ _ _ _
    have key : ε ≤ a' / (1 - r) + ε * Lp / (1 - r) := by
      rcases hε0.eq_or_lt with h0 | hpos
      · rw [← h0]
        positivity
      · have h1 := one_le_fail_add_accZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ) _ hG hpos
        have h2 := (hj S R (zcStrat A Hs planZ τ) (tab S R u)).2
        rw [← add_div, le_div_iff₀ hr1]
        nlinarith
    show prob (· = true) (batchedSessionZC A Hs planZ S R) τ ≤
      ENNReal.ofReal (a' / (1 - r)) + ENNReal.ofReal (ε * Lp / (1 - r))
    rw [prob_batchedSessionZC_eq, ← ENNReal.ofReal_add (div_nonneg ha hr1.le)
      (div_nonneg (mul_nonneg hε0 hL) hr1.le)]
    exact ENNReal.ofReal_le_ofReal key

/-! ## The trace, the trial's mass, and the term -/

variable (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

open Classical in
/-- **The candidate trace at commit string `c`**: `c` is one of `Sk`, and the extraction from the reruns `bs` reads a
value at `c` other than the candidate's. -/
noncomputable def psiCandZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) : ℝ :=
  if c ∈ Sk ∧ ∃ v ∈ readsAtZC A Hs P planZ tab vb R τ ω bs c, v ≠ wv c then 1 else 0

open Classical in
/-- **The trial's mass of the candidate trace at `c`**: the probability that the trial for `c` (`trialGZC`) reads a
value at `c` other than the candidate's, `c` being one of `Sk`; `0` when no draw reads `c`. -/
noncomputable def candTrialZC (k Rw : ℕ) (c : vb.Pos) : ℝ :=
  if h : Nonempty (ΩcZ A P planZ tab vb L c) then
    haveI := h
    expect (trialValZC planZ R (psiCandZC A Hs P planZ tab vb Sk wv R τ c)) (trialGZC A Hs P planZ tab vb R τ c k Rw)
      (trialStratZC A Hs P planZ tab vb R τ c k Rw)
  else 0

/-- **The candidate term**: `(Σ_c π_c·(candTrialZC c + k/(eR_w)))/(1 − ρ)`, `π_c` the fraction of draws reading `c`. -/
noncomputable def candTermZC (k Rw : ℕ) (ρ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
    (candTrialZC A Hs P planZ tab vb Sk wv R τ k Rw c + k / (Real.exp 1 * Rw))) / (1 - ρ))

omit [Fintype K] in
theorem psiCand_nonnegZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    0 ≤ psiCandZC A Hs P planZ tab vb Sk wv R τ c ω bs := by
  unfold psiCandZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem psiCand_le_oneZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    psiCandZC A Hs P planZ tab vb Sk wv R τ c ω bs ≤ 1 := by
  unfold psiCandZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem candTrial_nonnegZC (k Rw : ℕ) (c : vb.Pos) : 0 ≤ candTrialZC A Hs P planZ tab vb Sk wv R τ k Rw c := by
  unfold candTrialZC
  split_ifs with h
  · refine expect_nonneg (fun e => ?_) _ _
    unfold trialValZC
    split_ifs
    · cases firsts (accωZC planZ R e.1) e.2.2 with
      | none => exact le_rfl
      | some bs => exact psiCand_nonnegZC A Hs P planZ tab vb Sk wv R τ c e.1 bs
    · exact le_rfl
  · exact le_rfl

/-! ## The candidate link theorem -/

/-- **The candidate link event leaves a trace**: if drawn unit `u`'s extracted table is committed but every close
satisfying message, on one of `u`'s wires, decodes differently from the value layer or (on `Sk`) from the candidate, then
some commit string reads a value other than its plurality or other than the candidate's. -/
theorem linkEvCand_le_psiZC (k Rw : ℕ) {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω)
    (bs : List (List (OutZC F D Sl))) :
    ind (fun bs => LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)
      (tabExtZC A Hs planZ tab R τ ω u bs)) bs ≤
      (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) + ∑ c, psiCandZC A Hs P planZ tab vb Sk wv R τ c ω bs := by
  classical
  have hsum0 : 0 ≤ ∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs :=
    Finset.sum_nonneg fun c _ => psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω bs
  have hsum0' : 0 ≤ ∑ c, psiCandZC A Hs P planZ tab vb Sk wv R τ c ω bs :=
    Finset.sum_nonneg fun c _ => psiCand_nonnegZC A Hs P planZ tab vb Sk wv R τ c ω bs
  by_cases hL : LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)
      (tabExtZC A Hs planZ tab R τ ω u bs)
  · obtain ⟨hc, hall⟩ := hL
    have hspec := Classical.choose_spec hc
    obtain ⟨g, hg, hd⟩ := hall (Classical.choose hc) hspec.1 hspec.2
    have hcand : candZC A Hs planZ tab R τ ω u bs = some (Classical.choose hc) := by
      unfold candZC; rw [dite_eq_left_of_eq_true (eq_true hc)]
    have hop : OpensAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g)
        (vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)))
        (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) :=
      ⟨u, hu, ⟨g, hg, rfl⟩, _, hcand, rfl, rfl⟩
    have hmem : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ∈
        readsAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g) := by
      unfold readsAtZC; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hop⟩
    have hbit := (vb.opening_spec (L.draw ω) R u hu (Classical.choose hc) hspec.2 g hg).2
    have h1 : ind (fun bs => LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u
        (XplurZC A Hs P planZ tab vb R τ k Rw) (tabExtZC A Hs planZ tab R τ ω u bs)) bs = 1 := by
      unfold ind; rw [ite_eq_left ⟨hc, hall⟩]
    rw [h1]
    rcases hd with hdec | ⟨hpos, hne⟩
    · have hne : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ≠
          XvalZC A Hs P planZ tab vb R τ k Rw (vb.pos g) := by
        intro heq
        apply hdec
        rw [← hbit, heq]
        rfl
      have hpsi : psiZC A Hs P planZ tab vb R τ k Rw (vb.pos g) ω bs = 1 := by
        unfold psiZC; rw [ite_eq_left ⟨_, hmem, hne⟩]
      have := Finset.single_le_sum (fun c _ => psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω bs)
        (Finset.mem_univ (vb.pos g))
      linarith
    · have hne' : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ≠ wv (vb.pos g) := by
        intro heq
        apply hne
        rw [← hbit, heq]
      have hpsi : psiCandZC A Hs P planZ tab vb Sk wv R τ (vb.pos g) ω bs = 1 := by
        unfold psiCandZC; rw [ite_eq_left ⟨hpos, _, hmem, hne'⟩]
      have := Finset.single_le_sum (fun c _ => psiCand_nonnegZC A Hs P planZ tab vb Sk wv R τ c ω bs)
        (Finset.mem_univ (vb.pos g))
      linarith
  · have h0 : ind (fun bs => LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u
        (XplurZC A Hs P planZ tab vb R τ k Rw) (tabExtZC A Hs planZ tab R τ ω u bs)) bs = 0 := by
      unfold ind; rw [ite_eq_right hL]
    rw [h0]
    exact add_nonneg hsum0 hsum0'

omit [Fintype K] in
/-- **The candidate analysis's knowledge term is `analysisZC`'s** (`zc_knowledgeSound`). -/
theorem cand_knowledgeSoundZC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisCandZC A Hs P planZ tab vb Sk wv Xc k hr hlow a hj).KnowledgeSound (ksAvgZC A Hs planZ k a) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (planZ (L.draw ω) R).length =>
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k)))
    (tab (L.draw ω) R u)

/-- **The candidate link theorem**: at the plurality value layer, the candidate analysis's link term is at most
`δ_link + candTermZC`, `linkBoundZC` under A2 for the link finders, prover by prover. -/
theorem zk_linkSoundCandC
    (hlow : LoweringSoundZK A P planZ tab decode) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') :
    (analysisCandZC (L := L) A Hs P planZ tab vb Sk wv (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).LinkSound
      (fun R τ => linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ +
        candTermZC A Hs P planZ tab vb Sk wv R τ k Rw ρ) := by
  intro R τ tgt
  show _ ≤ linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ + candTermZC A Hs P planZ tab vb Sk wv R τ k Rw ρ
  unfold linkBoundZC
  split_ifs with hCR
  swap
  · exact le_top.trans_eq (top_add _).symm
  unfold candTermZC
  set B := 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)
    with hBdef
  set B' : vb.Pos → ℝ := fun c => candTrialZC A Hs P planZ tab vb Sk wv R τ k Rw c + k / (Real.exp 1 * Rw)
    with hB'def
  have hρ1 : 0 < 1 - ρ := by linarith
  set f : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (psiZC A Hs P planZ tab vb R τ k Rw c ω) with hf
  set f' : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (psiCandZC A Hs P planZ tab vb Sk wv R τ c ω) with hf'
  set Λ : L.Ω → ℝ := fun ω => ((∑ c, f c ω) + ∑ c, f' c ω) / (1 - ρ) with hΛ
  have hf0 : ∀ c ω, 0 ≤ f c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ (psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω))
  have hf0' : ∀ c ω, 0 ≤ f' c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ (psiCand_nonnegZC A Hs P planZ tab vb Sk wv R τ c ω))
  have hΛ0 : ∀ ω, 0 ≤ Λ ω := fun ω =>
    div_nonneg (add_nonneg (Finset.sum_nonneg fun c _ => hf0 c ω) (Finset.sum_nonneg fun c _ => hf0' c ω)) hρ1.le
  have step1 : ∀ ω, atTgt (L.draw ω) (tgt ω) (fun u =>
      (analysisCandZC (L := L) A Hs P planZ tab vb Sk wv (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).link (L.draw ω) R (τ ω) u
        ((analysisCandZC (L := L) A Hs P planZ tab vb Sk wv (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
          (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).committed R τ)) ≤ ENNReal.ofReal (Λ ω) := by
    intro ω
    refine atTgt_le (fun u hu => ?_) (tgt ω)
    show ENNReal.ofReal (accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
        extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
          (zcStrat A Hs planZ (τ ω))
          (LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)) /
        (1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k)) ≤ ENNReal.ofReal (Λ ω)
    apply ENNReal.ofReal_le_ofReal
    have hε0 : 0 ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) :=
      expect_nonneg (ind_nonneg _) _ _
    have hLp0 := extractProbAccZ_nonneg A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
      (zcStrat A Hs planZ (τ ω))
      (LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw))
    have hLp : extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
        (zcStrat A Hs planZ (τ ω))
        (LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)) ≤
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) +
              ∑ c, psiCandZC A Hs P planZ tab vb Sk wv R τ c ω bs) :=
      expectAcc_mono _ _ _ _ fun bs => linkEvCand_le_psiZC A Hs P planZ tab vb Sk wv R τ k Rw hu bs
    have hrr : 1 - ρ ≤ 1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k := by
      linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
    have hsum : expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (fun bs => (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) +
            ∑ c, psiCandZC A Hs P planZ tab vb Sk wv R τ c ω bs) = (∑ c, f c ω) + ∑ c, f' c ω := by
      rw [expectAcc_add, expectAcc_sum, expectAcc_sum, mul_add, Finset.mul_sum, Finset.mul_sum]
    calc accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
          extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
            (zcStrat A Hs planZ (τ ω))
            (LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)) /
          (1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k)
        ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
          extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
            (zcStrat A Hs planZ (τ ω))
            (LinkEvCandZK A P planZ tab vb Sk wv (L.draw ω) R u (XplurZC A Hs P planZ tab vb R τ k Rw)) /
          (1 - ρ) := div_le_div_of_nonneg_left (mul_nonneg hε0 hLp0) hρ1 hrr
      _ ≤ expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) +
              ∑ c, psiCandZC A Hs P planZ tab vb Sk wv R τ c ω bs) / (1 - ρ) :=
          div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hLp hε0) hρ1.le
      _ = Λ ω := by rw [hsum]
  refine (avg_mono step1).trans ?_
  rw [avg_ofReal Λ hΛ0]
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hc : ∀ c, (∑ ω, f c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B := fun c =>
    ideal_mass_leZC A Hs P planZ tab vb R τ k Rw M hRw hM ht c (hCR c)
  have hc' : ∀ c, (∑ ω, f' c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B' c := fun c =>
    ideal_mass_genZC A Hs P planZ tab vb R τ k Rw hRw c (psiCandZC A Hs P planZ tab vb Sk wv R τ c)
      (psiCand_nonnegZC A Hs P planZ tab vb Sk wv R τ c) (psiCand_le_oneZC A Hs P planZ tab vb Sk wv R τ c)
      (fun ω hω bs => by simp [psiCandZC, readsAt_eq_emptyZC A Hs P planZ tab vb R τ hω bs])
      (candTrial_nonnegZC A Hs P planZ tab vb Sk wv R τ k Rw c)
      (fun hne => le_of_eq (by unfold candTrialZC; rw [dite_eq_left_of_eq_true (eq_true hne)]))
  have hQ : 0 ≤ ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω :=
    Finset.sum_nonneg fun c _ => div_nonneg (Nat.cast_nonneg _) hΩ.le
  have hB0 : 0 ≤ B := by positivity
  have hB0' : ∀ c, 0 ≤ B' c := fun c =>
    add_nonneg (candTrial_nonnegZC A Hs P planZ tab vb Sk wv R τ k Rw c) (by positivity)
  have hQ' : 0 ≤ ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B' c :=
    Finset.sum_nonneg fun c _ => mul_nonneg (div_nonneg (Nat.cast_nonneg _) hΩ.le) (hB0' c)
  rw [← ENNReal.ofReal_add (div_nonneg (mul_nonneg hQ hB0) hρ1.le) (div_nonneg hQ' hρ1.le)]
  apply ENNReal.ofReal_le_ofReal
  calc (∑ ω, Λ ω) / Fintype.card L.Ω
      = (∑ c, (∑ ω, f c ω) / Fintype.card L.Ω) / (1 - ρ) +
          (∑ c, (∑ ω, f' c ω) / Fintype.card L.Ω) / (1 - ρ) := by
        simp only [hΛ, div_eq_mul_inv, ← Finset.sum_mul, Finset.sum_add_distrib]
        rw [Finset.sum_comm, Finset.sum_comm (f := fun ω c => f' c ω)]
        ring
    _ ≤ (∑ c, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B) / (1 - ρ) +
          (∑ c, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B' c) / (1 - ρ) :=
        add_le_add (div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc c) hρ1.le)
          (div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc' c) hρ1.le)
    _ = _ := by rw [← Finset.sum_mul]

/-! ## The bound -/

/-- **The compiled `--zk` audit's drawn units agree with a candidate, but for the trial's readings off it.** With each
drawn unit read from its table at its place (`place`), the committed values the plurality value layer of `vb`, and a
candidate value `wv c` at each commit string `c` of `Sk`: `Pr[accept ∧ a drawn unit reads a wire of Sk whose layer bit
is not the candidate's] ≤ ε_ks + δ_link + candTermZC`, with `ε_ks = ksAvgStrictZ` under A1 for every table's finders
after each draw (`hCR`) and `δ_link = linkBoundZC` under A2 for the link finders. -/
theorem zk_flock_candValsC (hA : A.Correct)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (Sk : Finset vb.Pos)
    (wv : vb.Pos → vb.Val)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS) :
    prob (fun o => o.1 = true ∧ ∃ u ∈ o.2, ∃ g ∈ P.io u,
        CandBadZ A P planZ tab vb Sk wv (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw) g)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
        candTermZC A Hs P planZ tab vb Sk wv (reg σ) (cont σ) k Rw ρ := by
  classical
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  have h := extraction_audit_drawn
    (analysisCandZC A Hs P planZ tab vb Sk wv (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
    (cand_knowledgeSoundZC A Hs P planZ tab vb Sk wv _ _ hlow hj)
    (zk_linkSoundCandC A Hs P planZ tab vb Sk wv hlow hRw hM hρ hr hj ht) σ
  refine le_trans (prob_mono ?_ _ _) (h.trans ?_)
  · rintro o ⟨h1, u, hu, g, hg, hbad⟩
    exact ⟨h1, Finset.not_disjoint_iff.2 ⟨u, Finset.mem_union_right _
      (Finset.mem_filter.2 ⟨Finset.mem_univ _, g, hg, hbad⟩), hu⟩⟩
  · show ksAvgZC A Hs planZ k _ (reg σ) (cont σ) +
        (linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
          candTermZC A Hs P planZ tab vb Sk wv (reg σ) (cont σ) k Rw ρ) ≤ _
    rw [← add_assoc]
    exact add_le_add (add_le_add (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR) le_rfl) le_rfl

/-! ## The trial as a finder of openings -/

omit [Fintype K] in
/-- **A value binding's openings**, as `stage_bind` reads them. -/
abbrev openingsOf : Openings Hc vb.Val where
  Op := vb.Opening
  Dg := vb.Com
  com := vb.com
  val := vb.val
  collide := vb.collide
  collide_spec := vb.collide_spec

open Classical in
/-- **The trial for commit string `c` as a finder of openings**: the openings of `c` it reads (`opOfVZC` at each value
of `rdTZC`); none when no draw reads `c`. -/
noncomputable def trialFinderZC (k Rw : ℕ) (c : vb.Pos) : Finder (Finset vb.Opening) :=
  if h : Nonempty (ΩcZ A P planZ tab vb L c) then
    haveI := h
    oneRun (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
      fun e => (rdTZC A Hs P planZ tab vb R τ c e).image (opOfVZC A Hs P planZ tab vb R τ c e)
  else ⟨.ret ∅, PUnit.unit⟩

/-- **The weight of commit string `c`** in the candidate term: on `Sk`, the fraction of draws reading `c` over `1 − ρ`. -/
noncomputable def candWeightZC (ρ : ℝ) (c : vb.Pos) : ℝ≥0∞ :=
  if c ∈ Sk then ENNReal.ofReal ((Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω / (1 - ρ)) else 0

/-- **The candidate term's slack**: `(Σ_c π_c)·(k/(eR_w))/(1 − ρ)`. -/
noncomputable def candSlackZC (k Rw : ℕ) (ρ : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
    (k / (Real.exp 1 * Rw)) / (1 - ρ))

omit [Fintype K] in
theorem trialVal_psiCandZC (c : vb.Pos) (e : TrialOutZC F L D Sl) :
    trialValZC planZ R (psiCandZC A Hs P planZ tab vb Sk wv R τ c) e =
      ind (fun e => c ∈ Sk ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ wv c) e := by
  classical
  unfold trialValZC rdTZC
  by_cases ha : accωZC planZ R e.1 e.2.1
  · cases hf : firsts (accωZC planZ R e.1) e.2.2 with
    | none => simp [ind, ha, hf]
    | some bs => simp [ind, ha, hf, psiCandZC]
  · simp [ind, ha]

open Classical in
/-- **A reading of `c` off the candidate is an opening of `c`'s digest to another value** among the trial's. -/
theorem ind_rdT_le_offZC (c : vb.Pos) (e : TrialOutZC F L D Sl) :
    ind (fun e => c ∈ Sk ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ wv c) e ≤
      ind (OffAt (openingsOf A P planZ tab vb) (vb.cs R c) (wv c))
        ((rdTZC A Hs P planZ tab vb R τ c e).image (opOfVZC A Hs P planZ tab vb R τ c e)) := by
  unfold ind
  split_ifs with h h'
  · exact le_rfl
  · obtain ⟨-, v, hv, hne⟩ := h
    have s := opOfVZC_spec A Hs P planZ tab vb R τ hv
    exact absurd ⟨_, Finset.mem_image_of_mem _ hv, s.1, fun heq => hne (s.2.symm.trans heq)⟩ h'
  · norm_num
  · exact le_rfl

/-- **The trial's mass of the candidate trace at `c` is at most its finder's probability of an opening off the
candidate.** -/
theorem candTrial_le_offZC (k Rw : ℕ) (c : vb.Pos) :
    candTrialZC A Hs P planZ tab vb Sk wv R τ k Rw c ≤
      (prob (OffAt (openingsOf A P planZ tab vb) (vb.cs R c) (wv c)) (trialFinderZC A Hs P planZ tab vb R τ k Rw c).g
        (trialFinderZC A Hs P planZ tab vb R τ k Rw c).s).toReal := by
  unfold candTrialZC trialFinderZC
  by_cases h : Nonempty (ΩcZ A P planZ tab vb L c)
  · rw [dite_eq_left h, dite_eq_left h, prob_toReal]
    refine le_of_le_of_eq (expect_mono (fun e => ?_) _ _) (oneRun_ev _ _ _ _).symm
    rw [trialVal_psiCandZC]
    exact ind_rdT_le_offZC A Hs P planZ tab vb Sk wv R τ c e
  · rw [dite_eq_right h]
    exact ENNReal.toReal_nonneg

omit [Fintype K] in
theorem candTrial_eq_zeroZC (k Rw : ℕ) {c : vb.Pos} (hc : c ∉ Sk) :
    candTrialZC A Hs P planZ tab vb Sk wv R τ k Rw c = 0 := by
  unfold candTrialZC
  split_ifs with h
  · have h0 : trialValZC planZ R (psiCandZC A Hs P planZ tab vb Sk wv R τ c) = fun _ => 0 :=
      funext fun e => by rw [trialVal_psiCandZC]; simp [ind, hc]
    rw [h0, expect_const]
  · rfl

/-- **The candidate term in `stage_bind`'s form**: the slack plus, at each commit string, its weight times the
probability that the trial for it outputs an opening of its digest to a value other than the candidate's. -/
theorem candTermZC_le (k Rw : ℕ) {ρ : ℝ} (hρ : ρ < 1) :
    candTermZC A Hs P planZ tab vb Sk wv R τ k Rw ρ ≤
      candSlackZC (L := L) A P planZ tab vb k Rw ρ +
        ∑ c, candWeightZC (L := L) A P planZ tab vb Sk ρ c *
          prob (OffAt (openingsOf A P planZ tab vb) (vb.cs R c) (wv c))
            (trialFinderZC A Hs P planZ tab vb R τ k Rw c).g (trialFinderZC A Hs P planZ tab vb R τ k Rw c).s := by
  unfold candTermZC candSlackZC
  have hρ1 : 0 < 1 - ρ := by linarith
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hπ0 : ∀ c, 0 ≤ (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω :=
    fun c => div_nonneg (Nat.cast_nonneg _) hΩ.le
  have hT0 := candTrial_nonnegZC A Hs P planZ tab vb Sk wv R τ k Rw
  have heq : (∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (candTrialZC A Hs P planZ tab vb Sk wv R τ k Rw c + k / (Real.exp 1 * Rw))) / (1 - ρ) =
      (∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
          (k / (Real.exp 1 * Rw)) / (1 - ρ) +
        ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω / (1 - ρ) *
          candTrialZC A Hs P planZ tab vb Sk wv R τ k Rw c := by
    rw [Finset.sum_mul, Finset.sum_div, Finset.sum_div, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun c _ => by ring
  rw [heq]
  refine ENNReal.ofReal_add_le.trans (add_le_add le_rfl ?_)
  rw [ENNReal.ofReal_sum_of_nonneg fun c _ => mul_nonneg (div_nonneg (hπ0 c) hρ1.le) (hT0 c)]
  refine Finset.sum_le_sum fun c _ => ?_
  unfold candWeightZC
  split_ifs with hc
  · rw [ENNReal.ofReal_mul (div_nonneg (hπ0 c) hρ1.le)]
    refine mul_le_mul' le_rfl ?_
    exact (ENNReal.ofReal_le_ofReal (candTrial_le_offZC A Hs P planZ tab vb Sk wv R τ k Rw c)).trans_eq
      (ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)))
  · rw [candTrial_eq_zeroZC A Hs P planZ tab vb Sk wv R τ k Rw hc, mul_zero, ENNReal.ofReal_zero]
    exact zero_le

/-- **A stage's transfer for the compiled `--zk` audit, in `stage_bind`'s form** (`htr`): with the openings of `vb`
(`openingsOf`), the digest of commit string `c` the registration's (`vb.cs`), and at each `c` the trial for it
(`trialFinderZC`), `Pr[accept ∧ a drawn unit reads a wire of Sk whose layer bit is not the candidate's] ≤
ε_ks + δ_link + slack + Σ_c κ_c·Pr[the trial for c opens c's digest to a value other than wv c]`. -/
theorem zk_flock_candTransferC (hA : A.Correct)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (Sk : Finset vb.Pos)
    (wv : vb.Pos → vb.Val)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS) :
    prob (fun o => o.1 = true ∧ ∃ u ∈ o.2, ∃ g ∈ P.io u,
        CandBadZ A P planZ tab vb Sk wv (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw) g)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      (ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
          candSlackZC (L := L) A P planZ tab vb k Rw ρ) +
        ∑ c, candWeightZC (L := L) A P planZ tab vb Sk ρ c *
          prob (OffAt (openingsOf A P planZ tab vb) (vb.cs (reg σ) c) (wv c))
            (trialFinderZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw c).g
            (trialFinderZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw c).s := by
  exact ((zk_flock_candValsC A Hs P planZ tab hA place vb Sk wv hk hRw hM hρ hr ht σ hCR).trans
    (add_le_add le_rfl (candTermZC_le A Hs P planZ tab vb Sk wv (reg σ) (cont σ) k Rw hρ))).trans_eq
    (add_assoc _ _ _).symm

section Session

open Exec Refine Flock FlockLevel3 ZkExec ZkBind Integrate Placed DecodeZL Composed

open ZkHidden (planZHJ tabZHJ DrawSetupZHJ TagsOkZHJ DrawOkZKHJ placeZHJ)
open FlockSoundness.Discharge.Tables (LiveOutJ)

/-- **The committed values of a `--zk --registered` session are a candidate's on a set of commit strings, but for the
trial's readings off it**: `zk_flock_candValsC` at the session's simulation (`simZHJ`), at `J` hidden-output tables and
the value binding `vbHJ`, with `cr/sha-512` for the knowledge finders of the simulation (`hCR`). -/
theorem zk_session_candValsR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    -- the law
    (Lv : Law n)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (Lv.draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the commit strings and the candidate
    (Sk : Finset (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Pos)
    (wv : (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Pos → (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Val)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZHJ dj y₀ (Lv.draw ω) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
          CandBadZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) Sk wv
            (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
              (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
              (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw) g)
        (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) Lv k qF qS (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) k Rw M ρ t'
          (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        candTermZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) Sk wv (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw ρ := by
  have hdec := decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
    (ZkHidden.runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr Lv hTags hRec hDraw)
  have hc' := zk_flock_candValsC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
    (tabZHJ dj y₀) execArith_correct (placeZHJ dj y₀ ht hc hscope hU) (vbHJ dj y₀ ht hc hscope hU hpub) Sk wv
    hk hRw hM hρ hr ht' (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) hCR
  exact le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_drawnJ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun _ S => ∃ u ∈ S,
      ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
        CandBadZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) Sk wv
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw) g) _ hc')

end Session

end FlockSoundness.Discharge.Recursion
