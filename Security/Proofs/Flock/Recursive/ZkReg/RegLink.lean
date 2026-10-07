import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ

/-!
# The committed values against registered values, for the compiled `--zk` session

`zk_linkSoundC` binds the drawn units' extracted witnesses to the plurality value layer (`XplurZC`). A registrant's
openings bind it further: given an opening `wo c` of each commit string `c`, a wire `g` whose commit string's opening has
the commitment registered at `R` and whose value-layer bit is not the opening's is a **registered-value disagreement**
(`RegBadZ`). An accepted session with a drawn unit reading such a wire has, after extraction, one of two traces at the
unit's table: the extracted witness disagrees with the value layer at some wire (the link event), or it opens the wire's
commit string to a value other than the registrant's (`LinkEvRegZK`). The second is a collision of `Hc` between the
extracted opening and `wo c`, which a one-trial finder outputs (`regFinderZC`, at the same trial as the link finder,
`trialGZC`).

* `analysisRegZC`: `analysisZC` with the committed transcript paired with the registration, its wrong units the wrong
  units together with the units reading a registered-value disagreement (`regWrongZ`), and its link term at
  `LinkEvRegZK`;
* `RegCRZC`, `regTermZC`: `cr/sha-512` for the registered-value finders of one prover at budget `qW` on every outcome,
  and its term `Q_s/(1 − ρ)·(qW²/2^513 + k/(eR_w))`, `Q_s = Σ_c π_c` as in `linkBoundZC`;
* `zk_flock_regValsC_cr`: `Pr[accept ∧ a drawn unit reads a registered-value disagreement] ≤ ε_ks + δ_link + δ_reg`;
  `zk_flock_regDrawnC_cr`: the same with a wrong drawn unit counted too.

`qW` bounds the SHA-512 evaluations of one trial: a fresh run and at most `k` waits of at most `R_w` reruns each, so
`t'(1 + k R_w)` at `t'` per run is an honest budget. Lean counts no evaluations; whoever instantiates `RegCRZC` states it.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession ZkLink StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
variable {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc) (wo : vb.Pos → vb.Opening)

/-! ## The event and the analysis -/

/-- **Wire `g` disagrees with its registered value** in the transcript `Xt`: the registrant's opening `wo` of `g`'s
commit string has the commitment registered at `R`, and `Xt`'s bit on `g` is not the opening's. -/
def RegBadZ (R : Reg) (Xt : Fin C.N → Bool) (g : Fin C.N) : Prop :=
  vb.com (wo (vb.pos g)) = vb.cs R (vb.pos g) ∧ Xt g ≠ vb.bit (vb.val (wo (vb.pos g))) g

open Classical in
/-- The units that read a wire disagreeing with its registered value. -/
noncomputable def regWrongZ (R : Reg) (Xt : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ∃ g ∈ P.io u, RegBadZ A P planZ tab vb wo R Xt g

/-- **The registered link event** at drawn unit `u`'s table, for the registration `R'` and the transcript `Xc`: the
oracle `C₀` is committed, and every close satisfying padded witness, on one of `u`'s wires, decodes differently from
`Xc`, or decodes differently from the registrant's opening of the wire's commit string, that opening having the
commitment registered at `R'`. -/
def LinkEvRegZK (S : Finset (Fin n)) (R : Reg) (u : Fin n) (R' : Reg) (Xc : Fin C.N → Bool) (C₀ : Oracle F) : Prop :=
  (tabOfZ planZ tab S R u).Committed A C₀ ∧
    ∀ M ∈ padWits A (tabOfZ planZ tab S R u).sch (tabOfZ planZ tab S R u).t (tabOfZ planZ tab S R u).c₀ C₀,
      (tabOfZ planZ tab S R u).S.Satisfies (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
        ∃ g ∈ P.io u, decode S R u M g ≠ Xc g ∨
          (vb.com (wo (vb.pos g)) = vb.cs R' (vb.pos g) ∧ decode S R u M g ≠ vb.bit (vb.val (wo (vb.pos g))) g)

/-- **The registered link term**: `linkTermZC` at `LinkEvRegZK`. -/
noncomputable def linkTermRegZC (k : ℕ) (S : Finset (Fin n)) (R : Reg)
    (τ : Strategy (batchedSessionZC A Hs planZ S R)) (u : Fin n) (t : Reg × (Fin C.N → Bool)) : ℝ≥0∞ :=
  ENNReal.ofReal (accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ) *
    extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ)
      (LinkEvRegZK A P planZ tab vb wo S R u t.1 t.2) /
    (1 - rateZ (planZ S R) (tab S R u) k))

/-- **The registered-values term**: `Q_s/(1 − ρ)·(qW²/2^513 + k/(eR_w))`, with `Q_s = Σ_c π_c` the expected number of
commit strings the drawn units read. -/
noncomputable def regTermZC (L : Law n) (k Rw : ℕ) (ρ qW : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
    (qW ^ 2 / 2 ^ (513 : ℝ) + k / (Real.exp 1 * Rw)) / (1 - ρ))

variable {L : Law n}

/-- **The compiled `--zk` audit's analysis against registered values**: `analysisZC`, its committed transcript paired
with the registration, its wrong units also those that read a registered-value disagreement (`regWrongZ`), and its
link term at `LinkEvRegZK`. -/
noncomputable def analysisRegZC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) (k : ℕ)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (hj : JointZC A Hs planZ k a) :
    ExtractionAnalysis L Reg (batchedSessionZC A Hs planZ) where
  Tr := Reg × (Fin C.N → Bool)
  wrong t := P.wrong t.2 ∪ regWrongZ A P planZ tab vb wo t.1 t.2
  committed R τ := (R, Xc R τ)
  ks S R τ u := ENNReal.ofReal (a S R (zcStrat A Hs planZ τ) (tab S R u) / (1 - rateZ (planZ S R) (tab S R u) k))
  link := linkTermRegZC A Hs P planZ tab vb wo k
  cover S R τ u t hS hu := by
    classical
    have hG : ∀ T, (tabOfZ planZ tab S R u).Committed A T → LinkEvRegZK A P planZ tab vb wo S R u t.1 t.2 T :=
      fun T hc => ⟨hc, fun M _ hsat => by
        rcases Finset.mem_union.1 hu with hw | hw
        · by_contra hno
          refine P.mem_wrong.1 hw (P.correct_congr (fun g hg => ?_) (hlow S R u hS M hsat))
          by_contra hne
          exact hno ⟨g, hg, Or.inl (Ne.symm hne)⟩
        · obtain ⟨g, hg, hcom, hne⟩ := (Finset.mem_filter.1 hw).2
          by_cases hd : decode S R u M g = t.2 g
          · exact ⟨g, hg, Or.inr ⟨hcom, hd ▸ hne⟩⟩
          · exact ⟨g, hg, Or.inl hd⟩⟩
    set ε := accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ)
    set r := rateZ (planZ S R) (tab S R u) k
    set a' := a S R (zcStrat A Hs planZ τ) (tab S R u)
    set Lp := extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ)
      (LinkEvRegZK A P planZ tab vb wo S R u t.1 t.2)
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

/-! ## The registered-value finder -/

variable (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

open Classical in
/-- **The registered-value trace at commit string `c`**: the registrant's opening of `c` has the registered commitment,
and the extraction from the reruns `bs` reads a value at `c` other than the opening's. -/
noncomputable def psiRegZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) : ℝ :=
  if vb.com (wo c) = vb.cs R c ∧ ∃ v ∈ readsAtZC A Hs P planZ tab vb R τ ω bs c, v ≠ vb.val (wo c) then 1 else 0

open Classical in
/-- **The registered-value finder's output**: when the registrant's opening of `c` has the registered commitment and the
trial reads a value at `c` other than its, the collision of that reading's opening with the registrant's. -/
noncomputable def outRegZC (c : vb.Pos) (e : TrialOutZC F L D Sl) : Pair :=
  if h : vb.com (wo c) = vb.cs R c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c) then
    some (vb.collide (opOfVZC A Hs P planZ tab vb R τ c e (Classical.choose h.2)) (wo c))
  else none

/-- **The registered-value finder for commit string `c`**: one trial (`trialGZC`), its output read by `outRegZC`. -/
noncomputable def regFinderZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) : Finder Pair :=
  oneRun (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
    (outRegZC A Hs P planZ tab vb wo R τ c)

/-- **`cr/sha-512` for one prover's registered-value finders** (`Finder.CR`, `Assumptions.SHA512CRStrict`), at the
registration `R`, the continuation `τ` and the budget `qW` on every outcome, for the finder of each commit string a draw
can read. -/
def RegCRZC (k Rw : ℕ) (qW : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)),
    (regFinderZC A Hs P planZ tab vb wo R τ c k Rw).CR Hc qW

open Classical in
/-- **The registered-values bound, prover by prover**: `regTermZC` at a prover whose registered-value finders satisfy
`cr/sha-512` (`RegCRZC`), `⊤` at any other. -/
noncomputable def regBoundZC (k Rw : ℕ) (ρ qW : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) :
    ℝ≥0∞ :=
  if RegCRZC A Hs P planZ tab vb wo R τ k Rw qW then regTermZC A P planZ tab vb L k Rw ρ qW else ⊤

/-! ## The registered-value mass at one commit string -/

omit [Fintype K] in
theorem psiReg_nonnegZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    0 ≤ psiRegZC A Hs P planZ tab vb wo R τ c ω bs := by
  unfold psiRegZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem psiReg_le_oneZC (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    psiRegZC A Hs P planZ tab vb wo R τ c ω bs ≤ 1 := by
  unfold psiRegZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem trialVal_psiRegZC (c : vb.Pos) (e : TrialOutZC F L D Sl) :
    trialValZC planZ R (psiRegZC A Hs P planZ tab vb wo R τ c) e =
      ind (fun e => vb.com (wo c) = vb.cs R c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c)) e := by
  classical
  unfold trialValZC rdTZC
  by_cases ha : accωZC planZ R e.1 e.2.1
  · cases hf : firsts (accωZC planZ R e.1) e.2.2 with
    | none => simp [ind, ha, hf]
    | some bs => simp [ind, ha, hf, psiRegZC]
  · simp [ind, ha]

/-- **Every reading of `c` other than the registrant's is a collision** the finder outputs. -/
theorem ind_le_collRegZC (c : vb.Pos) (e : TrialOutZC F L D Sl) :
    ind (fun e => vb.com (wo c) = vb.cs R c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c)) e ≤
      ind (IsColl Hc) (outRegZC A Hs P planZ tab vb wo R τ c e) := by
  classical
  by_cases h : vb.com (wo c) = vb.cs R c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ vb.val (wo c)
  · have hcoll : IsColl Hc (outRegZC A Hs P planZ tab vb wo R τ c e) := by
      unfold outRegZC
      rw [dite_eq_left h]
      obtain ⟨hv, hne⟩ := Classical.choose_spec h.2
      have s := opOfVZC_spec A Hs P planZ tab vb R τ hv
      exact ⟨_, _, rfl, vb.collide_spec _ _ (s.1.trans h.1.symm) (by rw [s.2]; exact hne)⟩
    simp [ind, h, hcoll]
  · have h0 : ind (fun e => vb.com (wo c) = vb.cs R c ∧ ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e,
        v ≠ vb.val (wo c)) e = 0 := by
      unfold ind; rw [ite_eq_right h]
    rw [h0]
    exact ind_nonneg _ _

/-- **The trial's registered-value mass at `c`**, under `cr/sha-512` for the finder: at most `qW²/2^513`. -/
theorem regTrial_mass_leZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) {qW : ℝ}
    (hW : (regFinderZC A Hs P planZ tab vb wo R τ c k Rw).CR Hc qW) :
    expect (trialValZC planZ R (psiRegZC A Hs P planZ tab vb wo R τ c))
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤
      qW ^ 2 / 2 ^ (513 : ℝ) := by
  refine le_trans ?_ hW.le
  rw [regFinderZC, oneRun_ev]
  refine expect_mono (fun e => ?_) _ _
  rw [trialVal_psiRegZC]
  exact ind_le_collRegZC A Hs P planZ tab vb wo R τ c e

/-- **The ideal mass of a trace at `c`** (`ideal_mass_leZC` for any trace `Ψ` of the reruns in `[0, 1]` that vanishes
off the draws reading `c`): jointly with acceptance and averaged over the draw, at most `π c` times the trial's mass
bound `B₀` plus `k/(eRw)` for the waits that give up. -/
theorem ideal_mass_genZC (k Rw : ℕ) (hRw : 1 ≤ Rw) (c : vb.Pos) (Ψ : L.Ω → List (List (OutZC F D Sl)) → ℝ)
    (h0 : ∀ ω bs, 0 ≤ Ψ ω bs) (h1 : ∀ ω bs, Ψ ω bs ≤ 1)
    (hoff : ∀ ω, ¬ ReadsDrawZ A P planZ tab vb c ω → ∀ bs, Ψ ω bs = 0) {B₀ : ℝ} (hB₀ : 0 ≤ B₀)
    (htr : ∀ _ : Nonempty (ΩcZ A P planZ tab vb L c),
      expect (trialValZC planZ R Ψ) (trialGZC A Hs P planZ tab vb R τ c k Rw)
        (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤ B₀) :
    (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k (Ψ ω)) /
        Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * (B₀ + k / (Real.exp 1 * Rw)) := by
  classical
  set f : L.Ω → ℝ := fun ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k (Ψ ω) with hf
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hB : 0 ≤ B₀ + k / (Real.exp 1 * Rw) := add_nonneg hB₀ (by positivity)
  have hoff' : ∀ ω, ¬ ReadsDrawZ A P planZ tab vb c ω → f ω = 0 := by
    intro ω hω
    have hpsi : Ψ ω = fun _ => 0 := funext (hoff ω hω)
    simp only [hf, hpsi, expectAcc_zero, mul_zero]
  have hsum : ∑ ω : L.Ω, f ω = ∑ w : ΩcZ A P planZ tab vb L c, f w.1 := by
    rw [← Finset.sum_filter_of_ne (p := fun ω => ReadsDrawZ A P planZ tab vb c ω)
      (fun ω _ hne => by by_contra h; exact hne (hoff' ω h))]
    exact Finset.sum_subtype _ (fun ω => by simp) f
  rw [hsum, div_le_iff₀ hΩ]
  by_cases hne : Nonempty (ΩcZ A P planZ tab vb L c)
  · set E : ΩcZ A P planZ tab vb L c → ℝ := fun w =>
      expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 *
        geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
          (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
        expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
          (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod * Ψ w.1 bs) with hE
    have hw : ∀ w : ΩcZ A P planZ tab vb L c, f w.1 ≤ E w + k / (Real.exp 1 * Rw) := fun w =>
      expectAcc_le_waits _ _ _ k Rw hRw _ (h0 w.1) (h1 w.1)
    have htr' := expect_trialValZC A Hs P planZ tab vb R τ c k Rw Ψ
    have hmass := htr hne
    have hcard : (0 : ℝ) < Fintype.card (ΩcZ A P planZ tab vb L c) := by exact_mod_cast Fintype.card_pos
    rw [htr', div_le_iff₀ hcard] at hmass
    calc ∑ w : ΩcZ A P planZ tab vb L c, f w.1
        ≤ ∑ w : ΩcZ A P planZ tab vb L c, (E w + k / (Real.exp 1 * Rw)) :=
          Finset.sum_le_sum fun w _ => hw w
      _ = (∑ w : ΩcZ A P planZ tab vb L c, E w) +
          Fintype.card (ΩcZ A P planZ tab vb L c) * (k / (Real.exp 1 * Rw)) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      _ ≤ B₀ * Fintype.card (ΩcZ A P planZ tab vb L c) +
          Fintype.card (ΩcZ A P planZ tab vb L c) * (k / (Real.exp 1 * Rw)) := by linarith
      _ = _ := by field_simp
  · have h0 : (∑ w : ΩcZ A P planZ tab vb L c, f w.1) = 0 :=
      Finset.sum_eq_zero fun w _ => (hne ⟨w⟩).elim
    rw [h0]
    have : 0 ≤ (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * (B₀ + k / (Real.exp 1 * Rw)) :=
      mul_nonneg (div_nonneg (Nat.cast_nonneg _) hΩ.le) hB
    nlinarith

/-! ## The registered link theorem -/

/-- **The registered link event leaves a trace**: if drawn unit `u`'s extracted table is committed but every close
satisfying message, on one of `u`'s wires, decodes differently from the value layer or from the registrant's opening
(that opening having the registered commitment), then some commit string reads a value other than its plurality or
other than the registrant's. -/
theorem linkEvReg_le_psiZC (k Rw : ℕ) {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω)
    (bs : List (List (OutZC F D Sl))) :
    ind (fun bs => LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R (XplurZC A Hs P planZ tab vb R τ k Rw)
      (tabExtZC A Hs planZ tab R τ ω u bs)) bs ≤
      (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) + ∑ c, psiRegZC A Hs P planZ tab vb wo R τ c ω bs := by
  classical
  have hsum0 : 0 ≤ ∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs :=
    Finset.sum_nonneg fun c _ => psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω bs
  have hsum0' : 0 ≤ ∑ c, psiRegZC A Hs P planZ tab vb wo R τ c ω bs :=
    Finset.sum_nonneg fun c _ => psiReg_nonnegZC A Hs P planZ tab vb wo R τ c ω bs
  by_cases hL : LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R (XplurZC A Hs P planZ tab vb R τ k Rw)
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
    have h1 : ind (fun bs => LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R
        (XplurZC A Hs P planZ tab vb R τ k Rw) (tabExtZC A Hs planZ tab R τ ω u bs)) bs = 1 := by
      unfold ind; rw [ite_eq_left ⟨hc, hall⟩]
    rw [h1]
    rcases hd with hdec | ⟨hcom, hne⟩
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
    · have hne' : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ≠ vb.val (wo (vb.pos g)) := by
        intro heq
        apply hne
        rw [← hbit, heq]
      have hpsi : psiRegZC A Hs P planZ tab vb wo R τ (vb.pos g) ω bs = 1 := by
        unfold psiRegZC; rw [ite_eq_left ⟨hcom, _, hmem, hne'⟩]
      have := Finset.single_le_sum (fun c _ => psiReg_nonnegZC A Hs P planZ tab vb wo R τ c ω bs)
        (Finset.mem_univ (vb.pos g))
      linarith
  · have h0 : ind (fun bs => LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R
        (XplurZC A Hs P planZ tab vb R τ k Rw) (tabExtZC A Hs planZ tab R τ ω u bs)) bs = 0 := by
      unfold ind; rw [ite_eq_right hL]
    rw [h0]
    exact add_nonneg hsum0 hsum0'

omit [Fintype K] in
/-- **The registered analysis's knowledge term is `analysisZC`'s** (`zc_knowledgeSound`). -/
theorem reg_knowledgeSoundZC
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisRegZC A Hs P planZ tab vb wo Xc k hr hlow a hj).KnowledgeSound (ksAvgZC A Hs planZ k a) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (planZ (L.draw ω) R).length =>
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k)))
    (tab (L.draw ω) R u)

/-- **The registered link theorem**: at the plurality value layer, the registered analysis's link term is at most
`δ_link + δ_reg`, `linkBoundZC` under A2 for the link finders and `regBoundZC` under `cr/sha-512` for the
registered-value finders, prover by prover. -/
theorem zk_linkSoundRegC
    (hlow : LoweringSoundZK A P planZ tab decode) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (qW : ℝ) :
    (analysisRegZC (L := L) A Hs P planZ tab vb wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).LinkSound
      (fun R τ => linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ +
        regBoundZC A Hs P planZ tab vb wo k Rw ρ qW R τ) := by
  intro R τ tgt
  show _ ≤ linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ + regBoundZC A Hs P planZ tab vb wo k Rw ρ qW R τ
  unfold linkBoundZC regBoundZC
  split_ifs with hCR hW
  rotate_left
  · exact le_top.trans_eq (add_top _).symm
  · exact le_top.trans_eq (top_add _).symm
  · exact le_top.trans_eq (top_add _).symm
  unfold regTermZC
  set B := 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)
    with hBdef
  set B' := qW ^ 2 / 2 ^ (513 : ℝ) + k / (Real.exp 1 * Rw) with hB'def
  have hρ1 : 0 < 1 - ρ := by linarith
  set f : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (psiZC A Hs P planZ tab vb R τ k Rw c ω) with hf
  set f' : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (psiRegZC A Hs P planZ tab vb wo R τ c ω) with hf'
  set Λ : L.Ω → ℝ := fun ω => ((∑ c, f c ω) + ∑ c, f' c ω) / (1 - ρ) with hΛ
  have hf0 : ∀ c ω, 0 ≤ f c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ (psi_nonnegZC A Hs P planZ tab vb R τ k Rw c ω))
  have hf0' : ∀ c ω, 0 ≤ f' c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ (psiReg_nonnegZC A Hs P planZ tab vb wo R τ c ω))
  have hΛ0 : ∀ ω, 0 ≤ Λ ω := fun ω =>
    div_nonneg (add_nonneg (Finset.sum_nonneg fun c _ => hf0 c ω) (Finset.sum_nonneg fun c _ => hf0' c ω)) hρ1.le
  have step1 : ∀ ω, atTgt (L.draw ω) (tgt ω) (fun u =>
      (analysisRegZC (L := L) A Hs P planZ tab vb wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).link (L.draw ω) R (τ ω) u
        ((analysisRegZC (L := L) A Hs P planZ tab vb wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
          (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).committed R τ)) ≤ ENNReal.ofReal (Λ ω) := by
    intro ω
    refine atTgt_le (fun u hu => ?_) (tgt ω)
    show ENNReal.ofReal (accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
        extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
          (zcStrat A Hs planZ (τ ω))
          (LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R (XplurZC A Hs P planZ tab vb R τ k Rw)) /
        (1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k)) ≤ ENNReal.ofReal (Λ ω)
    apply ENNReal.ofReal_le_ofReal
    have hε0 : 0 ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) :=
      expect_nonneg (ind_nonneg _) _ _
    have hLp0 := extractProbAccZ_nonneg A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
      (zcStrat A Hs planZ (τ ω))
      (LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R (XplurZC A Hs P planZ tab vb R τ k Rw))
    have hLp : extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
        (zcStrat A Hs planZ (τ ω))
        (LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R (XplurZC A Hs P planZ tab vb R τ k Rw)) ≤
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) +
              ∑ c, psiRegZC A Hs P planZ tab vb wo R τ c ω bs) :=
      expectAcc_mono _ _ _ _ fun bs => linkEvReg_le_psiZC A Hs P planZ tab vb wo R τ k Rw hu bs
    have hrr : 1 - ρ ≤ 1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k := by
      linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
    have hsum : expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (fun bs => (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) +
            ∑ c, psiRegZC A Hs P planZ tab vb wo R τ c ω bs) = (∑ c, f c ω) + ∑ c, f' c ω := by
      rw [expectAcc_add, expectAcc_sum, expectAcc_sum, mul_add, Finset.mul_sum, Finset.mul_sum]
    calc accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
          extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
            (zcStrat A Hs planZ (τ ω))
            (LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R (XplurZC A Hs P planZ tab vb R τ k Rw)) /
          (1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k)
        ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
          extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
            (zcStrat A Hs planZ (τ ω))
            (LinkEvRegZK A P planZ tab vb wo (L.draw ω) R u R (XplurZC A Hs P planZ tab vb R τ k Rw)) /
          (1 - ρ) := div_le_div_of_nonneg_left (mul_nonneg hε0 hLp0) hρ1 hrr
      _ ≤ expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => (∑ c, psiZC A Hs P planZ tab vb R τ k Rw c ω bs) +
              ∑ c, psiRegZC A Hs P planZ tab vb wo R τ c ω bs) / (1 - ρ) :=
          div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hLp hε0) hρ1.le
      _ = Λ ω := by rw [hsum]
  refine (avg_mono step1).trans ?_
  rw [avg_ofReal Λ hΛ0]
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hc : ∀ c, (∑ ω, f c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B := fun c =>
    ideal_mass_leZC A Hs P planZ tab vb R τ k Rw M hRw hM ht c (hCR c)
  have hc' : ∀ c, (∑ ω, f' c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B' := fun c =>
    ideal_mass_genZC A Hs P planZ tab vb R τ k Rw hRw c (psiRegZC A Hs P planZ tab vb wo R τ c)
      (psiReg_nonnegZC A Hs P planZ tab vb wo R τ c) (psiReg_le_oneZC A Hs P planZ tab vb wo R τ c)
      (fun ω hω bs => by simp [psiRegZC, readsAt_eq_emptyZC A Hs P planZ tab vb R τ hω bs])
      (by positivity) (fun hne => regTrial_mass_leZC A Hs P planZ tab vb wo R τ c k Rw (hW c hne))
  have hQ : 0 ≤ ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω :=
    Finset.sum_nonneg fun c _ => div_nonneg (Nat.cast_nonneg _) hΩ.le
  have hB0 : 0 ≤ B := by positivity
  have hB0' : 0 ≤ B' := by positivity
  rw [← ENNReal.ofReal_add (div_nonneg (mul_nonneg hQ hB0) hρ1.le) (div_nonneg (mul_nonneg hQ hB0') hρ1.le)]
  apply ENNReal.ofReal_le_ofReal
  calc (∑ ω, Λ ω) / Fintype.card L.Ω
      = (∑ c, (∑ ω, f c ω) / Fintype.card L.Ω) / (1 - ρ) +
          (∑ c, (∑ ω, f' c ω) / Fintype.card L.Ω) / (1 - ρ) := by
        simp only [hΛ, div_eq_mul_inv, ← Finset.sum_mul, Finset.sum_add_distrib]
        rw [Finset.sum_comm, Finset.sum_comm (f := fun ω c => f' c ω)]
        ring
    _ ≤ (∑ c, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B) / (1 - ρ) +
          (∑ c, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B') / (1 - ρ) :=
        add_le_add (div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc c) hρ1.le)
          (div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc' c) hρ1.le)
    _ = _ := by rw [← Finset.sum_mul, ← Finset.sum_mul]

/-! ## The bound -/

/-- **The compiled `--zk` audit's drawn units agree with the registered values, every term on the accepted
assumptions.** With each drawn unit read from its table at its place (`place`), the committed values the plurality value
layer of `vb`, and `wo` a registrant's opening of each commit string: at a prover whose registered-value finders satisfy
`cr/sha-512` at budget `qW` (`hW`),
`Pr[accept ∧ a drawn unit reads a wire disagreeing with its registered value] ≤ ε_ks + δ_link + δ_reg`, with
`ε_ks = ksAvgStrictZ` under A1 for every table's finders after each draw (`hCR`), `δ_link = linkBoundZC` under A2 for
the link finders, and `δ_reg = regTermZC`. -/
theorem zk_flock_regValsC_cr (hA : A.Correct)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (wo : vb.Pos → vb.Opening)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {qW : ℝ} (hW : RegCRZC A Hs P planZ tab vb wo (reg σ) (cont σ) k Rw qW) :
    prob (fun o => o.1 = true ∧ ∃ u ∈ o.2, ∃ g ∈ P.io u,
        RegBadZ A P planZ tab vb wo (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw) g)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
        regTermZC A P planZ tab vb L k Rw ρ qW := by
  classical
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  have h := extraction_audit_drawn
    (analysisRegZC A Hs P planZ tab vb wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
    (reg_knowledgeSoundZC A Hs P planZ tab vb wo _ _ hlow hj)
    (zk_linkSoundRegC A Hs P planZ tab vb wo hlow hRw hM hρ hr hj ht qW) σ
  have hb : regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ) =
      regTermZC A P planZ tab vb L k Rw ρ qW := ite_eq_left hW
  refine le_trans (prob_mono ?_ _ _) (h.trans ?_)
  · rintro o ⟨h1, u, hu, g, hg, hbad⟩
    exact ⟨h1, Finset.not_disjoint_iff.2 ⟨u, Finset.mem_union_right _
      (Finset.mem_filter.2 ⟨Finset.mem_univ _, g, hg, hbad⟩), hu⟩⟩
  · show ksAvgZC A Hs planZ k _ (reg σ) (cont σ) +
        (linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
          regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ)) ≤ _
    rw [hb, ← add_assoc]
    exact add_le_add (add_le_add (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR) le_rfl) le_rfl

/-- **`zk_flock_regValsC_cr` with the wrong units counted too**: on its assumptions,
`Pr[accept ∧ a drawn unit is wrong or reads a wire disagreeing with its registered value] ≤ ε_ks + δ_link + δ_reg`, the
event `analysisRegZC`'s at its committed transcript. -/
theorem zk_flock_regDrawnC_cr (hA : A.Correct)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc) (wo : vb.Pos → vb.Opening)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planZ (L.draw ω) (reg σ)) j (zcStrat A Hs planZ (cont σ ω)) qF qS)
    {qW : ℝ} (hW : RegCRZC A Hs P planZ tab vb wo (reg σ) (cont σ) k Rw qW) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw) ∪
          regWrongZ A P planZ tab vb wo (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      ksAvgStrictZ planZ L k qF qS (reg σ) + linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
        regTermZC A P planZ tab vb L k Rw ρ qW := by
  classical
  have hlow := loweringSoundZK_of_place A P planZ tab place
  have hj := jointZC_ksBound A hA Hs planZ hk
  have h := extraction_audit_drawn
    (analysisRegZC A Hs P planZ tab vb wo (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow _ hj)
    (reg_knowledgeSoundZC A Hs P planZ tab vb wo _ _ hlow hj)
    (zk_linkSoundRegC A Hs P planZ tab vb wo hlow hRw hM hρ hr hj ht qW) σ
  have hb : regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ) =
      regTermZC A P planZ tab vb L k Rw ρ qW := ite_eq_left hW
  refine h.trans ?_
  show ksAvgZC A Hs planZ k _ (reg σ) (cont σ) +
      (linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) +
        regBoundZC A Hs P planZ tab vb wo k Rw ρ qW (reg σ) (cont σ)) ≤ _
  rw [hb, ← add_assoc]
  exact add_le_add (add_le_add (ksAvgZC_le_strict A Hs planZ k (reg σ) (cont σ) hCR) le_rfl) le_rfl

end FlockSoundness.Discharge.ZkReg
