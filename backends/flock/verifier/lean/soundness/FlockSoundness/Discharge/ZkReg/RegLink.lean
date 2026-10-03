import FlockSoundness.Discharge.ZkLink.StrictZ

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
* `zk_flock_regValsC_cr`: `Pr[accept ∧ a drawn unit reads a registered-value disagreement] ≤ ε_ks + δ_link + δ_reg`.

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
  sorry

end FlockSoundness.Discharge.ZkReg
