module

public import Proofs.Flock.SampledGeneric.Compose
public import Proofs.Flock.Bridge.RightRegZc

/-!
# Flock is an instance of the generic sampled audit

Flock's batched `--zk` session is a proof system for proof blocks (`proofSystem`), its binding of wires to commit
strings a binding commitment to the trace (`binding`: a position is a commit string), "right on the registration with
canonical openings" at the trace's value layer the proof blocks' correctness (`correct`), its expected-time extractor
the extractor (`extractor`), and its link trials the trials (`trials`), with slack `k/(eR_w)` for the waits that give up
and `ρ` their rate. Every hypothesis of the generic theorem is a lemma of the compiled link analysis:

* `extractor_covers`: `analysisZCRegC`'s cover;
* `trials_opens`, `trials_wellFormed`: `opOfVZC_spec`, `cT_nonnegZC`;
* `trials_transfers`: the link theorem against any value layer (`zk_linkSoundC_at`), at a constant one;
* `trials_plurality`: the plurality value (`Xval_maxZC`), whose value layer is `XplurZC` (`incorrect_eq`);
* `trials_cR`: Flock's A2 at the link finders (`LinkCRZC`) is the generic `Trials.CR`.

The generic bound is no worse than Flock's: its knowledge term is at most `ksAvgZC` (`ksAvg_le`), and its binding term
at most `linkBoundZC` (`bindErr_le`, from `expect_cT_leZC`). So `countC_regC_generic` is `zk_flock_countC_regC`'s
statement, the count curve `SampledKSLiveV5` runs through, proved through the generic theorem, and
`escapeC_regC_generic` the same at a set of proof blocks.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink.Generic

open Game Model Accounting Audit LinkFinder ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) (canon : Fin C.N → vb.Opening → Prop)

/-- **Flock's batched `--zk` session**, as the compiler's proof system. -/
noncomputable abbrev proofSystem : SampledGeneric.ProofSystem n Reg := ⟨batchedSessionZC A Hs planZ⟩

/-- **Flock's binding of wires to commit strings** (`ValueBindingZ`), as the compiler's commitment to the trace: a
position is a commit string, and its value the one its opening carries. -/
abbrev binding : SampledGeneric.Binding Reg Hc where
  Pos := vb.Pos
  Val := vb.Val
  Opening := vb.Opening
  Com := vb.Com
  val := vb.val
  com := vb.com
  cs := vb.cs
  collide := vb.collide
  collide_spec := vb.collide_spec

/-- **A proof block is correct in a trace**: right on the registration with canonical openings (`RightRegZc`) at the
trace's value layer (`XlayerZC`), each wire's bit in the value of its commit string. -/
def correct (R : Reg) (Xv : vb.Pos → vb.Val) (u : Fin n) : Prop :=
  RightRegZc A P planZ tab vb canon R (XlayerZC A P planZ tab vb Xv) u

/-- **Flock's expected-time extractor** at `k` accepted reruns and the knowledge bound `a`: its knowledge term is
`a/(1 − N₀/(ek))` at the proof block's table, and its binding error `linkTermZC` at the trace's value layer. -/
noncomputable def extractor (k : ℕ)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ) :
    SampledGeneric.Extractor (proofSystem A Hs planZ) (binding A P planZ tab vb) where
  ks S R τ u := ENNReal.ofReal (a S R (zcStrat A Hs planZ τ) (tab S R u) / (1 - rateZ (planZ S R) (tab S R u) k))
  link S R τ u Xv := linkTermZC A Hs P planZ tab decode k S R τ u (XlayerZC A P planZ tab vb Xv)

/-- **Flock's link trials** at commit string `c`: a sample that reads `c`, a fresh session and `k` waits of at most `Rw`
reruns each (`trialGZC`), reading the values the extraction opens `c` to (`rdTZC`), at `t'` SHA-512 calls a run
(`cTZC`). `ρ` bounds the waits' rate, and the slack `k/(eR_w)` is what the waits that give up lose. -/
noncomputable def trials (L : Law n) (k Rw : ℕ) (t' ρ : ℝ) :
    SampledGeneric.Trials L (proofSystem A Hs planZ) (binding A P planZ tab vb) where
  Out := TrialOutZC F L D Sl
  Read c := Nonempty (ΩcZ A P planZ tab vb L c)
  T R τ c h := haveI : Nonempty (ΩcZ A P planZ tab vb L c) := h
    trialGZC A Hs P planZ tab vb R τ c k Rw
  sT R τ c h := haveI : Nonempty (ΩcZ A P planZ tab vb L c) := h
    trialStratZC A Hs P planZ tab vb R τ c k Rw
  rd R τ c := rdTZC A Hs P planZ tab vb R τ c
  op R τ c := opOfVZC A Hs P planZ tab vb R τ c
  cost R _ := cTZC planZ R t'
  wt c := (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω
  ρ := ρ
  slack := k / (Real.exp 1 * Rw)

/-! ## The generic theorem's hypotheses -/

omit [Fintype K] in
/-- **The extractor's terms cover acceptance**: `analysisZCRegC`'s cover, at the trace's value layer. -/
theorem extractor_covers {k : ℕ} (hr : ∀ S R j, rateZ (planZ S R) j k < 1)
    (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) (hcan : OpeningsCanon A P planZ tab vb canon)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (extractor A Hs P planZ tab vb k a).Covers (correct A P planZ tab vb canon) := by
  intro S R τ u Xv hS hu
  have h := (analysisZCRegC (L := { Ω := Unit, draw := fun _ => ∅ }) A Hs P planZ tab vb canon
    (fun _ _ => XlayerZC A P planZ tab vb Xv) k hr hlow hown hcan a hj).cover S R τ u
    (R, XlayerZC A P planZ tab vb Xv) hS ((mem_wrongRegZc A P planZ tab vb canon).2 hu)
  simp only [analysisZCRegC, ↓reduceIte] at h
  exact h

theorem trials_opens (L : Law n) (k Rw : ℕ) (t' ρ : ℝ) : (trials A Hs P planZ tab vb L k Rw t' ρ).Opens :=
  fun R τ _ _ _ hv => opOfVZC_spec A Hs P planZ tab vb R τ hv

omit [Fintype K] in
theorem trials_wellFormed (L : Law n) (k Rw : ℕ) {t' ρ : ℝ} (ht : 0 ≤ t') (hρ : ρ < 1) :
    (trials A Hs P planZ tab vb L k Rw t' ρ).WellFormed :=
  ⟨fun R _ e => cT_nonnegZC planZ R ht e, fun _ => div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _), hρ,
    div_nonneg (Nat.cast_nonneg _) (mul_nonneg (Real.exp_pos 1).le (Nat.cast_nonneg _))⟩

omit [Fintype K] in
/-- The trials' binding error is the compiled analysis's (`errZC`). -/
theorem errOf_trials (L : Law n) (k Rw : ℕ) (t' ρ : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)
    (c : vb.Pos) (x : vb.Val) :
    (trials A Hs P planZ tab vb L k Rw t' ρ).errOf R τ c x = errZC A Hs P planZ tab vb R τ c k Rw x := by
  unfold SampledGeneric.Trials.errOf errZC
  split_ifs with h1 h2 h2
  · rfl
  · exact absurd h1 h2
  · exact absurd h2 h1
  · rfl

/-- **The binding error transfers to the trials**: the link theorem against any value layer (`zk_linkSoundC_at`), at the
value layer of a fixed trace. -/
theorem trials_transfers (hlow : LoweringSoundZK A P planZ tab decode) {k Rw : ℕ} (hRw : 1 ≤ Rw) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) (L : Law n) (t' : ℝ) :
    (trials A Hs P planZ tab vb L k Rw t' ρ).Transfers (extractor A Hs P planZ tab vb k a) := by
  intro R τ tgt Xv
  refine le_of_le_of_eq (zk_linkSoundC_at A Hs P planZ tab vb hlow hRw hρ hr hj (fun _ _ => Xv) R τ tgt) ?_
  unfold errBoundZC
  refine congrArg ENNReal.ofReal (congrArg (· / (1 - ρ)) (Finset.sum_congr rfl fun c _ => ?_))
  rw [errOf_trials]
  rfl

theorem trials_plurality (L : Law n) (k Rw : ℕ) (t' ρ : ℝ) :
    (trials A Hs P planZ tab vb L k Rw t' ρ).Plurality fun R τ => XvalZC A Hs P planZ tab vb R τ k Rw := by
  intro R τ c h v
  have : Nonempty (ΩcZ A P planZ tab vb L c) := h
  exact Xval_maxZC A Hs P planZ tab vb R τ k Rw c v

omit [Fintype K] in
/-- **Flock's A2 at the link finders is the generic `Trials.CR`.** -/
theorem trials_cR {L : Law n} {k Rw M : ℕ} {t' ρ : ℝ} {R : Reg} {τ : Cont L Reg (batchedSessionZC A Hs planZ) R}
    (h : LinkCRZC A Hs P planZ tab vb R τ k Rw M t') : (trials A Hs P planZ tab vb L k Rw t' ρ).CR R τ M :=
  fun c hc => h c hc

omit [Fintype K] in
/-- **The committed trace's incorrect set is Flock's**: the units wrong on the registration with canonical openings at
the plurality's value layer (`XplurZC`). -/
theorem incorrect_eq {L : Law n} (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) (k Rw : ℕ) :
    SampledGeneric.incorrect (binding A P planZ tab vb) (correct A P planZ tab vb canon) R
        (XvalZC A Hs P planZ tab vb R τ k Rw) =
      wrongRegZc A P planZ tab vb canon R (XplurZC A Hs P planZ tab vb R τ k Rw) := by
  ext u
  rw [SampledGeneric.mem_incorrect, mem_wrongRegZc]
  rfl

/-! ## No worse than Flock's bound -/

omit [Fintype K] in
/-- **The generic knowledge term is at most Flock's** (`ksAvgZC`): the largest over the sample's proof blocks is at most
the largest over their tables. -/
theorem ksAvg_le (L : Law n) (k : ℕ)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) :
    (extractor A Hs P planZ tab vb k a).ksAvg L R τ ≤ ksAvgZC A Hs planZ k a R τ := by
  unfold SampledGeneric.Extractor.ksAvg ksAvgZC
  exact avg_mono fun ω => iSup₂_le fun u _ => le_iSup (fun j : Fin (planZ (L.draw ω) R).length =>
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k)))
    (tab (L.draw ω) R u)

/-- **The generic binding term is at most Flock's link bound** (`linkBoundZC`): a trial costs at most `t'(1 + k)`
SHA-512 calls in expectation (`expect_cT_leZC`). -/
theorem bindErr_le {L : Law n} {k Rw M : ℕ} {t' ρ : ℝ} {R : Reg} {τ : Cont L Reg (batchedSessionZC A Hs planZ) R}
    (ht : 0 ≤ t') (hρ : ρ < 1) (h : LinkCRZC A Hs P planZ tab vb R τ k Rw M t') :
    (trials A Hs P planZ tab vb L k Rw t' ρ).bindErr R τ M ≤ linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ := by
  unfold linkBoundZC
  rw [ite_eq_left h]
  refine ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right ?_ (by show (0 : ℝ) ≤ 1 - ρ; linarith))
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun c _ =>
    mul_le_mul_of_nonneg_left ?_ (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  have hc : (trials A Hs P planZ tab vb L k Rw t' ρ).ecost R τ c ≤ t' * (1 + k) := by
    unfold SampledGeneric.Trials.ecost
    split_ifs with hne
    · have : Nonempty (ΩcZ A P planZ tab vb L c) := hne
      exact expect_cT_leZC A Hs P planZ tab vb R τ c k Rw ht
    · exact mul_nonneg ht (by positivity)
  have h2 : 2 * (trials A Hs P planZ tab vb L k Rw t' ρ).ecost R τ c / 2 ^ (256 : ℝ) ≤
      2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) :=
    div_le_div_of_nonneg_right (by linarith) (by positivity)
  have hs : (trials A Hs P planZ tab vb L k Rw t' ρ).slack = k / (Real.exp 1 * Rw) := rfl
  linarith

/-! ## Flock's bounds through the generic theorem -/

/-- **The compiled `--zk` count curve, through the generic theorem**: `zk_flock_countC_regC`'s statement, the count
curve `SampledKSLiveV5` runs through, from `Flock.Guarantees.GenericSampledKSCount` at Flock's instance. At a prover
whose link finders satisfy A2 its terms are the generic ones, at most Flock's (`ksAvg_le`, `bindErr_le`); at any other
Flock's link bound is `⊤`. -/
theorem countC_regC_generic (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u)
    (hcan : OpeningsCanon A P planZ tab vb canon) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ) {L : Law n}
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        K ≤ (wrongRegZc A P planZ tab vb canon (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) := by
  by_cases hCR : LinkCRZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw M t'
  · rw [← incorrect_eq A Hs P planZ tab vb canon]
    refine (Flock.SecurityProofs.GenericSampledKSCount n Reg Dh Hc L (proofSystem A Hs planZ)
      (binding A P planZ tab vb) (correct A P planZ tab vb canon) (extractor A Hs P planZ tab vb k a)
      (trials A Hs P planZ tab vb L k Rw t' ρ) (fun R τ => XvalZC A Hs P planZ tab vb R τ k Rw) M
      (extractor_covers A Hs P planZ tab vb canon (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan hj)
      (trials_opens A Hs P planZ tab vb L k Rw t' ρ) (trials_wellFormed A Hs P planZ tab vb L k Rw ht hρ)
      (trials_transfers A Hs P planZ tab vb hlow hRw hρ hr hj L t') (trials_plurality A Hs P planZ tab vb L k Rw t' ρ)
      hM K σ (trials_cR A Hs P planZ tab vb hCR)).trans ?_
    exact add_le_add (add_le_add le_rfl (ksAvg_le A Hs P planZ tab vb L k a _ _))
      (bindErr_le A Hs P planZ tab vb ht hρ hCR)
  · unfold linkBoundZC
    rw [ite_eq_right hCR, add_top]
    exact le_top

/-- **The compiled `--zk` escape bound at a set of proof blocks, through the generic theorem**:
`Pr[accept ∧ W ⊆ wrong] ≤ escape W + ksAvgZC + linkBoundZC`, from `Flock.Guarantees.GenericSampledKSEscape` at Flock's
instance. -/
theorem escapeC_regC_generic (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u)
    (hcan : OpeningsCanon A P planZ tab vb canon) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (W : Finset (Fin n)) {L : Law n}
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        W ⊆ wrongRegZc A P planZ tab vb canon (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw))
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.escape W + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) := by
  by_cases hCR : LinkCRZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw M t'
  · rw [← incorrect_eq A Hs P planZ tab vb canon]
    refine (Flock.SecurityProofs.GenericSampledKSEscape n Reg Dh Hc L (proofSystem A Hs planZ)
      (binding A P planZ tab vb) (correct A P planZ tab vb canon) (extractor A Hs P planZ tab vb k a)
      (trials A Hs P planZ tab vb L k Rw t' ρ) (fun R τ => XvalZC A Hs P planZ tab vb R τ k Rw) M
      (extractor_covers A Hs P planZ tab vb canon (fun S R j => (hr S R j).trans_lt hρ) hlow hown hcan hj)
      (trials_opens A Hs P planZ tab vb L k Rw t' ρ) (trials_wellFormed A Hs P planZ tab vb L k Rw ht hρ)
      (trials_transfers A Hs P planZ tab vb hlow hRw hρ hr hj L t') (trials_plurality A Hs P planZ tab vb L k Rw t' ρ)
      hM W σ (trials_cR A Hs P planZ tab vb hCR)).trans ?_
    exact add_le_add (add_le_add le_rfl (ksAvg_le A Hs P planZ tab vb L k a _ _))
      (bindErr_le A Hs P planZ tab vb ht hρ hCR)
  · unfold linkBoundZC
    rw [ite_eq_right hCR, add_top]
    exact le_top

end FlockSoundness.Discharge.ZkLink.Generic
