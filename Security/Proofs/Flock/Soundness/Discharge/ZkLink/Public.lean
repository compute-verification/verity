import Proofs.Flock.Soundness.Discharge.ZkLink.Count

/-!
# The compiled `--zk` count curve, "wrong" measured against the registration

`zk_flock_countC` counts the units wrong in the committed transcript `X` (`Partition.wrong`): a unit is wrong there when
its committed outputs are not its gates applied to its committed inputs, both read from `X`. Nothing in that event ties
`X` to what was registered. This file counts the units **wrong on the registration** instead (`wrongRegZ`): a unit is
right when it is correct in `X` and one opening per commit string makes the circuit's evaluation on `X`'s inputs
(`Circuit.eval`), on every committed wire the unit reads or computes, the bit of an opening of that wire's registered
commit string (`ValueBindingZ.cs`). On a unit's input wires the evaluation is `X` itself; on its outputs it is the
program's output on `X`'s inputs. So a unit is right when the program's output on the committed inputs opens the
registered output commitments, and the committed inputs open the registered input commitments. The second half says
something only of a binding commitment: under a hiding one (`hm96`, at the public file's rows) an opening to
essentially every value exists, so there it adds nothing to correctness in `X`.

* `Partition.ReadsOwn`, `Partition.correct_eval`: when a unit's gates read only its own gates and input gates, a unit
  correct in `X` carries, on each of its committed wires, the circuit's evaluation on `X`'s inputs;
* `wrong_subset_wrongRegZ`: a unit wrong in `X` is wrong on every registration; `wrongRegZ_congr`: `wrongRegZ` reads the
  registration only through its commit strings;
* `analysisZCReg`: `analysisZC` with "wrong" `wrongRegZ` at the registration. Its knowledge and link terms are
  `analysisZC`'s, and its `cover` is the same argument: a satisfying message near the table's oracle that decodes to `X`
  on the unit's wires makes the unit correct in `X` (the lowering) and gives an opening of each of its commit strings
  (`ValueBindingZ.opening_spec`), so the unit is right;
* `zk_flock_countPr_reg`: the count curve for it at the link finders' collision probabilities (`linkPrZC`), and
  `zk_flock_countC_reg` the same under A2, with `zk_flock_countC`'s bound.
-/

namespace FlockSoundness.Audit.Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **Unit `u` reads only its own gates and input gates.** -/
def ReadsOwn (u : Fin n) : Prop :=
  ∀ h x, P.unit h = some u → x ∈ (C.op h).args → P.unit x = some u ∨ P.unit x = none

variable {P}

/-- A unit that reads only its own gates and input gates computes, on its gates, the circuit's evaluation on the inputs
`Xv` holds. -/
theorem localEval_eq_eval {u : Fin n} (hr : P.ReadsOwn u) (Xv : Fin C.N → Bool) :
    ∀ m (g : Fin C.N), g.val = m → P.unit g = some u → P.localEval Xv u g = C.eval Xv g := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm hu
  have hni : (C.op g).isInput = false := by
    cases h : (C.op g).isInput
    · rfl
    · exact absurd ((P.input_iff g).2 h) (by simp [hu])
  rw [P.localEval_of_mem Xv hu, C.eval_op Xv hni]
  refine Op.apply_congr _ fun x hx => ?_
  rcases hr g x hu hx with hxu | hxn
  · exact ih x.val (hgm ▸ C.topo g x hx) x rfl hxu
  · rw [P.localEval_of_not_mem Xv (by rw [hxn]; simp), C.eval_input Xv ((P.input_iff x).1 hxn)]

/-- **The composition, for a unit that reads only its own gates and input gates**: correct in `Xv`, it carries on each
of its committed wires the circuit's evaluation on the inputs `Xv` holds. -/
theorem correct_eval {u : Fin n} (hr : P.ReadsOwn u) {Xv : Fin C.N → Bool} (hc : P.Correct Xv u) :
    ∀ g ∈ P.io u, Xv g = C.eval Xv g := by
  classical
  intro g hg
  simp only [io, Finset.mem_filter] at hg
  obtain ⟨hgc, hgu | ⟨h, hh, hgh⟩⟩ := hg
  · rw [hc g hgc hgu, P.localEval_eq_eval hr Xv g.val g rfl hgu]
  · rcases hr h g hh hgh with hgu | hgn
    · rw [hc g hgc hgu, P.localEval_eq_eval hr Xv g.val g rfl hgu]
    · exact (C.eval_input Xv ((P.input_iff g).1 hgn)).symm

end FlockSoundness.Audit.Partition

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

/-- **Unit `u` is right on the registration `R`** at the transcript `Xv`: it is correct in `Xv`, and one opening per
commit string makes the circuit's evaluation on `Xv`'s inputs, on every committed wire `u` reads or computes, the bit of
an opening of that wire's registered commit string. -/
def RightRegZ (R : Reg) (Xv : Fin C.N → Bool) (u : Fin n) : Prop :=
  P.Correct Xv u ∧ ∃ op : vb.Pos → vb.Opening, ∀ g ∈ P.io u,
    vb.com (op (vb.pos g)) = vb.cs R (vb.pos g) ∧ vb.bit (vb.val (op (vb.pos g))) g = C.eval Xv g

open Classical in
/-- **The units wrong on the registration `R`** at the transcript `Xv`: those not right on it (`RightRegZ`). -/
noncomputable def wrongRegZ (R : Reg) (Xv : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ RightRegZ A P planZ tab vb R Xv u

omit [Fintype K] [DecidableEq K] in
theorem mem_wrongRegZ {R : Reg} {Xv : Fin C.N → Bool} {u : Fin n} :
    u ∈ wrongRegZ A P planZ tab vb R Xv ↔ ¬ RightRegZ A P planZ tab vb R Xv u := by
  classical
  simp [wrongRegZ]

omit [Fintype K] [DecidableEq K] in
/-- A unit wrong in `Xv` is wrong on every registration. -/
theorem wrong_subset_wrongRegZ (R : Reg) (Xv : Fin C.N → Bool) : P.wrong Xv ⊆ wrongRegZ A P planZ tab vb R Xv :=
  fun _ hu => (mem_wrongRegZ A P planZ tab vb).2 fun h => P.mem_wrong.1 hu h.1

omit [Fintype K] [DecidableEq K] in
/-- `wrongRegZ` reads the registration only through its commit strings. -/
theorem wrongRegZ_congr {R R' : Reg} (h : ∀ p, vb.cs R p = vb.cs R' p) (Xv : Fin C.N → Bool) :
    wrongRegZ A P planZ tab vb R Xv = wrongRegZ A P planZ tab vb R' Xv := by
  ext u
  simp only [mem_wrongRegZ, RightRegZ, h]

variable {L : Law n}

open Classical in
/-- **The compiled `--zk` audit's analysis, "wrong" measured against the registration** (`analysisZC`'s, with
`wrongRegZ`). The transcript is the registration with the committed values, so that "wrong" can read the registered
commit strings; the link term is `analysisZC`'s at the session's own registration, and `⊤` at any other. -/
noncomputable def analysisZCReg
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) (k : ℕ)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (hj : JointZC A Hs planZ k a) :
    ExtractionAnalysis L Reg (batchedSessionZC A Hs planZ) where
  Tr := Reg × (Fin C.N → Bool)
  wrong Xt := wrongRegZ A P planZ tab vb Xt.1 Xt.2
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
        refine (mem_wrongRegZ A P planZ tab vb).1 hu ⟨hcX, fun c => vb.opening S R' u M c, fun g hg => ?_⟩
        obtain ⟨h1, h2⟩ := vb.opening_spec S R' u hS M hsat g hg
        exact ⟨h1, h2.trans ((hno g hg).trans (P.correct_eval (hown u) hcX g hg))⟩⟩
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
/-- `analysisZCReg`'s knowledge term is `analysisZC`'s, `ksAvgZC`. -/
theorem zc_knowledgeSound_reg
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (hown : ∀ u, P.ReadsOwn u)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZCReg A Hs P planZ tab vb Xc k hr hlow hown a hj).KnowledgeSound (ksAvgZC A Hs planZ k a) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (planZ (L.draw ω) R).length =>
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k)))
    (tab (L.draw ω) R u)

/-- `analysisZCReg`'s link term is `analysisZC`'s at the committed registration, so the link theorem at the link
finders' collision probabilities (`zk_linkSoundPr`) bounds it. -/
theorem zk_linkSoundPr_reg
    (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZCReg (L := L) A Hs P planZ tab vb (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).LinkSound
      (linkPrZC A Hs P planZ tab vb k Rw M ρ) := by
  intro R τ tgt
  refine le_of_eq_of_le (congrArg avg (funext fun ω => ?_))
    (zk_linkSoundPr A Hs P planZ tab vb hlow hRw hM hρ hr hj R τ tgt)
  simp only [analysisZCReg, analysisZC, ↓reduceIte]

/-- `zk_linkSoundPr_reg` under A2 (`linkPrZC_le_linkBoundZC`). -/
theorem zk_linkSoundC_reg
    (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') :
    (analysisZCReg (L := L) A Hs P planZ tab vb (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).LinkSound
      (linkBoundZC A Hs P planZ tab vb k Rw M ρ t') := fun R τ tgt =>
  (zk_linkSoundPr_reg A Hs P planZ tab vb hlow hown hRw hM hρ hr hj R τ tgt).trans
    (linkPrZC_le_linkBoundZC A Hs P planZ tab vb hρ ht R τ)

/-- **The compiled `--zk` count curve, "wrong" measured against the registration, at the link finders' collision
probabilities**: `Pr[accept ∧ at least K units in wrongRegZ] ≤ miss(K) + ksAvgZC + linkPrZC`, for a partition whose
units read only their own gates and input gates. No hash premise. -/
theorem zk_flock_countPr_reg (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        K ≤ (wrongRegZ A P planZ tab vb (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) + linkPrZC A Hs P planZ tab vb k Rw M ρ (reg σ) (cont σ) :=
  extraction_audit_count (analysisZCReg A Hs P planZ tab vb (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
      (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj)
    (zc_knowledgeSound_reg A Hs P planZ tab vb _ _ hlow hown hj)
    (zk_linkSoundPr_reg A Hs P planZ tab vb hlow hown hRw hM hρ hr hj) K σ

/-- `analysisZCReg`'s link term at the value layer of any values `Xv` is `analysisZC`'s at the committed registration,
so `zk_linkSoundC_at` bounds it. -/
theorem zk_linkSoundC_reg_at
    (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k : ℕ} {Rw : ℕ} (hRw : 1 ≤ Rw)
    {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a)
    (Xv : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → vb.Pos → vb.Val) :
    (analysisZCReg (L := L) A Hs P planZ tab vb (fun R τ => XlayerZC A P planZ tab vb (Xv R τ)) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow hown a hj).LinkSound
      (errBoundZC A Hs P planZ tab vb Xv k Rw ρ) := by
  intro R τ tgt
  refine le_of_eq_of_le (congrArg avg (funext fun ω => ?_))
    (zk_linkSoundC_at A Hs P planZ tab vb hlow hRw hρ hr hj Xv R τ tgt)
  simp only [analysisZCReg, analysisZC, ↓reduceIte]

/-- **The compiled `--zk` count curve, "wrong" measured against the registration**: `zk_flock_countC`'s bound, with
the units wrong on the registration (`wrongRegZ`) at the plurality value layer, for a partition whose units read only
their own gates and input gates (`zk_flock_countPr_reg` under A2). -/
theorem zk_flock_countC_reg (hlow : LoweringSoundZK A P planZ tab decode) (hown : ∀ u, P.ReadsOwn u) {k Rw M : ℕ}
    (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC A Hs planZ))) :
    prob (fun o => o.1 = true ∧
        K ≤ (wrongRegZ A P planZ tab vb (reg σ) (XplurZC A Hs P planZ tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC A Hs planZ)) σ ≤
      L.miss K + ksAvgZC A Hs planZ k a (reg σ) (cont σ) +
        linkBoundZC A Hs P planZ tab vb k Rw M ρ t' (reg σ) (cont σ) :=
  (zk_flock_countPr_reg A Hs P planZ tab vb hlow hown hRw hM hρ hr hj K σ).trans
    (add_le_add le_rfl (linkPrZC_le_linkBoundZC A Hs P planZ tab vb hρ ht _ _))

end FlockSoundness.Discharge.ZkLink
