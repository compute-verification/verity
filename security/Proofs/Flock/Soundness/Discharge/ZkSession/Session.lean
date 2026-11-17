module

public import Proofs.Flock.Soundness.Discharge.ZkSession.Table
public import Proofs.Flock.Soundness.SessionBatch
public import Proofs.Flock.Soundness.Audit.Extraction
public import Proofs.Flock.Soundness.Lowering
public import Proofs.Flock.Soundness.ExecSampleOS

/-! A batched `--zk` session and its audit at the interactive-oracle layer, where every oracle is a message, so it does
not cover the compiled layer's commitments, the executable refinement or the committed coin tree. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting Audit
open scoped ENNReal NNReal

/-- **A `--zk` table of a batched session**: the statement and its schedule, the padded level 0 (`t` padding columns,
`c₀`, `e` extra lanes), the link layout for the session's `m_pts`, and the inner proof's layout: the pads (level 0's
`pads`, the algebraic phase's `padsA`, the triple's `tr`), the pads code `RS[F, dom, np + nP]` read at `q` positions
`pos`, and a radius `δ` within its unique-decoding radius. -/
structure TabZK (F : Type) [Field F] [Fintype F] [DecidableEq F] (mPts : ℕ) where
  S : Statement
  sch : Schedule
  hshape : 2 ≤ sch.levels.length ∧ (sch.levels.headD default).lanes = sch.k0 ∧
    sch.m - 7 = sch.k0 + (sch.levels.headD default).logCols ∧ levelsChained sch.levels sch.ext = true
  hmsch : sch.m = S.m
  hm : 13 ≤ S.m
  t : ℕ
  hpad : PadOK sch.level₀ t
  c₀ : F
  e : ℕ
  ptLocal : ℕ
  hlay : S.LinkLayout ptLocal mPts
  np : ℕ
  pads : Pads0 np e
  padsA : PadsA np
  tr : Triple np
  N : ℕ
  dom : Fin N ↪ F
  nP : ℕ
  q : ℕ
  hnP : NeZero (np + nP)
  pos : F → Fin N
  hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N
  δ : ℝ≥0
  hL : np + nP ≤ N
  hδ : δ ≤ Code.relativeUniqueDecodingRadius (ReedSolomon.code dom (np + nP) : Set (Fin N → F))

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

namespace TabZK

noncomputable def after (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) :
    Game (Bool × Oracle F) :=
  tableAfterZK A T.pads T.padsA T.tr T.dom T.nP T.q T.pos T.S T.sch T.hm T.t T.c₀ T.ptLocal mPts C₀ pts

/-- The table's committed values satisfy its statement (padded level 0). -/
def Committed (A : Arith F K) (T : TabZK F mPts) (C : Oracle F) : Prop :=
  CommittedPad A T.S T.sch T.t T.c₀ C

/-- The table's soundness error: `padLinkError + (padRepError + εInner)²`. -/
noncomputable def bound (T : TabZK F mPts) : ℝ≥0∞ :=
  ENNReal.ofReal (padLinkError T.sch (T.S.shape mPts) T.t) +
    (ENNReal.ofReal (padRepError T.sch (T.S.shape mPts) T.t T.e) + εInner T.N F T.q T.δ) ^ 2

theorem after_sound (A : Arith F K) (hA : A.Correct) (T : TabZK F mPts) (C : Oracle F) :
    value (fun out => out.1 = true ∧ ¬ T.Committed A out.2)
      (.coin (Fin 2 → Fin mPts → F) fun pts => T.after A C pts) ≤ T.bound :=
  haveI := T.hnP
  tableAfterZK_sound A hA T.S T.sch T.hshape T.hmsch T.hm T.t T.hpad T.c₀ T.e T.ptLocal mPts T.hlay T.pads T.padsA
    T.tr T.dom T.nP T.q T.pos T.hpos T.δ T.hL T.hδ C

end TabZK

noncomputable def sessionZK (A : Arith F K) (ts : List (TabZK F mPts)) : Game (List (Bool × Oracle F)) :=
  .send (Fin ts.length → Oracle F) fun Cs => .coin (Fin 2 → Fin mPts → F) fun pts =>
    batch (List.ofFn fun i => (ts.get i).after A (Cs i) pts)

def dfltZK : Bool × Oracle F := (false, fun _ _ => 0)

def acceptedZK (ts : List (TabZK F mPts)) (outs : List (Bool × Oracle F)) : Prop :=
  ∀ i : Fin ts.length, (outs.getD i dfltZK).1 = true

theorem sessionZK_sound (A : Arith F K) (hA : A.Correct) (ts : List (TabZK F mPts)) (j : Fin ts.length)
    (σ : Strategy (sessionZK A ts)) :
    prob (fun outs => acceptedZK ts outs ∧ ¬ (ts.get j).Committed A (outs.getD j dfltZK).2) (sessionZK A ts) σ ≤
      (ts.get j).bound := by
  set Q : Bool × Oracle F → Prop := fun out => out.1 = true ∧ ¬ (ts.get j).Committed A out.2 with hQ
  refine (prob_le_value _ _ σ).trans ((value_mono (q := fun outs => Q (outs.getD j dfltZK))
    (fun outs h => ⟨h.1 j, h.2⟩) _).trans ?_)
  simp only [sessionZK, value_send]
  refine iSup_le fun Cs => le_trans ?_ ((ts.get j).after_sound A hA (Cs j))
  simp only [value_coin]
  refine Finset.sum_le_sum fun pts _ => mul_le_mul_right ?_ _
  have hs := ofFn_split (fun i => (ts.get i).after A (Cs i) pts) j
  have hl : ((List.ofFn fun i => (ts.get i).after A (Cs i) pts).take j).length = j := by simp [j.isLt.le]
  have h := value_batch_getD_le Q dfltZK ((ts.get j).after A (Cs j) pts)
    ((List.ofFn fun i => (ts.get i).after A (Cs i) pts).drop (j + 1))
    ((List.ofFn fun i => (ts.get i).after A (Cs i) pts).take j)
  rw [hl, ← hs] at h
  exact h

theorem prob_exists_le {α ι : Type} (p : ι → α → Prop) (g : Game α) (s : Strategy g) (I : Finset ι) :
    prob (fun a => ∃ i ∈ I, p i a) g s ≤ ∑ i ∈ I, prob (p i) g s := by
  classical
  induction I using Finset.induction_on with
  | empty =>
    simp only [Finset.notMem_empty, false_and, exists_false, Finset.sum_empty]
    exact (prob_false g s).le
  | insert i I hi ih =>
    rw [Finset.sum_insert hi]
    refine (prob_mono (q := fun a => p i a ∨ ∃ i ∈ I, p i a) (fun a ⟨k, hk, hp⟩ => ?_) g s).trans
      ((prob_or_le _ _ g s).trans (add_le_add le_rfl ih))
    rcases Finset.mem_insert.1 hk with rfl | hk
    · exact Or.inl hp
    · exact Or.inr ⟨k, hk, hp⟩

theorem sessionZK_sound_any (A : Arith F K) (hA : A.Correct) (ts : List (TabZK F mPts))
    (σ : Strategy (sessionZK A ts)) :
    prob (fun outs => acceptedZK ts outs ∧ ∃ j : Fin ts.length, ¬ (ts.get j).Committed A (outs.getD j dfltZK).2)
      (sessionZK A ts) σ ≤ ∑ j : Fin ts.length, (ts.get j).bound := by
  refine (prob_mono (q := fun outs => ∃ j ∈ Finset.univ, acceptedZK ts outs ∧
    ¬ (ts.get j).Committed A (outs.getD j dfltZK).2) (fun outs ⟨hacc, j, hj⟩ => ⟨j, Finset.mem_univ _, hacc, hj⟩)
    _ _).trans ((prob_exists_le _ _ _ _).trans (Finset.sum_le_sum fun j _ => sessionZK_sound A hA ts j σ))

section Audit

variable (A : Arith F K) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- The table of the plan that carries sampled proof block `u`. -/
abbrev tabOfZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) : TabZK F mPts := (planZ S R).get (tab S R u)

open Classical in
noncomputable def batchedSessionZK (S : Finset (Fin n)) (R : Reg) : Game Bool :=
  (sessionZK A (planZ S R)).map fun outs => decide (acceptedZK (planZ S R) outs)

noncomputable def zkStrat {S : Finset (Fin n)} {R : Reg} (τ : Strategy (batchedSessionZK A planZ S R)) :
    Strategy (sessionZK A (planZ S R)) :=
  Strategy.ofMap _ _ τ

omit [Fintype K] in
theorem prob_batchedSessionZK (S : Finset (Fin n)) (R : Reg) (τ : Strategy (batchedSessionZK A planZ S R)) :
    prob (· = true) (batchedSessionZK A planZ S R) τ =
      prob (acceptedZK (planZ S R)) (sessionZK A (planZ S R)) (zkStrat A planZ τ) := by
  classical
  refine (prob_map _ _ _ _).trans ?_
  congr 1
  funext outs
  exact propext decide_eq_true_iff

/-- A decoder: for each sampled proof block, the values a level-0 message of its table packs for it. -/
abbrev DecoderZK := (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
  Message F (tabOfZ planZ tab S R u).sch.k0 (tabOfZ planZ tab S R u).sch.level₀.logCols → (Fin C.N → Bool)

def LoweringSoundZK (decode : DecoderZK (C := C) planZ tab) : Prop :=
  ∀ S R u, u ∈ S → ∀ M, (tabOfZ planZ tab S R u).S.Satisfies
    (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) → P.Correct (decode S R u M) u

def LinkEvZK (decode : DecoderZK (C := C) planZ tab) (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (Xc : Fin C.N → Bool) (C₀ : Oracle F) : Prop :=
  (tabOfZ planZ tab S R u).Committed A C₀ ∧
    ∀ M ∈ padWits A (tabOfZ planZ tab S R u).sch (tabOfZ planZ tab S R u).t (tabOfZ planZ tab S R u).c₀ C₀,
      (tabOfZ planZ tab S R u).S.Satisfies (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
        ∃ g ∈ P.io u, decode S R u M g ≠ Xc g

/-- The decoder of the sampled proof blocks' places: each unit is read from its block of the witness. -/
noncomputable def decodeOfPlace (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u) :
    DecoderZK (C := C) planZ tab :=
  fun S R u M => if h : u ∈ S then
    (place S R u h).decode (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M)
  else fun _ => false

omit [Fintype K] [DecidableEq K] in
theorem loweringSoundZK_of_place (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u) :
    LoweringSoundZK A P planZ tab (decodeOfPlace A P planZ tab place) := by
  intro S R u hu M hM
  simp only [decodeOfPlace, hu, ↓reduceDIte]
  exact (place S R u hu).correct hM

variable {L : SamplingDesign n}

noncomputable def analysisZK (decode : DecoderZK (C := C) planZ tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZK A planZ) R → (Fin C.N → Bool))
    (hlow : LoweringSoundZK A P planZ tab decode) : ExtractionAnalysis L Reg (batchedSessionZK A planZ) where
  Tr := Fin C.N → Bool
  incorrect := P.incorrect
  committed := Xc
  ks S R τ u := prob (fun outs => acceptedZK (planZ S R) outs ∧
    ¬ (tabOfZ planZ tab S R u).Committed A (outs.getD (tab S R u) dfltZK).2) (sessionZK A (planZ S R)) (zkStrat A planZ τ)
  link S R τ u Xt := prob (fun outs => acceptedZK (planZ S R) outs ∧
    LinkEvZK A P planZ tab decode S R u Xt (outs.getD (tab S R u) dfltZK).2) (sessionZK A (planZ S R))
    (zkStrat A planZ τ)
  cover S R τ u Xt hS hu := by
    rw [prob_batchedSessionZK]
    refine (prob_mono (fun outs hacc => ?_) _ _).trans (prob_or_le _ _ _ _)
    by_cases hc : (tabOfZ planZ tab S R u).Committed A (outs.getD (tab S R u) dfltZK).2
    · refine Or.inr ⟨hacc, hc, fun M _ hsat => ?_⟩
      by_contra hno
      push Not at hno
      exact P.mem_incorrect.1 hu (P.correct_congr (fun g hg => (hno g hg).symm) (hlow S R u hS M hsat))
    · exact Or.inl ⟨hacc, hc⟩

noncomputable def ksAvgZK (L : SamplingDesign n) (R : Reg) : ℝ≥0∞ :=
  avg fun ω => ⨆ j : Fin (planZ (L.draw ω) R).length, ((planZ (L.draw ω) R).get j).bound

theorem zk_knowledgeSound (hA : A.Correct) (decode : DecoderZK (C := C) planZ tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZK A planZ) R → (Fin C.N → Bool))
    (hlow : LoweringSoundZK A P planZ tab decode) :
    (analysisZK A P planZ tab decode Xc hlow).KnowledgeSound fun R _ => ksAvgZK planZ L R := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact (sessionZK_sound A hA (planZ (L.draw ω) R) (tab (L.draw ω) R u) _).trans
    (le_iSup (fun j : Fin (planZ (L.draw ω) R).length => ((planZ (L.draw ω) R).get j).bound) (tab (L.draw ω) R u))

theorem zk_flock_count (hA : A.Correct) (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZK A planZ) R → (Fin C.N → Bool))
    {εbind : (R : Reg) → Cont L Reg (batchedSessionZK A planZ) R → ℝ≥0∞}
    (hlink : (analysisZK A P planZ tab (decodeOfPlace A P planZ tab place) Xc
      (loweringSoundZK_of_place A P planZ tab place)).BindSound εbind)
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionZK A planZ))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.incorrect (Xc (reg σ) (cont σ))).card) (audit L Reg (batchedSessionZK A planZ)) σ ≤
      L.miss K + ksAvgZK planZ L (reg σ) + εbind (reg σ) (cont σ) :=
  extraction_audit_count (analysisZK A P planZ tab _ Xc (loweringSoundZK_of_place A P planZ tab place))
    (zk_knowledgeSound A P planZ tab hA _ Xc _) hlink K σ

theorem zk_flock_count_execOS (hA : A.Correct)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    (Xc : (R : Reg) → Cont (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZK A planZ) R → (Fin C.N → Bool))
    {εbind : (R : Reg) → Cont (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZK A planZ) R → ℝ≥0∞}
    (hlink : (analysisZK A P planZ tab (decodeOfPlace A P planZ tab place) Xc
      (loweringSoundZK_of_place A P planZ tab place)).BindSound εbind)
    (K : ℕ)
    (σ : Strategy (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZK A planZ))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.incorrect (Xc (reg σ) (cont σ))).card)
        (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZK A planZ)) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        ksAvgZK planZ (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (reg σ) +
        εbind (reg σ) (cont σ) :=
  (zk_flock_count A P planZ tab hA place Xc hlink K σ).trans
    (add_le_add (add_le_add (SamplingDesign.execOS_miss_le σs ks hks St hS Kd pop Lb src hA3 K) le_rfl) le_rfl)

end Audit

end FlockSoundness.Discharge.ZkSession
