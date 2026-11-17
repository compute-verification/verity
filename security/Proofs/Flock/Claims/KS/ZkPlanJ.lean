module

public import Proofs.Flock.Claims.KS.ZkLiveJ
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Tables.Accepts
import Proofs.Flock.Soundness.Discharge.ZkLink.Sched

/-! The old `--zk` plan at the deployed masked statement is `zkTabs`. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (SampleSetupZJ SampleOkZKJ nTab inputsJ drawJ partS tableZJ planZJ outJ repJ)
open FlockSoundness.Discharge.Tables (DecodesTable)

local notation "𝔽" => FlockLevel3.GF128

/-- `Deploy.V`'s checks pin a record's sample to the statement's only up to key order (`Custody.checks_sample`), so a
theorem at the statement's setup names the sample, not a record. -/
def setupAt (I : Exec.Inputs) (dr : Option Lean.Json) : _root_.Flock.V (SessionSpec × Array Stmt) :=
  Stmt.setupTables I.tags I.circuitFile I.publicFile I.coins I.count I.tables dr I.partition I.program I.work
    I.stratified I.closure

theorem setupOf_eq_setupAt (I : Exec.Inputs) (record : String) :
    Exec.setupOf I record = setupAt I (Exec.recordDraw record) := rfl

theorem setupAt_tableJ {I : Exec.Inputs} {d : Lean.Json} {spec : SessionSpec} {sts : Array Stmt}
    (hh : I.tags.hiddenOutputs = false) (hs : setupAt (ZkExec.zkInputs I) (some d) = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupH (Exec.tagsAt (ZkExec.zkTags I) I.count j) I.circuitFile I.publicFile (ZkExec.zkCoins I) I.tables
        (some (drawJ I d j)) I.partition I.program = .ok sts[j] := by
  have hh' : (ZkExec.zkTags I).hiddenOutputs = false := (ZkExec.zkTags_hiddenOutputs I).trans hh
  by_cases hc : I.count ≤ 1
  · rcases Exec.setupTables_ok hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
    · refine ⟨by simp [nTab, hc], fun j hj => ?_⟩
      have : j = 0 := by simpa using hj
      subst this
      have e1 : Exec.tagsAt (ZkExec.zkTags I) I.count 0 = ZkExec.zkTags I := by simp [Exec.tagsAt, hc]
      have e2 : drawJ I d 0 = d := by simp [drawJ, hc]
      rw [e1, e2]
      exact hst
    · exact absurd hc' (by change ¬ 1 < I.count; omega)
  · have hc1 : 1 < I.count := by omega
    obtain ⟨-, -, population, units, hD1, -, -, -, hsz, hall⟩ := Tables.setupTables_parts hh' hc1 hs
    obtain ⟨u, hu, -, hpop, hun⟩ := hD1 _ rfl
    refine ⟨by simp only [nTab, hc, ite_false]; exact hsz, fun j hj => ?_⟩
    have hpart : drawJ I d j = Tables.partSample population units I.count j := by
      simp only [drawJ, hc, ite_false]
      unfold DecodeZLJ.partJson
      rw [hu]
      dsimp only
      rw [hpop, hun]
    have e1 : Exec.tagsAt (ZkExec.zkTags I) I.count j =
        { ZkExec.zkTags I with identity := (ZkExec.zkTags I).tableIdentity I.count j } := by
      simp [Exec.tagsAt, hc]
    rw [e1, hpart]
    exact hall j hj

theorem table_eq_zkTab {n mPts : ℕ} {I' : Exec.Inputs} {S : Finset (Fin n)} {d : Lean.Json}
    (ht : I'.tags.typed = false) (y : SampleSetupZJ I' mPts S d) {st : Stmt} (hst : y.x.st = st)
    (L : Layout st) {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule st.wideLeaf st.m = .ok (sch, tPad))
    (hk : st.c.kLog ≤ 27) (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL mPts) :
    y.table = zkTab L hz hk hlay := by
  obtain ⟨x, sch', tPad', hz', _, _, _, _, _, _⟩ := y
  subst hst
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz'.symm.trans hz))
  have hS : Refine.stmtOf x.st x.wf.1 (Refine.regionsOf x.st) = L.model (Refine.regionsOf x.st) :=
    (Integrate.model_eq_stmtOf ht x.h x.wf.1 _).symm
  unfold SampleSetupZJ.table SampleSetupZJ.tabAt zkTab
  congr 1

theorem setupZJ_of_setupAt {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj S)) = .ok (spec, sts)) (hok : SampleOkZKJ I dj mPts S)
    (j : Fin (nTab I)) (hj : j.val < sts.size) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule sts[j.val].wideLeaf sts[j.val].m = .ok (sch, tPad)) :
    ∃ y : SampleSetupZJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j),
      y.x.st = sts[j.val] ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨-, hstJ⟩ := setupAt_tableJ hh hs
  have hst := hstJ j.val hj
  obtain ⟨-, -, _, hf⟩ := schedule_spec hz
  have ht' : (inputsJ I j.val).tags.typed = false := Composed.typedJ ht j.val
  obtain ⟨_, -, hdr⟩ := Layout.setupH_sampled hst
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec hdr
  obtain ⟨x, hx⟩ := sampleSetup_of_setupH_g (I := inputsJ I j.val) (S := partS I (dj S) S j) (mPts := mPts)
    ht' hst hf hD ((hok j).1 D hD) ((hok j).2 _ hst)
  obtain ⟨y, hyx, hys, hyt⟩ := Composed.sampleSetupZJ_of ht' x (sch := sch) (tPad := tPad) (by rw [hx]; exact hz)
  exact ⟨y, by rw [hyx, hx], hys, hyt⟩

theorem tableZJ_eq_zkTabs {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZJ (inputsJ I 0) mPts S₀ d₀) {S : Finset (Fin n)} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj S)) = .ok (spec, sts)) (hok : SampleOkZKJ I dj mPts S)
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts)
    (j : Fin (nTab I)) (hj : j.val < sts.size) :
    tableZJ dj y₀ S j = zkTabs h0 L hz hk hlay j.val := by
  obtain ⟨y, hyst, -, -⟩ := setupZJ_of_setupAt hh ht dj hs hok j hj (hz j.val hj)
  unfold tableZJ
  rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩), SampleSetupZJ.table_eq _ y, zkTabs_of_lt h0 L hz hk hlay hj]
  exact table_eq_zkTab (Composed.typedJ ht j.val) y hyst (L j.val hj) (hz j.val hj) (hk j.val hj) (hlay j.val hj)

theorem planZJ_eq_tabsJ {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZJ (inputsJ I 0) mPts S₀ d₀) {S : Finset (Fin n)} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj S)) = .ok (spec, sts)) (hok : SampleOkZKJ I dj mPts S)
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].wideLeaf sts[j].m = .ok (Claims.Flock.zkSched sts[j].wideLeaf sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (R : Reg) :
    planZJ (Reg := Reg) dj y₀ S R = tabsJ sts.size (zkTabs h0 L hz hk hlay) := by
  have hsz := (setupAt_tableJ hh hs).1
  refine List.ext_get (by rw [Composed.planZJ_length, tabsJ_length, hsz]) fun i h1 h2 => ?_
  have hi : i < nTab I := by rwa [Composed.planZJ_length] at h1
  simp only [planZJ, tabsJ, List.get_eq_getElem, List.getElem_ofFn]
  exact tableZJ_eq_zkTabs hh ht dj y₀ hs hok h0 L hz hk hlay ⟨i, hi⟩ (by omega)

theorem sampleOkZKJ_empty {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts)) : SampleOkZKJ I dj spec.mPts ∅ := by
  obtain ⟨hsz, hstJ⟩ := setupAt_tableJ hh hs
  have hh' : (ZkExec.zkTags I).hiddenOutputs = false := (ZkExec.zkTags_hiddenOutputs I).trans hh
  have hpts := (Exec.setupTables_points hh' hs).2
  intro j
  refine ⟨fun _ _ u hu => by simp [partS] at hu, fun st hst => ?_⟩
  have hj : j.val < sts.size := by have := j.isLt; omega
  obtain rfl : st = sts[j.val] := Except.ok.inj (hst.symm.trans (hstJ j.val hj))
  exact hpts j.val hj

noncomputable def opnD {mPts : ℕ} (sts : Array Stmt) (ts : List (TabZK 𝔽 mPts)) (proofs : Array ByteArray)
    (j : Fin ts.length) : OpensZ 𝔽 Refine.Dig (List UInt8) :=
  match sts[j.val]? with
  | some st => match Zk.schedule st.wideLeaf st.m with
    | .ok (sch, _) => ZkExec.opnZ (Flock.Setup.ofCircuit st) sch (Composed.nXZJ (ts.get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | .error _ => default
  | none => default

noncomputable def decD {mPts : ℕ} (ts : List (TabZK 𝔽 mPts)) (j : Fin ts.length) : ℕ → ByteArray → List Msg :=
  ZkExec.decZ (ZkExec.wRestZ (ts.get j).S.m (ts.get j).S.kLog (2 + Composed.nXZJ (ts.get j)) (ts.get j).sch
    (ts.get j).t (ts.get j).np)

theorem opnAtZJ_eq_opnD {I : Exec.Inputs} {n mPts : ℕ} {Reg : Type}
    (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)}
    {R : Reg} {spec : SessionSpec} {sts : Array Stmt} (hh : I.tags.hiddenOutputs = false)
    (hs : setupAt (ZkExec.zkInputs I) (some (dj S)) = .ok (spec, sts)) (hlen : (planZ S R).length = sts.size) :
    Composed.opnAtZJ planZ I dj S R = opnD sts (planZ S R) := by
  funext proofs j
  have hj : j.val < sts.size := hlen ▸ j.isLt
  unfold Composed.opnAtZJ opnD
  rw [(setupAt_tableJ hh hs).2 j.val hj, Array.getElem?_eq_getElem hj]
  rfl

end FlockDeploy.KS
