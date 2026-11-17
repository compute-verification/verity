module

public import Proofs.Flock.Claims.KS.PlainSampleKDefs
public import Proofs.Flock.Soundness.Discharge.Exec.VerifyKJ
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Bridge.CustodyPlain
import Proofs.Flock.Claims.KS.ConflictOC

/-! The plain verdict at the statement's sample at a parsed value (`TranscriptJ`), which need not be any text's parse. -/

@[expose] public section

namespace Claims.Bridge.PlainSample

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open FlockSoundness.Discharge.Exec (verifyWithMK_ok_iff)
open FlockSoundness.Refine (ptsOf shapeOf ebind_ok)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

structure TranscriptJ where
  j : Json
  proofs : Array ByteArray

def AcceptsAtDJK (keep : Bool → Bool) (I : Exec.Inputs) (dr : Option Json) (t : TranscriptJ) : Prop :=
  ∃ spec sts, setupAtD I dr = .ok (spec, sts) ∧
    verifyJWithMK keep With.std spec (sts.map _root_.Flock.Setup.ofCircuit) t.j t.proofs = .ok ()

def RecordsLiveAtDJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts)
    (t : TranscriptJ) : Prop :=
  AcceptsAtDJK keep I (some d) t →
    t.proofs = o.proofs ∧
    ∀ spec sts sess, setupAtD I (some d) = .ok (spec, sts) →
      Record.decodeJWithK keep With.std spec t.j = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList.map Round.erase = (o.reps j r).map Round.erase

def TxNoOffAtJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {N : ℕ} (C : Fin N → List 𝔽) (t : TranscriptJ) :
    Prop :=
  ∀ (spec : SessionSpec) (st : _root_.Flock.Stmt) (sess : RunRecord) (sch : _root_.Flock.Schedule) (s₀ s₁ : StreamRec),
    setupAtD I (some d) = .ok (spec, #[st]) → Record.decodeJWithK keep With.std spec t.j = .ok sess →
    _root_.Flock.fast100 st.m = some sch →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ →
    sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ →
    ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₀.rounds t.proofs[0]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) s₁.rounds
          t.proofs[1]!)) p.val = some u ∧ u ≠ C p

open Classical in
noncomputable def wrPDJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) :
    TranscriptJ :=
  if h : ∃ t, AcceptsAtDJK keep I (some d) t ∧ RecordsLiveAtDJK keep I d o t then h.choose else ⟨Json.null, #[]⟩

theorem verifyJWithMK_nil (keep : Bool → Bool) (hs : Hashes _root_.Flock.V) (spec : SessionSpec) (tabs : Array Setup)
    (j : Json) : verifyJWithMK keep hs spec tabs j #[] ≠ .ok () := by
  intro h
  unfold verifyJWithMK at h
  simp only [] at h
  by_cases c1 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c1] at h
    obtain ⟨_, h', _⟩ := ebind_ok h
    cases h'
  rw [ite_eq_right c1] at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  have c3 : ((#[] : Array ByteArray).size != spec.repStreams.size) = true := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at c1
    simp only [List.size_toArray, List.length_nil, Custody.repStreams_size, bne_iff_ne, ne_eq]
    omega
  rw [ite_eq_left c3] at h
  obtain ⟨_, h', _⟩ := ebind_ok h
  cases h'

theorem not_acceptsAtDJK_nil (keep : Bool → Bool) (I : Exec.Inputs) (dr : Option Json) (j : Json) :
    ¬ AcceptsAtDJK keep I dr ⟨j, #[]⟩ := by
  rintro ⟨spec, sts, -, h⟩
  exact verifyJWithMK_nil keep _ _ _ _ h

theorem recordsLiveAtDJK_wrPDJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {J mPts : ℕ}
    (o : Tables.LiveOutJ J mPts) : RecordsLiveAtDJK keep I d o (wrPDJK keep I d o) := by
  unfold wrPDJK
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_acceptsAtDJK_nil _ _ _ _)

theorem acceptsAtDJK_wrPDJK {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ}
    {o : Tables.LiveOutJ J mPts} {t : TranscriptJ} (hA : AcceptsAtDJK keep I (some d) t)
    (hR : RecordsLiveAtDJK keep I d o t) : AcceptsAtDJK keep I (some d) (wrPDJK keep I d o) := by
  have hex : ∃ t, AcceptsAtDJK keep I (some d) t ∧ RecordsLiveAtDJK keep I d o t := ⟨t, hA, hR⟩
  unfold wrPDJK
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

theorem acceptsAtDK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {dr : Option Json} {text : String} {j : Json}
    (proofs : Array ByteArray) (h : Flock.JsonParse.parse text = .ok j) :
    AcceptsAtDK keep I dr ⟨text, proofs⟩ ↔ AcceptsAtDJK keep I dr ⟨j, proofs⟩ := by
  have hv : ∀ spec tabs, verifyWithMK keep With.std spec tabs text proofs =
      verifyJWithMK keep With.std spec tabs j proofs := fun spec tabs =>
    verifyWithMK_of_parse keep With.std spec tabs proofs h
  simp only [AcceptsAtDK, AcceptsAtDJK, hv]

theorem acceptsAtDK_iff_exists_J {keep : Bool → Bool} {I : Exec.Inputs} {dr : Option Json} (t : Exec.Transcript) :
    AcceptsAtDK keep I dr t ↔ ∃ j, Flock.JsonParse.parse t.record = .ok j ∧ AcceptsAtDJK keep I dr ⟨j, t.proofs⟩ := by
  refine ⟨fun ⟨spec, sts, hs, hv⟩ => ?_, fun ⟨j, hj, hA⟩ => (acceptsAtDK_iff_J t.proofs hj).2 hA⟩
  obtain ⟨j, hj, hvj⟩ := (verifyWithMK_ok_iff keep With.std spec _ t.record t.proofs).1 hv
  exact ⟨j, hj, spec, sts, hs, hvj⟩

theorem recordsLiveAtDK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ}
    {o : Tables.LiveOutJ J mPts} {text : String} {j : Json} (proofs : Array ByteArray) (h : Flock.JsonParse.parse text = .ok j) :
    RecordsLiveAtDK keep I d o ⟨text, proofs⟩ ↔ RecordsLiveAtDJK keep I d o ⟨j, proofs⟩ := by
  have hdec : ∀ spec, Record.decodeWithK keep With.std spec text = Record.decodeJWithK keep With.std spec j :=
    fun spec => Record.decodeWithK_of_parse keep With.std spec h
  simp only [RecordsLiveAtDK, RecordsLiveAtDJK, acceptsAtDK_iff_J proofs h, hdec]

theorem txNoOffAtK_iff_J {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {N : ℕ} {C : Fin N → List 𝔽}
    {text : String} {j : Json} (proofs : Array ByteArray) (h : Flock.JsonParse.parse text = .ok j) :
    TxNoOffAtK keep I d C ⟨text, proofs⟩ ↔ TxNoOffAtJK keep I d C ⟨j, proofs⟩ := by
  have hdec : ∀ spec, Record.decodeWithK keep With.std spec text = Record.decodeJWithK keep With.std spec j :=
    fun spec => Record.decodeWithK_of_parse keep With.std spec h
  simp only [TxNoOffAtK, TxNoOffAtJK, hdec]

theorem acceptsAtDJK_wrPDJK_of_K {keep : Bool → Bool} {I : Exec.Inputs} {d : Json} {J mPts : ℕ}
    {o : Tables.LiveOutJ J mPts} {t : Exec.Transcript} (hA : AcceptsAtDK keep I (some d) t)
    (hR : RecordsLiveAtDK keep I d o t) : AcceptsAtDJK keep I (some d) (wrPDJK keep I d o) := by
  obtain ⟨j, hj, hAj⟩ := (acceptsAtDK_iff_exists_J t).1 hA
  exact acceptsAtDJK_wrPDJK hAj ((recordsLiveAtDK_iff_J t.proofs hj).1 hR)

end Claims.Bridge.PlainSample

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainSample
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

def plainVerdictAtJK (keep : Bool → Bool) (I : Exec.Inputs) (d : Json) {N mPts : ℕ} (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (o : Tables.LiveOutJ 1 mPts) : Prop :=
  AcceptsAtDJK keep I (some d) (wrPDJK keep I d o) ∧ o.rootB[0]! = rootB₀[0]! ∧
    TxNoOffAtJK keep I d C (wrPDJK keep I d o)

end FlockDeploy.KS
