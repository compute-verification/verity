module

public import Proofs.Flock.Soundness.Assumptions.ZkP
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
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Proofs.Flock.Soundness.Discharge.ZkLink.SampleOkJson

/-! The record `verify --zk --public-inputs` decodes is the live session's (`RecordsLiveZKJA`), by
construction: the writer takes an accepted record, since `Json.parse` is a `partial def` that no proof reads
on a printed record. -/

@[expose] public section

namespace Claims.Bridge.CustodyZkP

open _root_.Flock FlockSoundness FlockSoundness.Discharge
open FlockSoundness.Refine (ptsOf)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Lean (Json)

variable {n mPts : ℕ} {Reg : Type}

section Any

variable (acc : Exec.Transcript → Prop) (setup : String → V (SessionSpec × Array Stmt))

/-- **`RecordsLiveZKJA` at one outcome**: where `acc` accepts `t`, its record carries the sample `d`, its proof files are
`o`'s, and the session it decodes to, at the session `setup` sets up for it, has `o`'s link points and, at each of `o`'s
`J` tables, `o`'s `root_B` and rep streams' rounds. -/
def RecordsLiveAtA (d : Json) {J : ℕ} (o : LiveOutJ J mPts) (t : Exec.Transcript) : Prop :=
  acc t → Exec.recordDraw t.record = some d ∧ t.proofs = o.proofs ∧
    ∀ spec sts sess, setup t.record = .ok (spec, sts) → Record.decode spec t.record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

theorem recordsLiveZKJA_iff (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript) :
    Composed.RecordsLiveZKJA acc setup dj Jf wr ↔ ∀ S R o, RecordsLiveAtA acc setup (dj S) o (wr S R o) :=
  Iff.rfl

def sessOf (record : String) : Option (SessionSpec × RunRecord) :=
  match setup record with
  | .ok (spec, _) => (Record.decode spec record).toOption.map (spec, ·)
  | .error _ => none

def repRounds (spec : SessionSpec) (sess : RunRecord) (j r : ℕ) : List Round :=
  ((sess.streams.find? fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r).map
    (·.rounds.toList)).getD []

def viewA (J mPts : ℕ) (t : Exec.Transcript) : LiveOutJ J mPts where
  rootB := ((sessOf setup t.record).map (·.2.rootB)).getD #[]
  pts := ptsOf mPts (((sessOf setup t.record).map (·.2.points)).getD #[])
  reps j r := ((sessOf setup t.record).map fun x => repRounds x.1 x.2 j.val r.val).getD []
  proofs := t.proofs

variable {acc setup}

theorem sessOf_eq {record : String} {spec : SessionSpec} {sts : Array Stmt} {sess : RunRecord}
    (hs : setup record = .ok (spec, sts)) (hd : Record.decode spec record = .ok sess) :
    sessOf setup record = some (spec, sess) := by
  simp only [sessOf, hs, hd]
  rfl

theorem recordsLiveAt_viewA {d : Json} {J : ℕ} {t : Exec.Transcript}
    (hd : acc t → Exec.recordDraw t.record = some d)
    (hs : acc t → ∀ spec sts sess, setup t.record = .ok (spec, sts) → Record.decode spec t.record = .ok sess →
      ∀ j < J, ∀ r < 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r) = some s) :
    RecordsLiveAtA acc setup d (viewA setup J mPts t) t := by
  intro ha
  refine ⟨hd ha, rfl, fun spec sts sess hset hdec => ?_⟩
  have e := sessOf_eq hset hdec
  refine ⟨by simp only [viewA, e, Option.map_some, Option.getD_some], fun j =>
    ⟨by simp only [viewA, e, Option.map_some, Option.getD_some], fun r => ?_⟩⟩
  obtain ⟨s, hsf⟩ := hs ha spec sts sess hset hdec j.val j.isLt r.val r.isLt
  exact ⟨s, hsf, by simp only [viewA, e, Option.map_some, Option.getD_some, repRounds, hsf]⟩

variable (acc setup)

open Classical in
/-- **The writer that keeps an accepted record**: at the live outcome `o` of the sample `S`, a transcript `acc` accepts
that records `o` (`RecordsLiveAtA`), where one exists, else a transcript without proof files. -/
noncomputable def wrA (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ) (S : Finset (Fin n)) (R : Reg)
    (o : LiveOutJ (Jf S R) mPts) : Exec.Transcript :=
  if h : ∃ t, acc t ∧ RecordsLiveAtA acc setup (dj S) o t then h.choose else ⟨"", #[]⟩

variable {acc setup}

theorem recordsLiveZKJA_wrA (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (hnil : ¬ acc ⟨"", #[]⟩) :
    Composed.RecordsLiveZKJA (mPts := mPts) acc setup dj Jf (wrA acc setup dj Jf) := by
  refine (recordsLiveZKJA_iff acc setup dj Jf _).2 fun S R o => ?_
  unfold wrA
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h hnil

theorem acc_wrA (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ) {S : Finset (Fin n)} {R : Reg}
    {o : LiveOutJ (Jf S R) mPts} {t : Exec.Transcript} (ha : acc t) (hr : RecordsLiveAtA acc setup (dj S) o t) :
    acc (wrA acc setup dj Jf S R o) := by
  have hex : ∃ t, acc t ∧ RecordsLiveAtA acc setup (dj S) o t := ⟨t, ha, hr⟩
  unfold wrA
  rw [dite_eq_left hex]
  exact (Classical.choose_spec hex).1

end Any

section Zk

open ZkHidden (AcceptsZKP)
open ZkExec (zkInputs)

theorem repStreams_size_two (spec : SessionSpec) : spec.repStreams.size = 2 * spec.tables.size := by
  have e : spec.repStreams.toList = (List.range spec.tables.size).flatMap fun i =>
      [(i, 0, s!"{spec.tables[i]!}/rep{0}"), (i, 1, s!"{spec.tables[i]!}/rep{1}")] := by
    simp [SessionSpec.repStreams, Array.toList_flatMap]
  rw [← Array.length_toList, e]
  simp [List.length_flatMap, List.sum_replicate, Nat.mul_comm]

theorem not_acceptsZKP_nil (I : Exec.Inputs) (own : Option ByteArray) (record : String) :
    ¬ AcceptsZKP I own ⟨record, #[]⟩ := by
  rintro ⟨spec, sts, -, -, h⟩
  obtain ⟨_, _, _, _, htab, hpos, hp, _⟩ := ZkExec.zk_verify_ok _ _ _ _ h
  have hp' : spec.repStreams.size = 0 := hp.symm
  rw [repStreams_size_two] at hp'
  omega

/-- The writer at the verifier's public inputs `own` (`wrA` at `AcceptsZKP I own`). -/
noncomputable abbrev wrZP (I : Exec.Inputs) (own : Option ByteArray) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript :=
  wrA (AcceptsZKP I own) (Exec.setupOfP (zkInputs I) own) dj Jf

abbrev viewZP (I : Exec.Inputs) (own : Option ByteArray) (J mPts : ℕ) (t : Exec.Transcript) : LiveOutJ J mPts :=
  viewA (Exec.setupOfP (zkInputs I) own) J mPts t

theorem recordsLiveZKJA_wrZP (I : Exec.Inputs) (own : Option ByteArray) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) :
    Composed.RecordsLiveZKJA (mPts := mPts) (AcceptsZKP I own) (Exec.setupOfP (zkInputs I) own) dj Jf
      (wrZP I own dj Jf) :=
  recordsLiveZKJA_wrA dj Jf (not_acceptsZKP_nil I own "")

theorem recordCustodyZKJP_wrZP (I : Exec.Inputs) (own : Option ByteArray) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (kd : ℕ) (hdf : ZkLink.SampleFileZK dj (.subset kd) n) :
    Assumptions.Zk.RecordCustodyZKJP (mPts := mPts) I own dj Jf (wrZP I own dj Jf) kd :=
  ⟨recordsLiveZKJA_wrZP I own dj Jf, hdf⟩

def djPrinted (kd : ℕ) (S : Finset (Fin n)) : Json := (ZkLink.drawOf (.subset kd) n S).toJson

/-- **`RecordCustodyZKJP` at `wrZP` and the printed sample file, with no premise** (`SampleFileZK` by the sample file's round
trip, `sampleFileZK_of_toJson_subset`). -/
theorem recordCustodyZKJP_wrZP_printed (I : Exec.Inputs) (own : Option ByteArray) (Jf : Finset (Fin n) → Reg → ℕ)
    (kd : ℕ) :
    Assumptions.Zk.RecordCustodyZKJP (mPts := mPts) I own (djPrinted kd) Jf (wrZP I own (djPrinted kd) Jf) kd :=
  recordCustodyZKJP_wrZP I own _ Jf kd (ZkLink.sampleFileZK_of_toJson_subset _ kd n fun _ => rfl)

theorem acceptsZKP_streams {I : Exec.Inputs} {own : Option ByteArray} {t : Exec.Transcript}
    (ha : AcceptsZKP I own t) {spec : SessionSpec} {sts : Array Stmt} {sess : RunRecord}
    (hs : Exec.setupOfP (zkInputs I) own t.record = .ok (spec, sts)) (hd : Record.decode spec t.record = .ok sess) :
    ∀ j < sts.size, ∀ r < 2, ∃ s : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r) = some s := by
  obtain ⟨spec', sts', hs', -, hv⟩ := ha
  rw [hs] at hs'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
  obtain ⟨-, sess', schs, hd', -, -, -, -, hrep⟩ := ZkExec.zk_verify_ok _ _ _ _ hv
  rw [hd] at hd'
  cases hd'
  intro j hj r hr
  obtain ⟨_, c, _, h0, h1⟩ := hrep j (by simpa using hj)
  interval_cases r
  · obtain ⟨s, _, _, _, hf, _⟩ := ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ h0
    exact ⟨s, hf⟩
  · obtain ⟨s, _, _, _, hf, _⟩ := ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ h1
    exact ⟨s, hf⟩

theorem recordsLiveAt_viewZP {I : Exec.Inputs} {own : Option ByteArray} {d : Json} {J : ℕ} {t : Exec.Transcript}
    (hd : Exec.recordDraw t.record = some d)
    (hJ : ∀ spec sts, Exec.setupOfP (zkInputs I) own t.record = .ok (spec, sts) → J ≤ sts.size) :
    RecordsLiveAtA (AcceptsZKP I own) (Exec.setupOfP (zkInputs I) own) d (viewZP I own J mPts t) t :=
  recordsLiveAt_viewA (fun _ => hd) fun ha spec sts _ hs hdec j hj r hr =>
    acceptsZKP_streams ha hs hdec j (lt_of_lt_of_le hj (hJ spec sts hs)) r hr

theorem acceptsZKP_wrZP {I : Exec.Inputs} {own : Option ByteArray} (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) {S : Finset (Fin n)} {R : Reg} {t : Exec.Transcript} (ha : AcceptsZKP I own t)
    (hd : Exec.recordDraw t.record = some (dj S))
    (hJ : ∀ spec sts, Exec.setupOfP (zkInputs I) own t.record = .ok (spec, sts) → Jf S R ≤ sts.size) :
    AcceptsZKP I own (wrZP I own dj Jf S R (viewZP I own (Jf S R) mPts t)) :=
  acc_wrA dj Jf ha (recordsLiveAt_viewZP hd hJ)

theorem recordsLiveAt_wrZP {I : Exec.Inputs} {own : Option ByteArray} (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (S : Finset (Fin n)) (R : Reg) (t : Exec.Transcript) :
    RecordsLiveAtA (AcceptsZKP I own) (Exec.setupOfP (zkInputs I) own) (dj S) (viewZP I own (Jf S R) mPts t)
      (wrZP I own dj Jf S R (viewZP I own (Jf S R) mPts t)) :=
  (recordsLiveZKJA_iff _ _ dj Jf _).1 (recordsLiveZKJA_wrZP I own dj Jf) S R _

theorem acceptsZKP_wrZP_planZHJP {I : Exec.Inputs} {own : Option ByteArray} (hh : I.tags.hiddenOutputs = true)
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : ZkHidden.SampleSetupZHJP (Composed.inputsJ I 0) own mPts S₀ d₀) {S : Finset (Fin n)} {R : Reg}
    {t : Exec.Transcript} (ha : AcceptsZKP I own t) (hd : Exec.recordDraw t.record = some (dj S)) :
    AcceptsZKP I own (wrZP I own dj (fun S R => (ZkHidden.planZHJP (Reg := Reg) dj y₀ S R).length) S R
      (viewZP I own (ZkHidden.planZHJP (Reg := Reg) dj y₀ S R).length mPts t)) :=
  acceptsZKP_wrZP dj _ ha hd fun _ _ hs => by
    rw [ZkHidden.planZHJP_length, (ZkHidden.setupOf_tableJHP hh hd hs).1]

end Zk

end Claims.Bridge.CustodyZkP
