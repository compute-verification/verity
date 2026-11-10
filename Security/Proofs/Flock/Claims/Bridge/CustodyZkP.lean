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
import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOkJson

/-!
# The record `verify --zk --public-inputs` decodes is the live session's: `RecordsLiveZKJA`, by construction

`RecordCustodyZKJP` (`live-verifier` at `flock-verify verify --zk --public-inputs`, the acceptance `AcceptsZKP I own`
the V5 leg reads) asks of the writer `wr` that the record `AcceptsZKP I own` accepts of `wr S R o` carries `S`'s sample
and `o`'s proof files and decodes to `o`'s link points, roots and rounds (`RecordsLiveZKJA`), and of the sample files that
they are the verifier's (`DrawFileZK`). This file proves the first half at one writer, by the record's decode round
trip: the outcome a transcript records is read off its decoded record (`viewA`), and the writer writes, for an outcome,
an accepted transcript whose decoded record is that outcome (`wrA`). At any acceptance `acc` and setup `setup`:

* `RecordsLiveAtA`, `recordsLiveZKJA_iff`: `RecordsLiveZKJA` at one outcome.
* `viewA setup J mPts t`: the live outcome `t` records: at the session `setup` sets up for its record and the session
  `Record.decode` decodes the record to there (`sessOf`), the decoded `root_B`, link points and each table's two rep
  streams' rounds, with `t`'s proof files.
* `recordsLiveAt_viewA`: a transcript records its own view wherever its record carries the sample and its decoded session
  has both rep streams of each of the `J` tables.
* `wrA`: at the outcome `o` of the sample `S`, an accepted transcript that records `o`, where one exists, else a
  transcript without proof files. `recordsLiveZKJA_wrA`: it satisfies `RecordsLiveZKJA` at every acceptance that refuses
  the latter. `acc_wrA`: at an outcome some accepted transcript records, what it writes is accepted.

At the verifier's acceptance at its public inputs `own` (`ZkHidden.AcceptsZKP I own`, `Exec.setupOfP (zkInputs I) own`):

* `not_acceptsZKP_nil`: `Zk.verify` refuses a session without proof files.
* `wrZP`, `viewZP`, `recordsLiveZKJA_wrZP`, `recordCustodyZKJP_wrZP`: the writer and the view, and `RecordCustodyZKJP`
  at the writer from its sample file half alone. `recordCustodyZKJP_wrZP_printed`: at the sample file `flock-verify draw`
  prints (`djPrinted`), with no premise.
* `acceptsZKP_streams`: an accepted record's decoded session has both rep streams of each of its tables, since
  `Zk.verifyRep` refuses a table without them (S11; `zk_verify_ok`, `zk_verifyRep_ok`).
* `acceptsZKP_wrZP`: where `AcceptsZKP I own` accepts `t`, whose record carries the sample `dj S`, it accepts what `wrZP`
  writes for `t`'s view at any `J` of `t`'s tables, and that transcript records the view (`recordsLiveAt_wrZP`).
  `acceptsZKP_wrZP_planZHJP`: at the V5 leg's `J`, `planZHJP`'s tables, under hidden outputs.

The writer takes an accepted record rather than printing one from the outcome: `Json.parse`, which `Record.decode` and
`recordDraw` read the record through, is a `partial def`, so no proof reads what it returns on a printed record.
-/

@[expose] public section

namespace Claims.Bridge.CustodyZkP

open _root_.Flock FlockSoundness FlockSoundness.Discharge
open FlockSoundness.Refine (ptsOf)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Lean (Json)

variable {n mPts : ℕ} {Reg : Type}

/-! ## At any acceptance and setup -/

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

/-- The session `setup` sets up for `record`, and the session `Record.decode` decodes `record` to there. -/
def sessOf (record : String) : Option (SessionSpec × RunRecord) :=
  match setup record with
  | .ok (spec, _) => (Record.decode spec record).toOption.map (spec, ·)
  | .error _ => none

/-- Table `j`'s rep `r` stream's rounds in the decoded session `sess` of `spec` (`<table>/rep<r>`, as `Zk.verifyRep`
finds it), none where it has no such stream. -/
def repRounds (spec : SessionSpec) (sess : RunRecord) (j r : ℕ) : List Round :=
  ((sess.streams.find? fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r).map
    (·.rounds.toList)).getD []

/-- **The live outcome the transcript `t` records**, at `J` tables and `mPts` link coordinates: at its decoded record
(`sessOf`), the decoded `root_B`, link points and each table's rep streams' rounds, and `t`'s proof files; no roots,
points or rounds where the record doesn't decode. -/
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

/-- **A transcript records its own view**: where its record carries the sample `d` and, wherever it decodes, its decoded
session has both rep streams of each of the `J` tables, the transcript records `viewA` (`RecordsLiveAtA`). -/
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

/-- **`wrA` records the live session** (`RecordsLiveZKJA`) at every acceptance that refuses a transcript without proof
files. -/
theorem recordsLiveZKJA_wrA (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (hnil : ¬ acc ⟨"", #[]⟩) :
    Composed.RecordsLiveZKJA (mPts := mPts) acc setup dj Jf (wrA acc setup dj Jf) := by
  refine (recordsLiveZKJA_iff acc setup dj Jf _).2 fun S R o => ?_
  unfold wrA
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h hnil

/-- **`wrA` is accepted at an outcome an accepted transcript records.** -/
theorem acc_wrA (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ) {S : Finset (Fin n)} {R : Reg}
    {o : LiveOutJ (Jf S R) mPts} {t : Exec.Transcript} (ha : acc t) (hr : RecordsLiveAtA acc setup (dj S) o t) :
    acc (wrA acc setup dj Jf S R o) := by
  have hex : ∃ t, acc t ∧ RecordsLiveAtA acc setup (dj S) o t := ⟨t, ha, hr⟩
  unfold wrA
  rw [dite_eq_left hex]
  exact (Classical.choose_spec hex).1

end Any

/-! ## At the verifier's public inputs -/

section Zk

open ZkHidden (AcceptsZKP)
open ZkExec (zkInputs)

theorem repStreams_size_two (spec : SessionSpec) : spec.repStreams.size = 2 * spec.tables.size := by
  have e : spec.repStreams.toList = (List.range spec.tables.size).flatMap fun i =>
      [(i, 0, s!"{spec.tables[i]!}/rep{0}"), (i, 1, s!"{spec.tables[i]!}/rep{1}")] := by
    simp [SessionSpec.repStreams, Array.toList_flatMap]
  rw [← Array.length_toList, e]
  simp [List.length_flatMap, List.sum_replicate, Nat.mul_comm]

/-- **`verify --zk --public-inputs` refuses a session without proof files**: `Zk.verify` expects one per rep stream,
two per table, and at least one table. -/
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

/-- The live outcome a transcript records at the verifier's public inputs `own` (`viewA` at `setupOfP`). -/
abbrev viewZP (I : Exec.Inputs) (own : Option ByteArray) (J mPts : ℕ) (t : Exec.Transcript) : LiveOutJ J mPts :=
  viewA (Exec.setupOfP (zkInputs I) own) J mPts t

/-- **`wrZP` records the live session** (`RecordsLiveZKJA` at `AcceptsZKP I own`, the record half of
`RecordCustodyZKJP`). -/
theorem recordsLiveZKJA_wrZP (I : Exec.Inputs) (own : Option ByteArray) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) :
    Composed.RecordsLiveZKJA (mPts := mPts) (AcceptsZKP I own) (Exec.setupOfP (zkInputs I) own) dj Jf
      (wrZP I own dj Jf) :=
  recordsLiveZKJA_wrA dj Jf (not_acceptsZKP_nil I own "")

/-- **`RecordCustodyZKJP` at `wrZP`** is its sample file half (`DrawFileZK`) alone. -/
theorem recordCustodyZKJP_wrZP (I : Exec.Inputs) (own : Option ByteArray) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (kd : ℕ) (hdf : ZkLink.DrawFileZK dj (.subset kd) n) :
    Assumptions.Zk.RecordCustodyZKJP (mPts := mPts) I own dj Jf (wrZP I own dj Jf) kd :=
  ⟨recordsLiveZKJA_wrZP I own dj Jf, hdf⟩

/-- The sample file `flock-verify draw` prints for the verifier's own `subset kd` sample of `S` over the `n` proof
blocks. -/
def djPrinted (kd : ℕ) (S : Finset (Fin n)) : Json := (ZkLink.drawOf (.subset kd) n S).toJson

/-- **`RecordCustodyZKJP` at `wrZP` and the printed sample file, with no premise** (`DrawFileZK` by the sample file's round
trip, `drawFileZK_of_toJson_subset`). -/
theorem recordCustodyZKJP_wrZP_printed (I : Exec.Inputs) (own : Option ByteArray) (Jf : Finset (Fin n) → Reg → ℕ)
    (kd : ℕ) :
    Assumptions.Zk.RecordCustodyZKJP (mPts := mPts) I own (djPrinted kd) Jf (wrZP I own (djPrinted kd) Jf) kd :=
  recordCustodyZKJP_wrZP I own _ Jf kd (ZkLink.drawFileZK_of_toJson_subset _ kd n fun _ => rfl)

/-- **An accepted record has both rep streams of each of its tables**: where `AcceptsZKP I own` accepts `t`, the session
its record decodes to at its setup has, at each table `j` of the setup and each rep `r`, the stream `<table>/rep<r>`,
since `Zk.verifyRep` refuses a table without it (S11). -/
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

/-- **An accepted transcript records its view**: where `AcceptsZKP I own` accepts `t`, whose record carries the sample
`d`, `t` records `viewZP` at any `J` of its tables. -/
theorem recordsLiveAt_viewZP {I : Exec.Inputs} {own : Option ByteArray} {d : Json} {J : ℕ} {t : Exec.Transcript}
    (hd : Exec.recordDraw t.record = some d)
    (hJ : ∀ spec sts, Exec.setupOfP (zkInputs I) own t.record = .ok (spec, sts) → J ≤ sts.size) :
    RecordsLiveAtA (AcceptsZKP I own) (Exec.setupOfP (zkInputs I) own) d (viewZP I own J mPts t) t :=
  recordsLiveAt_viewA (fun _ => hd) fun ha spec sts _ hs hdec j hj r hr =>
    acceptsZKP_streams ha hs hdec j (lt_of_lt_of_le hj (hJ spec sts hs)) r hr

/-- **The verifier accepts what `wrZP` writes for an accepted transcript's view**: where `AcceptsZKP I own` accepts `t`,
whose record carries the sample `dj S`, it accepts what `wrZP` writes for `t`'s view at any `Jf S R` of `t`'s tables. -/
theorem acceptsZKP_wrZP {I : Exec.Inputs} {own : Option ByteArray} (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) {S : Finset (Fin n)} {R : Reg} {t : Exec.Transcript} (ha : AcceptsZKP I own t)
    (hd : Exec.recordDraw t.record = some (dj S))
    (hJ : ∀ spec sts, Exec.setupOfP (zkInputs I) own t.record = .ok (spec, sts) → Jf S R ≤ sts.size) :
    AcceptsZKP I own (wrZP I own dj Jf S R (viewZP I own (Jf S R) mPts t)) :=
  acc_wrA dj Jf ha (recordsLiveAt_viewZP hd hJ)

/-- **What `wrZP` writes for an accepted transcript's view records that view** (`RecordsLiveAtA`, from
`recordsLiveZKJA_wrZP`): the sample `dj S`, `t`'s proof files, and `t`'s decoded link points, roots and rounds. -/
theorem recordsLiveAt_wrZP {I : Exec.Inputs} {own : Option ByteArray} (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (S : Finset (Fin n)) (R : Reg) (t : Exec.Transcript) :
    RecordsLiveAtA (AcceptsZKP I own) (Exec.setupOfP (zkInputs I) own) (dj S) (viewZP I own (Jf S R) mPts t)
      (wrZP I own dj Jf S R (viewZP I own (Jf S R) mPts t)) :=
  (recordsLiveZKJA_iff _ _ dj Jf _).1 (recordsLiveZKJA_wrZP I own dj Jf) S R _

/-- **`acceptsZKP_wrZP` at the V5 leg's tables** (`planZHJP`'s, `nTab I` of them): under hidden outputs, a record whose
sample is `dj S` sets up `nTab I` tables (`setupOf_tableJHP`). -/
theorem acceptsZKP_wrZP_planZHJP {I : Exec.Inputs} {own : Option ByteArray} (hh : I.tags.hiddenOutputs = true)
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : ZkHidden.DrawSetupZHJP (Composed.inputsJ I 0) own mPts S₀ d₀) {S : Finset (Fin n)} {R : Reg}
    {t : Exec.Transcript} (ha : AcceptsZKP I own t) (hd : Exec.recordDraw t.record = some (dj S)) :
    AcceptsZKP I own (wrZP I own dj (fun S R => (ZkHidden.planZHJP (Reg := Reg) dj y₀ S R).length) S R
      (viewZP I own (ZkHidden.planZHJP (Reg := Reg) dj y₀ S R).length mPts t)) :=
  acceptsZKP_wrZP dj _ ha hd fun _ _ hs => by
    rw [ZkHidden.planZHJP_length, (ZkHidden.setupOf_tableJHP hh hd hs).1]

end Zk

end Claims.Bridge.CustodyZkP
