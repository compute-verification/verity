import Proofs.Flock.Claims.KS.PlainDrawAcc
import Proofs.Flock.Claims.KS.PlainDrawDecK

/-!
# The deployed plain decoders at the statement's draw

`plain_hdec` and `plain_hdecH` are right for `plainVerdict`, whose acceptance sets the session up at the record's
draw. Here the same decoders are right for `plainVerdictAt`, which sets it up at the statement's draw `d` and reads no
draw in the record: `PlainDrawDecK`'s decoders at `keep := id` and the writer `wrPD`, which records its outcome there
(`recordsLiveAtDK_id_of`), with `AcceptsAtD` and `TxNoOffAt` for `AcceptsAtDK id` and `TxNoOffAtK id`
(`acceptsAtD_iff_acceptsAtDK_id`, `txNoOffAt_iff_txNoOffAtK_id`):

* `recordsLive_wrPD1`: `wrPD` at a one-table outcome records it as one table's (`recordsLive_wrP1` at the setup at
  `d`), its rounds the outcome's.
* `decodesTable_wrPD_sc`, `decodesTable_wrPD_scH`: `decodesTable_wrK_sc_of` and `decodesTable_wrK_scH_of` at `wrPD`,
  `decodesTable_wrP_sc` and `decodesTable_wrP_scH` with the verdict at the draw `dj S`.
* `plain_hdecAt`, `plain_hdecHAt`: `plain_hdecAtK_of` and `plain_hdecHAtK_of` at `wrPD`, `plain_hdec` and
  `plain_hdecH` with `plainVerdictAt`.
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **`wrPD` records one table's live session at the setup at `d`**: its proof files are the outcome's, and the
session its record decodes to there has the outcome's root, link points and rep streams' rounds, read as one table's
(`oneOf`). -/
theorem recordsLive_wrPD1 (I : Exec.Inputs) (d : Json) {mPts : ℕ} (o : Tables.LiveOutJ 1 mPts)
    (hacc : AcceptsAtD I (some d) (wrPD I d o)) :
    (wrPD I d o).proofs = (oneOf o).proofs ∧
    ∀ spec sts sess, setupAtD I (some d) = .ok (spec, sts) → Record.decode spec (wrPD I d o).record = .ok sess →
      sess.rootB[0]! = (oneOf o).rootB ∧ ptsOf mPts sess.points = (oneOf o).pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = pick ((oneOf o).rep0, (oneOf o).rep1) r := by
  obtain ⟨hpr, hrec⟩ := recordsLiveAtD_wrPD I d o hacc
  refine ⟨hpr, fun spec sts sess hset hdec => ?_⟩
  obtain ⟨hpts, hj⟩ := hrec spec sts sess hset hdec
  obtain ⟨hroot, hstr⟩ := hj 0
  refine ⟨hroot, hpts, fun r => ?_⟩
  obtain ⟨s, hs, hr⟩ := hstr r
  refine ⟨s, hs, ?_⟩
  rw [hr]
  fin_cases r <;> rfl

open DecodeZL Integrate in
/-- **The deployed plain decoders are right at `wrPD` for accepting with no opening off `C` without a self-clash**, at
one table: `decodesTable_wrP_sc` on the outcomes whose transcript flock-verify accepts at the setup at the draw `dj S`,
whose root is `rootB₀`'s and on which the deployed plain reader looks up no column off `C` there (`TxNoOffAt`). -/
theorem decodesTable_wrPD_sc {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch o))
      (fun o => AcceptsAtD I (some (dj S)) (wrPD I (dj S) o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAt I (dj S) C (wrPD I (dj S) o))
      (fun ps _ => DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R ps)
      (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].get j)) j := by
  intro o hv
  obtain ⟨hacc, hr₀, hno⟩ := hv
  exact decodesTable_wrK_sc_of id ht hc ho dj x₀ hT hS R hN C rootB₀ (wrPD I (dj S))
    (fun o => recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) j o
    ⟨(acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc, hr₀, (txNoOffAt_iff_txNoOffAtK_id _ _ _ _).1 hno⟩

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at `wrPD` at hidden outputs**: `decodesTable_wrPD_sc` at the hidden tables
and their openings `opnAtH` (`decodesTable_wrP_scH` at the draw `dj S`). -/
theorem decodesTable_wrPD_scH {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leaf512 (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => AcceptsAtD I (some (dj S)) (wrPD I (dj S) o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAt I (dj S) C (wrPD I (dj S) o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j := by
  intro o hv
  obtain ⟨hacc, hr₀, hno⟩ := hv
  exact decodesTable_wrK_scH_of id ht hc dj x₀ hT hS R hN C rootB₀ (wrPD I (dj S))
    (fun o => recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) j o
    ⟨(acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc, hr₀, (txNoOffAt_iff_txNoOffAtK_id _ _ _ _).1 hno⟩

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw**: `plain_hdec` with
`plainVerdictAt` for `plainVerdict`, from `decodesTable_wrPD_sc` at `d` and no drawn unit. -/
theorem plain_hdecAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAt ins.read.I d C rootB₀)
      (fun ps _ => DecodeZL.opnAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j := by
  intro j o hv
  obtain ⟨hacc, hr₀, hno⟩ := hv
  exact plain_hdecAtK_of id hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay C rootB₀ (wrPD ins.read.I d)
    (fun o => recordsLiveAtDK_id_of (recordsLiveAtD_wrPD ins.read.I d o)) j o
    ⟨(acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc, hr₀, (txNoOffAt_iff_txNoOffAtK_id _ _ _ _).1 hno⟩

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at hidden outputs and
`hm96-sha512/v1` leaves**: `plain_hdecH` with `plainVerdictAt` for `plainVerdict`, from `decodesTable_wrPD_scH`. -/
theorem plain_hdecHAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAt ins.read.I d C rootB₀)
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j := by
  intro j o hv
  obtain ⟨hacc, hr₀, hno⟩ := hv
  exact plain_hdecHAtK_of id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay C rootB₀ (wrPD ins.read.I d)
    (fun o => recordsLiveAtDK_id_of (recordsLiveAtD_wrPD ins.read.I d o)) j o
    ⟨(acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc, hr₀, (txNoOffAt_iff_txNoOffAtK_id _ _ _ _).1 hno⟩

end FlockDeploy.KS
