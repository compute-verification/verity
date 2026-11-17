module

public import Proofs.Flock.Claims.KS.PlainLeafDec

/-! `PlainClash0` and `PlainClashLater` at hidden outputs under plain SHA-512 leaves (`Refine.leafPlain`). -/

@[expose] public section

namespace FlockSoundness.Discharge.AtSample

open Claims.Bridge.PlainSample (setupAtD AcceptsAtD AcceptsAtDK named_of_setupAtD acceptsAtD_iff_acceptsAtDK_id)
open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

theorem accepts_opensLDUK (keep : Bool → Bool)
    {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtDK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeWithK keep With.std spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 leafPlain cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          pend₀.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₀.ligerito.recursiveCaps[j]!) ∧
          pend₁.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₁.ligerito.recursiveCaps[j]!) ∧
          ∀ ℓ, 1 ≤ ℓ →
            (slotOpensL out 0 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ℓ ∧
            (slotOpensL out 1 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! ℓ :=
  accepts_opensLDLfK keep leafPlain h

theorem accepts_opensLDU {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 leafPlain cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          pend₀.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₀.ligerito.recursiveCaps[j]!) ∧
          pend₁.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p₁.ligerito.recursiveCaps[j]!) ∧
          ∀ ℓ, 1 ≤ ℓ →
            (slotOpensL out 0 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ℓ ∧
            (slotOpensL out 1 ℓ (lvOf (schedOf sch) ℓ).queries).map (fun x => (x.1, x.2.1)) =
              FlockDeploy.KS.plainLevelColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! ℓ := by
  simpa only [decodeWithK_id_std] using accepts_opensLDUK id ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

end FlockSoundness.Discharge.AtSample

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainSample
  Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

def Clash₀LU (sch : FlockSoundness.Accounting.Schedule) (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ClashL H512 enc512 leafPlain out.cap₀ (lvOf sch 0).lanes (Model.pathLen (lvOf sch 0)) (opens₀L sch out)

def LaterClashLU (sch : FlockSoundness.Accounting.Schedule) (out : Model.OutCL 𝔽 Dig (List UInt8)) : Prop :=
  ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).caps[ℓ - 1]? = some cap ∧
    ClashL H512 enc512 leafPlain cap (lvOf sch ℓ).lanes (Model.pathLen (lvOf sch ℓ))
      (slotOpensL out r ℓ (lvOf sch ℓ).queries)

end FlockDeploy.KS

namespace Claims.Bridge.PlainSample

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtSample
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- Its proof is `laterEntriesClashK_of`'s, which reads no leaf. -/
theorem laterEntriesClashK_ofU (keep : Bool → Bool)
    {sch : FlockSoundness.Accounting.Schedule} {I : Exec.Inputs} {d : Json} {t : Exec.Transcript}
    {out : Model.OutCL 𝔽 Dig (List UInt8)} (he : ∀ r < 2, LaterEntriesAtK keep sch I d t out r)
    (hcl : FlockDeploy.KS.LaterClashLU sch out) : LaterEntriesClashK keep I d t :=
  laterEntriesClashKLf_of keep leafPlain he hcl

theorem laterEntriesClash_ofU {sch : FlockSoundness.Accounting.Schedule} {I : Exec.Inputs} {d : Json} {t : Exec.Transcript}
    {out : Model.OutCL 𝔽 Dig (List UInt8)} (he : ∀ r < 2, LaterEntriesAt sch I d t out r)
    (hcl : FlockDeploy.KS.LaterClashLU sch out) : LaterEntriesClash I d t :=
  (laterEntriesClash_iff_K_id _ _ _).2 <|
    laterEntriesClashK_ofU id (fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).1 (he r hr)) hcl

open DecodeZL HiddenExec HiddenDec in
theorem clash₀_wrKHU_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : SampleOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leafPlain (capOfRoot o.rootB[0]!)
      (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0))
      (opens₀L (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leafPlain
        (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)))) :
    EntriesClashK keep I (dj S) t :=
  clash₀_wrKHLf_of keep LeafForm.plain ht hc dj x₀ hT hS R o hrl hacc pend' hp hcl

open DecodeZL HiddenExec HiddenDec in
theorem clash₀_wrPDHU {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : SampleOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leafPlain (capOfRoot o.rootB[0]!)
      (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0))
      (opens₀L (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leafPlain
        (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)))) :
    EntriesClash I (dj S) (wrPD I (dj S) o) :=
  (entriesClash_iff_K_id _ _ _).2 <| clash₀_wrKHU_of id ht hc dj x₀ hT hS R o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp hcl

open DecodeZL HiddenExec HiddenDec in
theorem laterEntries_wrKHU_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : SampleOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep I (dj S) o t)
    (hacc : AcceptsAtDK keep I (some (dj S)) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) t
      (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs))
      r :=
  laterEntries_wrKHLf_of keep LeafForm.plain ht hc dj x₀ hT hS R o hrl hacc pend' hp

open DecodeZL HiddenExec HiddenDec in
theorem laterEntries_wrPDHU {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : SampleOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAt (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) (wrPD I (dj S) o)
      (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)) r :=
  fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).2 <| laterEntries_wrKHU_of id ht hc dj x₀ hT hS R o
    (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD I (dj S) o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp
    r hr

open DecodeZL HiddenExec HiddenDec in
theorem laterClash_wrPDHU {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : SampleSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : SampleOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtD I (some (dj S)) (wrPD I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashLU (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leafPlain
      (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs))) :
    LaterEntriesClash I (dj S) (wrPD I (dj S) o) :=
  laterEntriesClash_ofU (laterEntries_wrPDHU ht hc dj x₀ hT hS R o hacc pend' hp) hcl

end Claims.Bridge.PlainSample

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainSample
  Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

theorem plain_clash₀HUAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true)
    (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json} (hd : ins.read.draw = some d)
    (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep ins.read.I d o t) (hacc : AcceptsAtDK keep ins.read.I (some d) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀LU ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClashK keep ins.read.I d t :=
  plain_clash₀HLfAtK_of keep AtSample.LeafForm.plain hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hrl hacc
    pend' hp hcl

theorem plain_clash₀HUAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true)
    (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀LU ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClash ins.read.I d (wrPD ins.read.I d o) :=
  (entriesClash_iff_K_id _ _ _).2 <| plain_clash₀HUAtK_of id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay
    o (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD _ d o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp hcl

theorem plain_laterEntriesHUAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    {t : Exec.Transcript} (hrl : RecordsLiveAtDK keep ins.read.I d o t) (hacc : AcceptsAtDK keep ins.read.I (some d) t)
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d t
      (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r :=
  plain_laterEntriesHLfAtK_of keep AtSample.LeafForm.plain hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hrl
    hacc pend' hp

theorem plain_laterEntriesHUAt {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAt ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPD ins.read.I d o)
      (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r :=
  fun r hr => (laterEntriesAt_iff_K_id _ _ _ _ _ _).2 <|
    plain_laterEntriesHUAtK_of id hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o
      (recordsLiveAtDK_id_of (recordsLiveAtD_wrPD _ d o)) ((acceptsAtD_iff_acceptsAtDK_id _ _ _).1 hacc) pend' hp r hr

theorem plain_laterClashHUAt {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o)) (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : LaterClashLU ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClash ins.read.I d (wrPD ins.read.I d o) :=
  laterEntriesClash_ofU
    (plain_laterEntriesHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp) hcl

def plainDecQU {mPts : ℕ} (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) (T : TabSpec mPts) (I : Exec.Inputs)
    (d : Json) {M : ℕ} (C : Fin M → List 𝔽) (rootB₀ : Array ByteArray)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)) (o : LiveOutJ 1 mPts) :
    Prop :=
  ∃ a, Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
    (plainVerdictAt I d C rootB₀) opn (plainDec T) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (fun j => repsOf (D := Refine.Dig) execArith ([T].get j)) o a ∧
    Q (a.getD 0 dfltOutL)

def PlainClashQU {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {spec : SessionSpec} (st : Stmt)
    (T : TabSpec spec.mPts) (hw : 2 * spec.mPts ≤ 65536) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (Q : Model.OutCL 𝔽 Refine.Dig (List UInt8) → Prop) : Prop :=
  ∀ (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536),
    plainDecQU Q T I d C (plainRoot P x ω) opn (liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec)
      (plainW spec hw) sha512 (plainSim P x spec st) x (ω, u) r') →
    (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).out = .acc →
    PlainRawClashL P x ω (plainΦ x spec st (u, r'))

theorem plainClashQ₀U {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : FlockSoundness.Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        Clash₀LU sch (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        EntriesClash ins.read.I d (wrPD ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQU P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (Clash₀LU sch) := by
  intro C u r' hq hacc
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨a, ⟨hV, pend, hpl, rfl⟩, hcl⟩ := hq
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hcl
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := acc_decide hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRun P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hV hpl hcl
  obtain ⟨spec'', st', sess, sch', s₀, s₁, hsu, hdec, hf', hs₀, hs₁, q, a₁, b₁, ha, hb, hab⟩ :=
    hE _ hV.1 (pend 0) (hpl 0) hcl
  have hacc' := run_acc_next sha512 P (V false) x ω _ hacc
  replace hacc' : ((next false x ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc'
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hsu₀⟩ := setupOf_some (act_decide hact)
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtD_wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec #[st'] sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decision_runs hsu₀ hrun
  have hv' : verifyWithM (hashesIn ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdict_within hvr
  have hcols := rawCols_plain (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec) →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hcl' : ∀ {q : ℕ} {v : List 𝔽}, (q, v) ∈ Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch'
        (2 * st'.regions.size)) y₀.rounds (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[0]! ++
      Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st') sch' (2 * st'.regions.size)) y₁.rounds
        (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[1]! →
      (q, v) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen := by
    intro q v hqv
    rw [hpr', plainRepColsOf_congr _ _ (hco 0 y₀ hr₀), plainRepColsOf_congr _ _ (hco 1 y₁ hr₁), ← getD_empty,
      ← getD_empty, ← hcols] at hqv
    exact hqv
  exact Or.inl ⟨q, a₁, b₁, hcl' ha, hcl' hb, hab⟩

theorem plainClashQ_orU {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {spec : SessionSpec} {st : Stmt}
    {T : TabSpec spec.mPts} {hw : 2 * spec.mPts ≤ 65536} {I : Exec.Inputs} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8)}
    {Q₁ Q₂ : Model.OutCL 𝔽 Dig (List UInt8) → Prop} (Q : Model.OutCL 𝔽 Dig (List UInt8) → Prop)
    (h₁ : PlainClashQU P x ω st T hw I d opn Q₁) (h₂ : PlainClashQU P x ω st T hw I d opn Q₂)
    (hQ : ∀ out, Q out → Q₁ out ∨ Q₂ out) : PlainClashQU P x ω st T hw I d opn Q := by
  rintro C u r' ⟨a, ha, hq⟩ hacc
  rcases hQ _ hq with hq | hq
  · exact h₁ C u r' ⟨a, ha, hq⟩ hacc
  · exact h₂ C u r' ⟨a, ha, hq⟩ hacc

theorem plainClashQLU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)]) {d : Json}
    (hd : ins.read.draw = some d) (T : TabSpec spec.mPts) (sch : FlockSoundness.Accounting.Schedule)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Dig × Model.Salts (List UInt8))
    (hE : ∀ o : LiveOutJ 1 spec.mPts, AcceptsAtD ins.read.I (some d) (wrPD ins.read.I d o) →
      ∀ pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig,
        (∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith T o.pts r)
          (repTapeJ (plainDec T) (fun _ _ => true) (fun _ _ _ => []) o 0 r) (sel pend' r)) →
        LaterClashLU sch (tableOutL H512 enc512 leafPlain (capOfRoot o.rootB[0]!) pend' (opn o.proofs 0)) →
        LaterEntriesClash ins.read.I d (wrPD ins.read.I d o))
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQU P x ω st T (plain_mPts hs hin hset).1 ins.read.I d opn (LaterClashLU sch) := by
  intro C u r' hq hacc
  have hw := (plain_mPts hs hin hset).1
  have hm1 := (plain_mPts hs hin hset).2.1
  have hp2 := (plain_mPts hs hin hset).2.2
  obtain ⟨a, ⟨hV, pend, hpl, rfl⟩, hcl⟩ := hq
  simp only [List.ofFn_succ, List.ofFn_zero, List.getD_cons_zero] at hcl
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := acc_decide hacc
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hview := plainSimRun P ω hset h hw hm1 hp2 u r' hacc ins spec sts link c₁ mids f hact
  rw [hview] at hV hpl hcl
  obtain ⟨spec'', st', sess, sch', hsu, hdec, hf', k, hk, s₀, hs₀, ℓ, q, a₁, b₁, ha, hb, hab⟩ :=
    hE _ hV.1 (pend 0) (hpl 0) hcl
  have hacc' := run_acc_next sha512 P (V false) x ω _ hacc
  replace hacc' : ((next false x ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).msgs.map Prod.fst).length).map
        (Claims.coinSeq (plainΦ x spec st (u, r'))))).eval sha512).1 = .acc := hacc'
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc'
  have hpi := publicInputs_none_of_checks hc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hsu₀⟩ := setupOf_some (act_decide hact)
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  have hsetA : setupAtD ins.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu₀) hzk hpi hd
  obtain ⟨e₄, e₅⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hsu))
  subst spec'' e₅
  obtain ⟨hpr, hall⟩ := recordsLiveAtD_wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec #[st'] sess hsu hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs ⟨0, Nat.one_pos⟩).2 ⟨k, hk⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  obtain ⟨hvr, -⟩ := decision_runs hsu₀ hrun
  have hv' : verifyWithM (hashesIn ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).vLog.map
      Prod.fst)) spec (tabsOf #[st']) f.record f.proofs = .ok () := verdict_within hvr
  have hcols := rawLevelCols_plain (st := st') _ hf (by rw [hx]; exact act_decide hact) hlv hv' (by simp) hf'
    (HiddenExec.setupTables_pointsAny hsetA).1 hk ℓ
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (k : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[2 * 0 + k]!).toList.map (withSha spec) →
      ∀ j : ℕ, y.rounds[j]?.map (fun rd : Round => rd.coins) =
        (lv.streams.getD k #[])[j]?.map (fun rd : Round => rd.coins) := by
    intro k y hy j
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  have hcl' : ∀ {q : ℕ} {v : List 𝔽}, (q, v) ∈ plainLevelColsOf (shapeOf (Setup.ofCircuit st') sch'
        (2 * st'.regions.size)) y₀.rounds (wrPD ins.read.I d (viewOf 1 spec.mPts spec link c₁ f)).proofs[k]! ℓ →
      (q, v) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).seen k ℓ := by
    intro q v hqv
    rw [hpr', plainLevelColsOf_congr _ _ _ (hco k y₀ hr₀), ← getD_empty, ← hcols] at hqv
    exact hqv
  exact Or.inr (Or.inl (plain_levelClash hacc (hcl' ha) (hcl' hb) hab))

theorem plainSimClashHU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (regionsOf st)).m) (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) :
    PlainClashQU P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
      ins.read.I d (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (SelfClashL H512 enc512 leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) :=
  plainClashQ_orU _ (plainClashQ₀U hs hin hset h hd _ _ _
      (plain_clash₀HUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω)
    (plainClashQLU hs hin hset h hd _ _ _
      (plain_laterClashHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) P ω) fun _ hq => hq

end FlockDeploy.KS
