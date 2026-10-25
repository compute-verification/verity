import Proofs.Flock.Claims.KS.PlainClash0

/-!
# The live self-clash at level 0, at `wrPDK keep`

`PlainClash0`'s draw-layer lemmas at `AcceptsAtDK keep` and what `wrPDK keep` writes: `clash₀_wrK_of`,
`clash₀_wrKH_of`, `plain_clash₀AtK_of` and `plain_clash₀HAtK_of` at `wrPDK keep`, which records its outcome
(`recordsLiveAtDK_wrPDK`).

* `clash₀_wrPDK`, `clash₀_wrPDHK`: `clash₀_wrPD` and `clash₀_wrPDH` at `wrPDK keep`.
* `plain_clash₀AtK`, `plain_clash₀HAtK`: `plain_clash₀At` and `plain_clash₀HAt` there.

The run-level half (`rawCols_plain`, `plainClashQ₀`, `plainClashQ_or`) reads `P`'s run at the verifier, and is not
restated here.
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open DecodeZL Integrate in
/-- **A level-0 self-clash at the deployed plain decoders is a clash of the reps' entries**: `clash₀_wrK_of` at what
`wrPDK keep` writes for `o` (`clash₀_wrPD`). -/
theorem clash₀_wrPDK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leaf512 (capOfRoot o.rootB[0]!)
      (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch 0))
      (opens₀L (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch (tableOutL H512 enc512 leaf512
        (capOfRoot o.rootB[0]!) pend' (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs)))) :
    EntriesClashK keep I (dj S) (wrPDK keep I (dj S) o) :=
  clash₀_wrK_of keep ht hc ho dj x₀ hT hS R o (recordsLiveAtDK_wrPDK keep I (dj S) o) hacc pend' hp hcl

open DecodeZL HiddenExec HiddenDec in
/-- **A level-0 self-clash at the deployed plain decoders at hidden outputs is a clash of the reps' entries**:
`clash₀_wrKH_of` at what `wrPDK keep` writes for `o` (`clash₀_wrPDH`). -/
theorem clash₀_wrPDHK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : ClashL H512 enc512 leaf512 (capOfRoot o.rootB[0]!)
      (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).lanes
      (Model.pathLen (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0))
      (opens₀L (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leaf512
        (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)))) :
    EntriesClashK keep I (dj S) (wrPDK keep I (dj S) o) :=
  clash₀_wrKH_of keep ht hc dj x₀ hT hS R o (recordsLiveAtDK_wrPDK keep I (dj S) o) hacc pend' hp hcl

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw**: `plain_clash₀AtK_of` at what
`wrPDK keep` writes for `o` (`plain_clash₀At`). -/
theorem plain_clash₀AtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = false) {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀L ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o) :=
  plain_clash₀AtK_of keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o (recordsLiveAtDK_wrPDK keep _ d o)
    hacc pend' hp hcl

/-- **The deployed plain decoders' level-0 self-clash at `plainTab`, at the draw, at hidden outputs under hm96 leaves**:
`plain_clash₀HAtK_of` at what `wrPDK keep` writes for `o` (`plain_clash₀HAt`). -/
theorem plain_clash₀HAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true)
    (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : Clash₀L ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    EntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o) :=
  plain_clash₀HAtK_of keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o
    (recordsLiveAtDK_wrPDK keep _ d o) hacc pend' hp hcl

end FlockDeploy.KS
