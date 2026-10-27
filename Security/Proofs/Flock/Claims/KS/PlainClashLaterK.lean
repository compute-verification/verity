import Proofs.Flock.Claims.KS.PlainClashLater

/-!
# The live self-clash past level 0, at `wrPDK keep`

`PlainClashLater`'s draw-layer lemmas at `AcceptsAtDK keep` and what `wrPDK keep` writes: `laterEntries_wrK_of`,
`laterEntries_wrKH_of`, `plain_laterEntriesAtK_of` and `plain_laterEntriesHAtK_of` at `wrPDK keep`, which records its
outcome (`recordsLiveAtDK_wrPDK`), and the clashes they give by `laterEntriesClashK_of`.

* `laterEntries_wrPDK`, `laterClash_wrPDK`, `laterEntries_wrPDHK`, `laterClash_wrPDHK`: `laterEntries_wrPD`,
  `laterClash_wrPD`, `laterEntries_wrPDH`, `laterClash_wrPDH` at `wrPDK keep`.
* `plain_laterEntriesAtK`, `plain_laterClashAtK`, `plain_laterEntriesHAtK`, `plain_laterClashHAtK`: those at the
  statement's draw, at `plainTab`.

The run-level half (`rawLevelCols_plain`, `plainClashQL`, `plainSimLaterClashAt`) reads `P`'s run at the verifier, and
is not restated here.
-/

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open DecodeZL Integrate in
/-- **At the deployed plain decoders, the reps' recorded caps and later openings are the deployed reps'**:
`laterEntries_wrK_of` at what `wrPDK keep` writes for `o` (`laterEntries_wrPD`). -/
theorem laterEntries_wrPDK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch I (dj S) (wrPDK keep I (dj S) o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs)) r :=
  laterEntries_wrK_of keep ht hc ho dj x₀ hT hS R o (recordsLiveAtDK_wrPDK keep I (dj S) o) hacc pend' hp

open DecodeZL Integrate in
/-- **A self-clash past level 0 at the deployed plain decoders is a clash of one rep's level-`ℓ` entries**, at
`wrPDK keep` (`laterClash_wrPD`). -/
theorem laterClash_wrPDK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashL (Integrate.tablesAt (Reg := Reg) ht dj x₀ S R).sch (tableOutL H512 enc512 leaf512
      (capOfRoot o.rootB[0]!) pend' (DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R o.proofs))) :
    LaterEntriesClashK keep I (dj S) (wrPDK keep I (dj S) o) :=
  laterEntriesClashK_of keep (laterEntries_wrPDK keep ht hc ho dj x₀ hT hS R o hacc pend' hp) hcl

open DecodeZL HiddenExec HiddenDec in
/-- **At the deployed plain decoders at hidden outputs, the reps' recorded caps and later openings are the deployed
reps'**: `laterEntries_wrKH_of` at what `wrPDK keep` writes for `o` (`laterEntries_wrPDH`). -/
theorem laterEntries_wrPDHK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep (tablesAtH (Reg := Reg) dj x₀ S R).sch I (dj S) (wrPDK keep I (dj S) o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs)) r :=
  laterEntries_wrKH_of keep ht hc dj x₀ hT hS R o (recordsLiveAtDK_wrPDK keep I (dj S) o) hacc pend' hp

open DecodeZL HiddenExec HiddenDec in
/-- **A self-clash past level 0 at the deployed plain decoders at hidden outputs is a clash of one rep's level-`ℓ`
entries**, at `wrPDK keep` (`laterClash_wrPDH`). -/
theorem laterClash_wrPDHK (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkH I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg)
    (o : Tables.LiveOutJ 1 mPts) (hacc : AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (tablesAtH (Reg := Reg) dj x₀ S R) o.pts r)
      (Tables.repTapeJ (fun _ => DecodeZL.decAt (tablesAtH dj x₀) DecodeZL.wOP DecodeZL.wLIG S R)
        (fun _ _ => true) (fun _ _ _ => []) o 0 r) (Tables.sel pend' r))
    (hcl : FlockDeploy.KS.LaterClashL (tablesAtH (Reg := Reg) dj x₀ S R).sch (tableOutL H512 enc512 leaf512
      (capOfRoot o.rootB[0]!) pend' (opnAtH (tablesAtH dj x₀) I dj S R o.proofs))) :
    LaterEntriesClashK keep I (dj S) (wrPDK keep I (dj S) o) :=
  laterEntriesClashK_of keep (laterEntries_wrPDHK keep ht hc dj x₀ hT hS R o hacc pend' hp) hcl

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at `wrPDK keep`**:
`plain_laterEntriesAtK_of` at what `wrPDK keep` writes for `o` (`plain_laterEntriesAt`). -/
theorem plain_laterEntriesAtK (keep : Bool → Bool)
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
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPDK keep ins.read.I d o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r :=
  plain_laterEntriesAtK_of keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o
    (recordsLiveAtDK_wrPDK keep _ d o) hacc pend' hp

/-- **The deployed plain decoders' self-clash past level 0 at `plainTab`, at the draw, at `wrPDK keep`**:
`laterEntriesClashK_of` at `plain_laterEntriesAtK`. -/
theorem plain_laterClashAtK (keep : Bool → Bool)
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
    (hcl : LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o) :=
  laterEntriesClashK_of keep
    (plain_laterEntriesAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp) hcl

/-- **The deployed plain decoders' later entries at `plainTab`, at the draw, at hidden outputs under hm96 leaves, at
`wrPDK keep`**: `plain_laterEntriesHAtK_of` at what `wrPDK keep` writes for `o` (`plain_laterEntriesHAt`). -/
theorem plain_laterEntriesHAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r)) :
    ∀ r < 2, LaterEntriesAtK keep ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ins.read.I d (wrPDK keep ins.read.I d o)
      (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0)) r :=
  plain_laterEntriesHAtK_of keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o
    (recordsLiveAtDK_wrPDK keep _ d o) hacc pend' hp

/-- **The deployed plain decoders' self-clash past level 0 at `plainTab`, at the draw, at hidden outputs under hm96
leaves, at `wrPDK keep`**: `laterEntriesClashK_of` at `plain_laterEntriesHAtK`. -/
theorem plain_laterClashHAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1") {d : Json}
    (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ (L.model (regionsOf st)).m)
    (hlay : (L.model (regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (o : LiveOutJ 1 spec.mPts)
    (hacc : AcceptsAtDK keep ins.read.I (some d) (wrPDK keep ins.read.I d o))
    (pend' : Model.PendL 𝔽 Dig × Model.PendL 𝔽 Dig)
    (hp : ∀ r : Fin 2, PlaysT (repsOf (D := Dig) execArith (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)
      o.pts r) (repTapeJ (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
        (fun _ _ _ => []) o 0 r) (sel pend' r))
    (hcl : LaterClashL ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tableOutL H512 enc512 leaf512 (capOfRoot o.rootB[0]!) pend'
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d o.proofs 0))) :
    LaterEntriesClashK keep ins.read.I d (wrPDK keep ins.read.I d o) :=
  laterEntriesClashK_of keep
    (plain_laterEntriesHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay o hacc pend' hp) hcl

end FlockDeploy.KS
