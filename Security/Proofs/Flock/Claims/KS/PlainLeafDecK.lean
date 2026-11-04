import Proofs.Flock.Claims.KS.PlainDrawDecK
import Proofs.Flock.Claims.KS.PlainLive

/-!
# The deployed plain decoders at plain SHA-512 leaves, at `VK keep`

The scope admits hidden outputs under plain SHA-512 leaves (`flock-leaf/sha512-unsalted`) without `--zk`
(`ProvedScope.leafOk`). Their model leaf is `Refine.leafPlain` (`Refine.leafOfId`), not the salted `leaf512` that
`PlainDrawAccK` and `PlainDrawDecK` read, and the decoders there are right at `leafPlain` by the same proofs, through
`keep` and at any record writer, with `table_verdict_plain` and `setupHidden_plain` for `table_verdict_hm96` and
`setupHidden_hm96`. At `keep := id` and `wrPD` they are `PlainLeafDec`'s:

* `accepts_verdict_plainAnyDK`, `accepts_opens₀LDUK`, `accepts_repsOneDUK`: `accepts_verdict_hm96AnyDK`,
  `accepts_opens₀LDK`, `accepts_repsOneDK` at `leafPlain`.
* `TagsOkHU`, `decodesTable_wrK_scHU_of`: `TagsOkH` and `decodesTable_wrK_scH_of` under plain SHA-512 leaves;
  `decodesTable_wrPDK_scHU` at `wrPDK`.
* `plain_setupHiddenU`, `plain_hdecHUAtK_of`: `plain_setupHidden` and `plain_hdecHAtK_of` there; `plain_hdecHUAtK` at
  `wrPDK`, with `plainVerdictAtK keep`.
-/

namespace FlockSoundness.Discharge.HiddenDec

open FlockSoundness.Discharge.Exec (Inputs)

/-- **The verify command's tags under hidden outputs and plain SHA-512 leaves, without `--zk`**: `TagsOkH` with
`flock-leaf/sha512-unsalted` for `hm96-sha512/v1`. -/
def TagsOkHU (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "flock-leaf/sha512-unsalted" ∧
    I.count ≤ 1

end FlockSoundness.Discharge.HiddenDec

namespace FlockSoundness.Discharge.AtDraw

open Claims.Bridge.PlainDraw (setupAtD AcceptsAtDK named_of_setupAtD)

local notation "𝔽" => FlockLevel3.GF128

section Any

open Flock FlockLevel3 Refine Game Transcript Model Accounting
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **flock-verify's acceptance at `dr` through `keep` under hidden outputs and plain SHA-512 leaves makes each table's
verdict accept at the plain leaf** (`accepts_verdict_plainAnyD`). -/
theorem accepts_verdict_plainAnyDK (keep : Bool → Bool) {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript}
    (hh : I.tags.hiddenOutputs = true) (hl : I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (h : AcceptsAtDK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeWithK keep With.std spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m),
        ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ ∧
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ ∧
          (tableOutL H512 enc512 leafPlain (capOfRoot sess.rootB[j]!) (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verifyK_ok_tables keep (named_of_setupAtD hs) hv
  obtain ⟨-, hsetup⟩ := setupTables_setupOne hs
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hs
  have hpts := decodeWithK_points keep With.std hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨d, hset, -⟩ := hsetup j hj
  refine ⟨schs[j], hf, fun hwf hr hm => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  have hms : sts[j].merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases setupOne_tagsAt_leaf (Or.inr ⟨hh, hl⟩) hset with ⟨e, -⟩ | ⟨-, e⟩
    · exact absurd (hl.symm.trans e) (by decide)
    · exact e
  exact table_verdict_plain spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) hms _ sess hext (decodeWithK_roundsWF keep With.std hsess) schs[j] hf j _ _
    c h0 h1

/-- **Plain SHA-512 leaves as a leaf form**: `flock-leaf/sha512-unsalted`, `leafPlain`, with
`accepts_verdict_plainAnyDK`. -/
noncomputable abbrev LeafForm.plain : LeafForm where
  tag := "flock-leaf/sha512-unsalted"
  lf := leafPlain
  verdicts := by
    intro keep _ _ _ hh hl h
    exact accepts_verdict_plainAnyDK keep hh hl h
  digestLen := by
    intro _ _ _ _ _ _ _ _ _ hst hl
    rw [HiddenExec.setupHidden_plain hl hst]
    rfl

end Any

section Req0

open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **At flock-verify's acceptance at `dr` through `keep`, each table's level-0 openings at the plain leaf are its
reps' deployed entries** (`accepts_opens₀LDU`): `accepts_opens₀LDLfK` at `leafPlain`. -/
theorem accepts_opens₀LDUK (keep : Bool → Bool)
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
          out.accepted = true →
          (∀ x ∈ opens₀L (schedOf sch) out, VerifiedL H512 enc512 leafPlain cap
            (lvOf (schedOf sch) 0).lanes (Model.pathLen (lvOf (schedOf sch) 0)) x ∧
              x.1 < 2 ^ (sch.levels.getD 0 default).d) ∧
          (opens₀L (schedOf sch) out).map (fun x => (x.1, x.2.1)) =
            Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ++
              Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! :=
  accepts_opens₀LDLfK keep leafPlain h

end Req0

section Reps

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **flock-verify's acceptance at `dr` through `keep` of a one-table session under hidden outputs and plain SHA-512
leaves, rep by rep** (`accepts_repsOneDU`): `accepts_repsOneDLfK` at `leafPlain`. -/
theorem accepts_repsOneDUK (keep : Bool → Bool)
    {I : Inputs} {dr : Option Json} {t : Exec.Transcript} (hrr : I.tags.retainRounds = true)
    (hh : I.tags.hiddenOutputs = true) (hl : I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") (hc1 : I.count ≤ 1)
    (h : AcceptsAtDK keep I dr t) :
    ∃ (st : Flock.Stmt) (spec : SessionSpec) (sess : RunRecord) (sch : Flock.Schedule),
      setupOne I.tags I.circuitFile I.publicFile I.coins I.tables dr I.partition I.program = .ok st ∧
      setupAtD I dr = .ok (spec, #[st]) ∧ Record.decodeWithK keep With.std spec t.record = .ok sess ∧
      Flock.fast100 st.m = some sch ∧
      ∀ (hwf : StmtWF st) (_hr : RegionsWF st) (hm : 13 ≤ st.m), ∃ pend : Fin 2 → PendL 𝔽 Dig,
        (∀ r : Fin 2, ∃ (s : StreamRec) (p : Proof) (Tv : F128),
          sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
            some s ∧
          decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
            st.mPts (ptsOf st.mPts sess.points)).length) t.proofs[r.val]! = .ok p ∧
          Plays fOf (Model.repC execArith r.val (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
              (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
            (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) (pend r) ∧
          (chunks (repFrames (Setup.ofCircuit st) sch (spec.domain spec.tables[0]! r.val).toUTF8
            (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
              (ptsOf st.mPts sess.points)).length) p Tv)).1.length = s.rounds.size ∧
          ∀ k < s.rounds.size, Retains (s.rounds[k]!) (encs ((chunks (repFrames (Setup.ofCircuit st) sch
            (spec.domain spec.tables[0]! r.val).toUTF8 (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p Tv)).1.getD k []))) ∧
        (tableOutL H512 enc512 leafPlain (capOfRoot sess.rootB[0]!) (pend 0, pend 1)
          (opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
            (ptsOf st.mPts sess.points)).length #[t.proofs[0]!, t.proofs[1]!])).accepted = true :=
  accepts_repsOneDLfK keep leafPlain hrr hc1 (accepts_verdict_plainAnyDK keep hh hl h) h

end Reps

end FlockSoundness.Discharge.AtDraw

namespace Claims.Bridge.PlainDraw

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain FlockSoundness.Discharge.AtDraw
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at a record writer at hidden outputs under plain SHA-512 leaves**:
`decodesTable_wrK_scHLf_of` at `LeafForm.plain`. -/
theorem decodesTable_wrK_scHU_of (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (wr : Tables.LiveOutJ 1 mPts → Exec.Transcript)
    (hwr : ∀ o, RecordsLiveAtDK keep I (dj S) o (wr o)) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => AcceptsAtDK keep I (some (dj S)) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep I (dj S) C (wr o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j :=
  decodesTable_wrK_scHLf_of keep LeafForm.plain ht hc dj x₀ hT hS R hN C rootB₀ wr hwr j

open DecodeZL HiddenExec HiddenDec in
/-- **The deployed plain decoders are right at `wrPDK` at hidden outputs under plain SHA-512 leaves**
(`decodesTable_wrK_scHU_of` at `wrPDK`). -/
theorem decodesTable_wrPDK_scHU (keep : Bool → Bool)
    {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : TagsOkHU I) {S : Finset (Fin n)} (hS : DrawOkH I dj mPts S) (R : Reg) {N : ℕ}
    (hN : N = 2 ^ (lvOf (tablesAtH (Reg := Reg) dj x₀ S R).sch 0).logLen) (C : Fin N → List 𝔽)
    (rootB₀ : Array ByteArray) (j : Fin [tablesAtH (Reg := Reg) dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch
        (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL H512 enc512 leafPlain (tablesAtH (Reg := Reg) dj x₀ S R).sch o))
      (fun o => AcceptsAtDK keep I (some (dj S)) (wrPDK keep I (dj S) o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep I (dj S) C (wrPDK keep I (dj S) o))
      (fun ps _ => opnAtH (tablesAtH dj x₀) I dj S R ps)
      (fun _ => decAt (tablesAtH dj x₀) wOP wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([tablesAtH (Reg := Reg) dj x₀ S R].get j)) j :=
  decodesTable_wrK_scHU_of keep ht hc dj x₀ hT hS R hN C rootB₀ (wrPDK keep I (dj S))
    (recordsLiveAtDK_wrPDK keep I (dj S)) j

end Claims.Bridge.PlainDraw

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **A plain statement in scope with hidden outputs and plain SHA-512 leaves sets up its one table at its draw**:
`plain_setupHiddenT` at `flock-leaf/sha512-unsalted`. -/
theorem plain_setupHiddenU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted") {d : Json}
    (hd : ins.read.draw = some d) :
    ins.read.I.tags.typed = false ∧ HmOut.parse ins.read.I.tags ins.read.I.circuitFile ins.read.I.tables = .ok st.c ∧
      HiddenDec.TagsOkHU ins.read.I ∧
      Stmt.setupHidden ins.read.I.tags ins.read.I.circuitFile ins.read.I.publicFile ins.read.I.coins
        ins.read.I.tables (some d) ins.read.I.partition ins.read.I.program = .ok st ∧ spec = st.spec ∧ 22 ≤ st.m ∧
      st.m ≤ 35 :=
  plain_setupHiddenT hs hin hset h hh hl hd

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at hidden outputs and plain
SHA-512 leaves, at a record writer**: `plain_hdecHLfAtK_of` at `LeafForm.plain`. -/
theorem plain_hdecHUAtK_of (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray)
    (wr : Tables.LiveOutJ 1 spec.mPts → Exec.Transcript) (hwr : ∀ o, RecordsLiveAtDK keep ins.read.I d o (wr o)) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (fun o => AcceptsAtDK keep ins.read.I (some d) (wr o) ∧ o.rootB[0]! = rootB₀[0]! ∧
        TxNoOffAtK keep ins.read.I d C (wr o))
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j :=
  plain_hdecHLfAtK_of keep AtDraw.LeafForm.plain hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay C rootB₀ wr
    hwr

open Classical Discharge.Tables Claims.Bridge.PlainDraw in
/-- **The deployed plain decoders are right at `plainTab` for the verdict at the draw, at hidden outputs and plain
SHA-512 leaves**: `plain_hdecHUAt` with `plainVerdictAtK keep` for `plainVerdictAt` (`plain_hdecHUAtK_of` at
`wrPDK`). -/
theorem plain_hdecHUAtK (keep : Bool → Bool)
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st)
    (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) :
    ∀ j, DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
      (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
      (plainVerdictAtK keep ins.read.I d C rootB₀)
      (fun ps _ => HiddenDec.opnAtH (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I (fun _ => d) ∅ () ps)
      (fun _ => DecodeZL.decAt (fun (_ : Finset (Fin 0)) (_ : Unit) =>
        plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) DecodeZL.wOP DecodeZL.wLIG ∅ ())
      (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
      j :=
  plain_hdecHUAtK_of keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay C rootB₀ (wrPDK keep ins.read.I d)
    (recordsLiveAtDK_wrPDK keep ins.read.I d)

end FlockDeploy.KS
