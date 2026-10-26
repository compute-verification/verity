import Proofs.Flock.Claims.KS.PlainDrawAccK

/-!
# flock-verify's acceptance at a draw value

`PlainDrawAccK`'s lemmas at `keep := id`, where `AcceptsAtDK id` is `AcceptsAtD` (`acceptsAtD_iff_acceptsAtDK_id`) and
the record decodes through `Record.decode` (`decodeWithK_id_std`):

* `accepts_rep_roundsAnyD`, `accepts_verdict_hm96AnyD`: `accepts_rep_roundsAny`, `accepts_verdict_hm96Any`, for either
  tag.
* `accepts_req0D`, `accepts_opens₀LD`: `accepts_req0`, `accepts_opens₀L`.
* `accepts_repsOneD`: `accepts_reps` and `accepts_repsH` at once, the one table's statement `setupOne`'s at `dr`.
* `tableAt_of_acceptsD`, `tableAtH_of_acceptsD`: `tableAt_of_accepts`, `tableAtH_of_acceptsI` at the draw `some d`,
  with no premise on a record's draw.
-/

namespace FlockSoundness.Discharge.AtDraw

open Claims.Bridge.PlainDraw (setupAtD AcceptsAtD acceptsAtD_iff_acceptsAtDK_id)
open FlockSoundness.Discharge.Exec (decodeWithK_id_std)

local notation "𝔽" => FlockLevel3.GF128

section Any

open Flock FlockLevel3 Refine Game Transcript Model Accounting
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **flock-verify's acceptance at `dr`, round by round** (`accepts_rep_roundsAny`). -/
theorem accepts_rep_roundsAnyD {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript}
    (hr : I.tags.retainRounds = true) (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 0 t.proofs[2 * j]! ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 1 t.proofs[2 * j + 1]! := by
  simpa only [decodeWithK_id_std] using accepts_rep_roundsAnyDK id hr ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

/-- **flock-verify's acceptance at `dr` makes each table's salted verdict accept** (`accepts_verdict_hm96Any`). -/
theorem accepts_verdict_hm96AnyD {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript}
    (hl : I.tags.merkleLeaf = "hm96-sha512/v1") (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧
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
          (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[j]!) (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true := by
  simpa only [decodeWithK_id_std] using accepts_verdict_hm96AnyDK id hl ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

end Any

section Req0

open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **An accepted rep at `dr` requests level 0 first** (`accepts_req0`). -/
theorem accepts_req0D {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (r : ℕ), r < 2 →
        ∀ (s : StreamRec) (p : Proof) (pend : PendL 𝔽 Dig),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r) =
            some s →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + r]! = .ok p →
          Plays fOf (Model.repC execArith r (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) pend →
          (∃ (R : Req) (rest : List Req), pend.reqs = R :: rest ∧ R.level = 0 ∧
            R.positions = (Flock.positions (sch.levels.getD 0 default)
              (s.rounds[Claims.Flock.plainQueryRound sts[j].m sts[j].c.kLog sch.k0
                (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length)]!).coins).toList ∧
            p.ligerito.initialProof.rows.size = R.positions.length) ∧
          ∀ R ∈ pend.reqs, R.lanes = (lvOf (schedOf sch) R.level).lanes ∧
            R.depth = Model.pathLen (lvOf (schedOf sch) R.level) := by
  simpa only [decodeWithK_id_std] using accepts_req0DK id ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

/-- **At flock-verify's acceptance at `dr`, each table's level-0 openings are its reps' deployed entries**
(`accepts_opens₀L`). -/
theorem accepts_opens₀LD {I : Exec.Inputs} {dr : Option Json} {t : Exec.Transcript} (h : AcceptsAtD I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
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
          out = tableOutL H512 enc512 leaf512 cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          out.accepted = true →
          (∀ x ∈ opens₀L (schedOf sch) out, VerifiedL H512 enc512 leaf512 cap
            (lvOf (schedOf sch) 0).lanes (Model.pathLen (lvOf (schedOf sch) 0)) x ∧
              x.1 < 2 ^ (sch.levels.getD 0 default).d) ∧
          (opens₀L (schedOf sch) out).map (fun x => (x.1, x.2.1)) =
            Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ++
              Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! := by
  simpa only [decodeWithK_id_std] using accepts_opens₀LDK id ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

end Req0

section Reps

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **flock-verify's acceptance at `dr` of a one-table session, rep by rep**: `accepts_reps` and `accepts_repsH`, with
the one table's statement `st` `setupOne`'s at `dr` under either tag. -/
theorem accepts_repsOneD {I : Inputs} {dr : Option Json} {t : Exec.Transcript} (hrr : I.tags.retainRounds = true)
    (hl : I.tags.merkleLeaf = "hm96-sha512/v1") (hc1 : I.count ≤ 1) (h : AcceptsAtD I dr t) :
    ∃ (st : Flock.Stmt) (spec : SessionSpec) (sess : Session) (sch : Flock.Schedule),
      setupOne I.tags I.circuitFile I.publicFile I.coins I.tables dr I.partition I.program = .ok st ∧
      setupAtD I dr = .ok (spec, #[st]) ∧ Record.decode spec t.record = .ok sess ∧
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
        (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[0]!) (pend 0, pend 1)
          (opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
            (ptsOf st.mPts sess.points)).length #[t.proofs[0]!, t.proofs[1]!])).accepted = true := by
  simpa only [decodeWithK_id_std] using accepts_repsOneDK id hrr hl hc1 ((acceptsAtD_iff_acceptsAtDK_id I dr t).1 h)

end Reps

section Tables

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.HiddenExec

/-- **At a draw flock-verify accepts at, the headline's table is `stmtOf` of the statement it accepted**:
`tableAt_of_accepts` at `AcceptsAtD I (some (dj S))`, with no premise on the record's draw. -/
theorem tableAt_of_acceptsD {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {c : Flock.Circuit}
    (ht : I.tags.typed = false) (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (hh : I.tags.hiddenOutputs = false) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {t : Exec.Transcript} (h : AcceptsAtD I (some (dj S)) t)
    (hS : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program = .ok st ∧
      ∃ hwf : StmtWF st, (tableAt ht dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  exact tableAt_of_acceptsDK id ht hc ho hh hc1 dj x₀ ((acceptsAtD_iff_acceptsAtDK_id I _ t).1 h) hS hmPts

/-- **At a draw flock-verify accepts at under hidden outputs, the hidden table is `stmtOf` of the statement it
accepted**: `tableAtH_of_acceptsI` at `AcceptsAtD I (some (dj S))`, with no premise on the record's draw. -/
theorem tableAtH_of_acceptsD {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {c : Flock.Circuit}
    (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {t : Exec.Transcript}
    (h : AcceptsAtD I (some (dj S)) t)
    (hS : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition
      I.program = .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
        .ok st ∧ ∃ hwf : StmtWF st, (tableAtH dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  exact tableAtH_of_acceptsDK id ht hc hh hc1 dj x₀ ((acceptsAtD_iff_acceptsAtDK_id I _ t).1 h) hS hmPts

end Tables

end FlockSoundness.Discharge.AtDraw
