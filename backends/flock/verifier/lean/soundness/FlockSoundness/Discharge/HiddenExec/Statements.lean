import FlockSoundness.Discharge.HiddenExec.Tables
import FlockSoundness.Protocol.Discharge.HiddenExec.Statements

/-!
# The statements: flock-verify's acceptance under either tag, hidden outputs included

e2e-exec's acceptance lemmas (`Discharge/Exec`) take `hh : I.tags.hiddenOutputs = false`: they are about `Stmt.setupH`'s
statements, and under hidden outputs `Stmt.setupTables` calls `Stmt.setupHidden`. Each statement here is one of them
without `hh`, so it covers the sessions main's prover writes since #757 (`verity/flock-circuit`, the default
`--statement`). Where the original names `setupH`, this one names `setupOne`, the setup `setupTables` calls for the
tag; where it doesn't, the conclusion is the original's.

* `SetupTablesOkAny`, `AcceptsSetupOne`, `AcceptsOneAny`: e2e-exec's `setupTables_ok`, `accepts_setupH`, `accepts_one`.
* `AcceptsRetainedAny`, `AcceptsAllRetainedAny`, `AcceptsRepRoundsAny`: `accepts_retained`, `accepts_allRetained`,
  `accepts_rep_rounds`.
* `AcceptsTablesHm96Any`, `AcceptsTablesHm96OfLeafAny`, `AcceptsVerdictHm96Any`: `accepts_tables_hm96`,
  `accepts_tables_hm96_of_leaf`, `accepts_verdict_hm96`.
* `TableAtHOfAccepts`: Integrate's `tableAt_of_accepts` for the hidden tables (`tableAtH`, `Tables.lean`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine Game Transcript Model Accounting
open FlockSoundness.Discharge.Exec
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **flock-verify's acceptance sets up each table with `setupOne`** (`Exec.accepts_setupH` without `hh`). -/
def AcceptsSetupOne : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, Accepts I t →
    ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
      Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        setupOne (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition I.program =
          .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

/-- **flock-verify's acceptance under retaining tags has every round's bytes** (`Exec.accepts_retained` without
`hh`). -/
def AcceptsRetainedAny : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, I.tags.retainRounds = true → Accepts I t →
    ∃ spec sts sess, setupOf I t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      spec.retainRounds = true ∧ ∀ s ∈ sess.streams.toList, ∀ r ∈ s.rounds.toList, ∃ b, r.bytes = some b

/-- **flock-verify's acceptance under retaining tags gives `AllRetained`** for every stream
(`Exec.accepts_allRetained` without `hh`). -/
def AcceptsAllRetainedAny : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, I.tags.retainRounds = true → Accepts I t →
    ∃ spec sts sess, setupOf I t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds

/-- **flock-verify refines each table of the session it accepts, under `hm96-sha512/v1`** (`Exec.accepts_tables_hm96`
without `hh`, with `setupOne` for `setupH`). -/
def AcceptsTablesHm96Any : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, Accepts I t →
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size),
        (∃ d, setupOne (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)) ∧
        ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧ 22 ≤ sts[j].m ∧
          ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m)
            (hms : sts[j].merkle = MerkleScheme.hm96Sha512),
          ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutCL 𝔽 Dig (List UInt8)),
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
            Plays fOf (tableAfterL execArith H512 enc512 leaf512 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
                PT_LOCAL sts[j].mPts (capDig p₀.cap) (ptsOf sts[j].mPts sess.points)) ms
              (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
            out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch ∧ out.salts = saltsOf p₀ p₁ sch

/-- **flock-verify refines each table of the session it accepts, for tags that implement `hm96-sha512/v1`**
(`Exec.accepts_tables_hm96_of_leaf` without `hh`, with `setupOne` for `setupH`). -/
def AcceptsTablesHm96OfLeafAny : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, I.tags.merkleLeaf = "hm96-sha512/v1" → Accepts I t →
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size),
        (∃ d, setupOne (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)) ∧
        sts[j].merkle = MerkleScheme.hm96Sha512 ∧
        ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧ 22 ≤ sts[j].m ∧
          ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m),
          ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutCL 𝔽 Dig (List UInt8)),
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
            decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
            sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
            Plays fOf (tableAfterL execArith H512 enc512 leaf512 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
                PT_LOCAL sts[j].mPts (capDig p₀.cap) (ptsOf sts[j].mPts sess.points)) ms
              (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
            out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch ∧ out.salts = saltsOf p₀ p₁ sch

end FlockSoundness.Discharge.HiddenExec
