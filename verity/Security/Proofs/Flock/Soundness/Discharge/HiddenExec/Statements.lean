import Proofs.Flock.Soundness.Discharge.HiddenExec.Tables

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

/-- **`setupTables` sets up each table with `setupOne`**, under any tag (`Exec.setupTables_ok` without `hh`): for one
table, at the inputs and the session's draw, and the session's spec is the table's; for `count > 1`, `count` tables,
table `j` under `Tags.tableIdentity count j` at a subset draw of its part. -/
def SetupTablesOkAny : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
    {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt},
    Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure = .ok (spec, sts) →
    (count ≤ 1 ∧ ∃ st, setupOne tags cf pf coins tables draw partition program = .ok st ∧ spec = st.spec ∧
        sts = #[st]) ∨
      (1 < count ∧ sts.size = count ∧ (∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        setupOne { tags with identity := tags.tableIdentity count j } cf pf coins tables (some d) partition
          program = .ok sts[j]) ∧ (∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts) ∧ spec.points = 2)

/-- **flock-verify's acceptance sets up each table with `setupOne`** (`Exec.accepts_setupH` without `hh`). -/
def AcceptsSetupOne : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, Accepts I t →
    ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
      Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        setupOne (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition I.program =
          .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

/-- **A one-table acceptance is one `setupOne` statement's** (`Exec.accepts_one` without `hh`). -/
def AcceptsOneAny : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, I.count ≤ 1 → Accepts I t →
    ∃ st, setupOne I.tags I.circuitFile I.publicFile I.coins I.tables (recordDraw t.record) I.partition I.program =
        .ok st ∧ Flock.verify st.spec #[Setup.ofCircuit st] t.record t.proofs = .ok ()

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

/-- **flock-verify's acceptance, round by round** (`Exec.accepts_rep_rounds` without `hh`). -/
def AcceptsRepRoundsAny : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, I.tags.retainRounds = true → Accepts I t →
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 0 t.proofs[2 * j]! ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 1 t.proofs[2 * j + 1]!

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

/-- **flock-verify's acceptance makes each table's salted verdict accept** (`Exec.accepts_verdict_hm96` without
`hh`). -/
def AcceptsVerdictHm96Any : Prop :=
  ∀ {I : Exec.Inputs} {t : Exec.Transcript}, I.tags.merkleLeaf = "hm96-sha512/v1" → Accepts I t →
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupOf I t.record = .ok (spec, sts) ∧
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
            true

/-- **At a draw flock-verify accepts under hidden outputs, the headline's table is `stmtOf` of the statement it
accepted** (Integrate's `tableAt_of_accepts` for `tableAtH`): for a one-table session under a tag with hidden outputs
whose circuit file parses (untyped) to a circuit whose unit returns the outputs, when the session's record carries the
challenger's draw `dj S`, that draw proves `S`, and setupHidden's statement at it has link points `mPts`. -/
def TableAtHOfAccepts : Prop :=
  ∀ {I : Exec.Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {c : Flock.Circuit}, I.tags.typed = false →
    HmOut.parse I.tags I.circuitFile I.tables = .ok c → c.outNet = c.unitNet →
    I.tags.hiddenOutputs = true → I.count ≤ 1 →
    ∀ (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {t : Exec.Transcript},
      Accepts I t → recordDraw t.record = some (dj S) →
      (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) →
      (∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition
        I.program = .ok st → st.mPts = mPts) →
      ∃ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
          .ok st ∧ ∃ hwf : StmtWF st, (tableAtH dj x₀ S).T.S = stmtOf st hwf (regionsOf st)

end FlockSoundness.Discharge.HiddenExec
