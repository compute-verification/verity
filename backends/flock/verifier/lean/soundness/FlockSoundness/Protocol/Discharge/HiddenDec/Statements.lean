import FlockSoundness.Protocol.Discharge.DecodeZL.OP
import FlockSoundness.Protocol.Discharge.DecodeZL.LIG
import FlockSoundness.Protocol.Discharge.HiddenExec.Tables

/-!
# The decoder at the hidden-output headline's draws: statements

`flock_headline_exec_hidden`'s open guarantee is `hdec : DecodesOn …` at the hidden tables `tablesAtH`. #862's
`DecodesOnExec` (`Discharge/DecodeZL`) is the same guarantee at #828's tables, under `Stmt.setupH`; the decoder `decAt`,
the number of extra claims `nX`, `RecordsLive` and the chunk lemmas (`zl_carries`, `op_carries` at `wOP`, `lig_carries`
at `wLIG`) are generic in the table or the setup and are DecodeZL's. What changes is where the draw is set up:

* `TagsOkH`: DecodeZL's `TagsOk` with hidden outputs (`hiddenOutputs = true`).
* `DrawOkH`: DecodeZL's `DrawOk` with `Stmt.setupHidden` for `Stmt.setupH`.
* `opnAtH`: DecodeZL's `opnAt` at `Stmt.setupHidden`'s statement.
* `DecodesOnExecH`: `DecodesOn` at the hidden tables, the decoder `decAt` at DecodeZL's wires `wOP` and `wLIG`,
  `one := true`, no final messages, `capOfRoot`.
-/

namespace FlockSoundness.Discharge.HiddenDec

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.HiddenExec

variable {n mPts : ℕ} {Reg : Type}

/-- **The coin server's draw at `S`, under hidden outputs**: it proves `S`, and setupHidden sets up at it with `mPts`
link points. -/
def DrawOkH (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts

/-- **The openings at draw `S`, under hidden outputs**: `opnOf` at the statement setupHidden sets up at the record's
draw `dj S` and its `fast100` schedule, on the first two proof files. -/
noncomputable def opnAtH (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) : Opens 𝔽 Dig × Salts (List UInt8) :=
  match Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program with
  | .ok st => match Flock.fast100 st.m with
    | some sch => opnOf (Setup.ofCircuit st) sch (nX T S R) #[proofs[0]!, proofs[1]!]
    | none => default
  | .error _ => default

end FlockSoundness.Discharge.HiddenDec
