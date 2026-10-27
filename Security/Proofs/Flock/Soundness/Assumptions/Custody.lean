import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements
import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements

/-!
# The session's record custody without `--zk`

`live-verifier` (the custody of the verifier's `PROTOCOL.md` §17.2) as `flock-verify verify` without `--zk` takes it:
`RecordCustodyExec` at public outputs, `RecordCustodyExecH` at hidden ones. `--zk`'s are `Assumptions.Zk.RecordCustodyZK`
and `RecordCustodyZKJ`.
-/

namespace FlockSoundness.Assumptions

open FlockSoundness

/-- **The record is the live session's, without `--zk`** (`live-verifier`, with A6, `uniform/flock-coin-server`, for the
coins in it). Nothing in a session record authenticates its writer, and `flock-verify verify` reads the draw and the
proofs only from the record, so no check of the record establishes either fact:

* `RecordsLive I dj wr`: the record `verify` reads is the one the challenger wrote for the live session. At every
  transcript it accepts, its draw is `dj S`, its proofs are the prover's, and its root, link points and rounds are the
  live outcome's.
* `DrawOk I dj mPts` at every draw of the live law `L`: the record's draw file `dj S` proves every unit of `S`, and the
  statement sets up at it with the outcome's `mPts` link points.

Both are stated at what flock-verify parses, so the record's serialization is outside them. -/
def RecordCustodyExec {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.DecodeZL.RecordsLive I dj wr ∧ ∀ ω, Discharge.DecodeZL.DrawOk I dj mPts (L.draw ω)

/-- **The record is the live session's, without `--zk`, under hidden outputs** (`live-verifier`, as
`RecordCustodyExec`): the record is the live session's (`RecordsLive`), and at every draw of `L` its draw file proves
the drawn units and the hidden-output statement sets up at it with `mPts` link points (`DrawOkH`). -/
def RecordCustodyExecH {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.DecodeZL.RecordsLive I dj wr ∧ ∀ ω, Discharge.HiddenDec.DrawOkH I dj mPts (L.draw ω)

end FlockSoundness.Assumptions
