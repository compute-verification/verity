module

public import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# The session's record custody without `--zk`

`live-verifier` (the custody of `Security/Proofs/Flock/Soundness/ASSUMPTIONS.md`) as `flock-verify verify` without `--zk` takes it:
`RecordCustodyExec` at public outputs, `RecordCustodyExecH` at hidden ones. `--zk`'s are `Assumptions.Zk.RecordCustodyZK`
and `RecordCustodyZKJ`.
-/

@[expose] public section

namespace FlockSoundness.Assumptions

open FlockSoundness

/-- **The record is the live session's, without `--zk`** (`live-verifier`, with A6, `uniform/flock-coin-server`, for the
coins in it). Nothing in a session record authenticates its writer, and `flock-verify verify` reads the sample and the
proofs only from the record, so no check of the record establishes either fact:

* `RecordsLive I dj wr`: the record `verify` reads is the one the challenger wrote for the live session. At every
  transcript it accepts, its sample is `dj S`, its proofs are the prover's, and its root, link points and rounds are the
  live outcome's.
* `DrawOk I dj mPts` at every sample of the live law `L`: the record's sample file `dj S` proves every unit of `S`, and the
  statement sets up at it with the outcome's `mPts` link points.

Both are stated at what flock-verify parses, so the record's serialization is outside them. -/
def RecordCustodyExec {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.DecodeZL.RecordsLive I dj wr ∧ ∀ ω, Discharge.DecodeZL.DrawOk I dj mPts (L.draw ω)

/-- **The record is the live session's, without `--zk`, under hidden outputs** (`live-verifier`, as
`RecordCustodyExec`): the record is the live session's (`RecordsLive`), and at every sample of `L` its sample file proves
the sampled proof blocks and the hidden-output statement sets up at it with `mPts` link points (`DrawOkH`). -/
def RecordCustodyExecH {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.DecodeZL.RecordsLive I dj wr ∧ ∀ ω, Discharge.HiddenDec.DrawOkH I dj mPts (L.draw ω)

end FlockSoundness.Assumptions
