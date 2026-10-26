import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

/-!
# The `--zk` session's record custody at a law's draws

`live-verifier` as `flock_verify_sound`'s claims under `--zk` take it: the record is the live session's, and the draw
file at every draw of the live law proves the drawn units and sets the statement up with the outcome's link points.
`RecordCustodyZKLaw` at one public-output table, `RecordCustodyZKHLaw` at one hidden-output table, `RecordCustodyZKJLaw`
and `RecordCustodyZKHJLaw` at `J`. `RecordCustodyZK` and `RecordCustodyZKJ` (`Assumptions.Zk`) state the draw fact as
the draw file being `flock-verify draw`'s instead.
-/

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- **The record is the live session's, at the live law's draws** (`live-verifier`, with A6,
`uniform/flock-coin-server`, for the coins in it). Nothing in a session record authenticates its writer, and
`flock-verify verify --zk` reads the draw and the proofs only from the record, so no check of the record establishes
either fact:

* `RecordsLiveZK I dj wr`: the record `verify --zk` reads is the one the challenger wrote for the live session. At every
  transcript it accepts, its draw is `dj S`, its proofs are the prover's, and its root, link points and rounds are the
  live outcome's.
* `DrawOkZK I dj mPts` at every draw of the live law `L`: the record's draw file `dj S` proves every unit of `S`, and
  the statement sets up at it under the `--zk` tags with the outcome's `mPts` link points.

Both are stated at what flock-verify parses, so the record's serialization is outside them. -/
def RecordCustodyZKLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.ZkLink.RecordsLiveZK I dj wr ∧ ∀ ω, Discharge.ZkLink.DrawOkZK I dj mPts (L.draw ω)

/-- **The record is the live session's, at the live law's draws, under hidden outputs** (`live-verifier`, as
`RecordCustodyZKLaw`): the record is the live session's (`RecordsLiveZK`), and at every draw of `L` its draw file proves
the drawn units and the hidden-output statement sets up at it with `mPts` link points (`DrawOkZKH`). -/
def RecordCustodyZKHLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.ZkLink.RecordsLiveZK I dj wr ∧ ∀ ω, Discharge.ZkHidden.DrawOkZKH I dj mPts (L.draw ω)

/-- **The record is the live session's, at the live law's draws, at `J` tables** (`live-verifier`, as
`RecordCustodyZKLaw`): the record `verify --zk --session-tables J` reads is the one the challenger wrote for the live
session (`RecordsLiveZKJ`: its draw, proofs and link points, and each table's root and rounds, are the live outcome's),
and at every draw of `L` each table's draw proves the units it is given and sets that table up with `mPts` link points
(`DrawOkZKJ`). -/
def RecordCustodyZKJLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (L : Audit.Law n) : Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ ∀ ω, Discharge.Composed.DrawOkZKJ I dj mPts (L.draw ω)

/-- **The record is the live session's, at the live law's draws, at `J` hidden-output tables** (`live-verifier`, as
`RecordCustodyZKJLaw`, with `DrawOkZKHJ`'s hidden-output setup for `DrawOkZKJ`'s). -/
def RecordCustodyZKHJLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (L : Audit.Law n) : Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ ∀ ω, Discharge.ZkHidden.DrawOkZKHJ I dj mPts (L.draw ω)

/-- **The record is the live session's, at `J` tables** (`live-verifier`'s record half, `RecordCustodyZKHJLaw`'s first
conjunct): V*'s sessions carry it apart from the draw half (`DrawCustodyZKHJ`), as `ZkOuter.hRec`. -/
def RecordsCustodyZKJ {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript) :
    Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr

/-- **The draw file proves the drawn units, at the live law's draws, at `J` hidden-output tables** (`live-verifier`'s
draw half, `RecordCustodyZKHJLaw`'s second conjunct): V*'s sessions carry it as `ZkOuterD.hDraw` and `ZkOuterR.hDraw`. -/
def DrawCustodyZKHJ {n : ℕ} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ)
    (L : Audit.Law n) : Prop :=
  ∀ ω, Discharge.ZkHidden.DrawOkZKHJ I dj mPts (L.draw ω)

/-- **The draw file proves the drawn units at one draw `S`, at `J` tables** (`live-verifier`'s draw half at one draw,
`Discharge.Composed.DrawOkZKJ`), as `setupZJ_of_acceptsZK` takes it. -/
def DrawAtZKJ := @Discharge.Composed.DrawOkZKJ

/-- **The draw file proves the drawn units at one draw `S`, at `J` hidden-output tables** (`DrawAtZKJ` with
`Discharge.ZkHidden.DrawOkZKHJ`'s hidden-output setup). -/
def DrawAtZKHJ := @Discharge.ZkHidden.DrawOkZKHJ

end FlockSoundness.Assumptions.Zk
