module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
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
# The `--zk` session's record custody at a sampling design's samples

`live-verifier` as `flock_verify_sound`'s claims under `--zk` take it: the record is the live session's, and the sample
file at every sample of the live law proves the sampled proof blocks and sets the statement up with the outcome's link points.
`RecordCustodyZKLaw` at one public-output table, `RecordCustodyZKHLaw` at one hidden-output table, `RecordCustodyZKJLaw`
and `RecordCustodyZKHJLaw` at `J`. `RecordCustodyZK` and `RecordCustodyZKJ` (`Assumptions.Zk`) state the sample fact as
the sample file being `flock-verify draw`'s instead.
-/

@[expose] public section

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- **The record is the live session's, at the live law's samples** (`live-verifier`, with A6,
`uniform/flock-coin-server`, for the coins in it). Nothing in a session record authenticates its writer, and
`flock-verify verify --zk` reads the sample and the proofs only from the record, so no check of the record establishes
either fact:

* `RecordsLiveZK I dj wr`: the record `verify --zk` reads is the one the challenger wrote for the live session. At every
  transcript it accepts, its sample is `dj S`, its proofs are the prover's, and its root, link points and rounds are the
  live outcome's.
* `DrawOkZK I dj mPts` at every sample of the live law `L`: the record's sample file `dj S` proves every unit of `S`, and
  the statement sets up at it under the `--zk` tags with the outcome's `mPts` link points.

Both are stated at what flock-verify parses, so the record's serialization is outside them. -/
def RecordCustodyZKLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.ZkLink.RecordsLiveZK I dj wr ∧ ∀ ω, Discharge.ZkLink.DrawOkZK I dj mPts (L.draw ω)

/-- **The record is the live session's, at the live law's samples, under hidden outputs** (`live-verifier`, as
`RecordCustodyZKLaw`): the record is the live session's (`RecordsLiveZK`), and at every sample of `L` its sample file proves
the sampled proof blocks and the hidden-output statement sets up at it with `mPts` link points (`DrawOkZKH`). -/
def RecordCustodyZKHLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (L : Audit.Law n) : Prop :=
  Discharge.ZkLink.RecordsLiveZK I dj wr ∧ ∀ ω, Discharge.ZkHidden.DrawOkZKH I dj mPts (L.draw ω)

/-- **The record is the live session's, at the live law's samples, at `J` tables** (`live-verifier`, as
`RecordCustodyZKLaw`): the record `verify --zk --session-tables J` reads is the one the challenger wrote for the live
session (`RecordsLiveZKJ`: its sample, proofs and link points, and each table's root and rounds, are the live outcome's),
and at every sample of `L` each table's sample proves the units it is given and sets that table up with `mPts` link points
(`DrawOkZKJ`). -/
def RecordCustodyZKJLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (L : Audit.Law n) : Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ ∀ ω, Discharge.Composed.DrawOkZKJ I dj mPts (L.draw ω)

/-- **The record is the live session's, at the live law's samples, at `J` hidden-output tables** (`live-verifier`, as
`RecordCustodyZKJLaw`, with `DrawOkZKHJ`'s hidden-output setup for `DrawOkZKJ`'s). -/
def RecordCustodyZKHJLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (L : Audit.Law n) : Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ ∀ ω, Discharge.ZkHidden.DrawOkZKHJ I dj mPts (L.draw ω)

/-- **The record is the live session's, at `J` tables** (`live-verifier`'s record half, `RecordCustodyZKHJLaw`'s first
conjunct): V*'s sessions carry it apart from the sample half (`DrawCustodyZKHJ`), as `ZkOuter.hRec`. -/
def RecordsCustodyZKJ {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript) :
    Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr

/-- **The sample file proves the sampled proof blocks, at the live law's samples, at `J` hidden-output tables** (`live-verifier`'s
sample half, `RecordCustodyZKHJLaw`'s second conjunct): V*'s sessions carry it as `ZkOuterD.hDraw` and `ZkOuterR.hDraw`. -/
def DrawCustodyZKHJ {n : ℕ} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ)
    (L : Audit.Law n) : Prop :=
  ∀ ω, Discharge.ZkHidden.DrawOkZKHJ I dj mPts (L.draw ω)

/-- **The sample file proves the sampled proof blocks at one sample `S`, at `J` tables** (`live-verifier`'s sample half at one sample,
`Discharge.Composed.DrawOkZKJ`), as `setupZJ_of_acceptsZK` takes it. -/
def DrawAtZKJ := @Discharge.Composed.DrawOkZKJ

/-- **The sample file proves the sampled proof blocks at one sample `S`, at `J` hidden-output tables** (`DrawAtZKJ` with
`Discharge.ZkHidden.DrawOkZKHJ`'s hidden-output setup). -/
def DrawAtZKHJ := @Discharge.ZkHidden.DrawOkZKHJ

end FlockSoundness.Assumptions.Zk
