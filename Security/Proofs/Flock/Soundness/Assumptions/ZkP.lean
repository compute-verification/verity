import Proofs.Flock.Soundness.Assumptions.Zk
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJP

/-!
# The `--zk` session's custody at the verifier's public inputs

`RecordCustodyZKJP`: `RecordCustodyZKJ` at flock-verify `--zk --public-inputs`, whose acceptance is `AcceptsZKP I own`
and whose session sets up at `setupOfP (zkInputs I) own`. `RecordCustodyZKJ` is stated at `setupOf (zkInputs I)`, which
refuses a circuit with public input ports (`HmIn.checkOwn` at `none`), so a session at public inputs needs this form.
At `own = none` it is `RecordCustodyZKJ` (`recordCustodyZKJP_none`).
-/

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- **The record is the live session's, at `J` tables, at the verifier's public inputs** (`live-verifier`, as
`RecordCustodyZKJ`): the record `verify --zk --session-tables J --public-inputs` reads is the one the challenger wrote
for the live session (`RecordsLiveZKJA` at that command's acceptance and setup: its draw, proofs and link points, and
each table's root and rounds, are the live outcome's), and its draw file the one `flock-verify draw` printed for the
verifier's own draw under the subset sampling design (`DrawFileZK` at `subset kd` over the `n` proof blocks).
Provisional. -/
def RecordCustodyZKJP {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : Prop :=
  Discharge.Composed.RecordsLiveZKJA (Discharge.ZkHidden.AcceptsZKP I own)
      (Discharge.Exec.setupOfP (Discharge.ZkExec.zkInputs I) own) dj Jf wr ∧
    Discharge.ZkLink.DrawFileZK dj (.subset kd) n

/-- At no public inputs, `RecordCustodyZKJP` is `RecordCustodyZKJ`. -/
theorem recordCustodyZKJP_none {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : RecordCustodyZKJP I none dj Jf wr kd ↔ RecordCustodyZKJ I dj Jf wr kd :=
  Iff.rfl

end FlockSoundness.Assumptions.Zk
