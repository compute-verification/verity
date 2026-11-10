module

public import Proofs.Flock.Soundness.Assumptions.ZkP

/-!
# The `--zk` session's premises that are instances of its named ones

`shape.premises` in `Security/lean-audit.json` names this module, and none of it sits in an `Assumptions` module, so the
audit reads each abbrev here as the premise it unfolds to.

* `RecordCustodyZKJ` (`live-verifier`): `Assumptions.Zk.RecordCustodyZKJPLaw` at no public inputs, every sample under the
  subset sampling design. `recordCustodyZKJ_iff` states it as the record's half at `setupOf (zkInputs I)` and the sample
  file's at `subset kd`.
-/

@[expose] public section

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- **The record is the live session's, at `J` tables** (`live-verifier`, as `RecordCustodyZK`): the record
`verify --zk --session-tables J` reads is the one the challenger wrote for the live session
(`RecordsLiveZKJ`: its sample, proofs and link points, and each table's root and rounds, are the live outcome's), and its
sample file the one `flock-verify draw` printed for the verifier's own sample under the subset sampling design (`DrawFileZK` at
`subset kd` over the `n` units; `RecordCustodyZKJS` is its stratified form): `RecordCustodyZKJPLaw` at no public inputs
and `subset kd` at every sample (`recordCustodyZKJ_iff`). -/
abbrev RecordCustodyZKJ {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : Prop :=
  RecordCustodyZKJPLaw I none dj Jf wr (fun _ => .subset kd)

/-- `RecordCustodyZKJ` is the record's custody at `setupOf (zkInputs I)` and the sample file's at `subset kd`. -/
theorem recordCustodyZKJ_iff {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) :
    RecordCustodyZKJ I dj Jf wr kd ↔
      Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ Discharge.ZkLink.DrawFileZK dj (.subset kd) n :=
  Iff.rfl

/-- At no public inputs, `RecordCustodyZKJP` is `RecordCustodyZKJ`. -/
theorem recordCustodyZKJP_none {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : RecordCustodyZKJP I none dj Jf wr kd ↔ RecordCustodyZKJ I dj Jf wr kd :=
  Iff.rfl

end FlockSoundness.Assumptions.Zk
