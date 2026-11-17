module

public import Proofs.Flock.Soundness.Assumptions.ZkP

/-! The `--zk` session's premises that are instances of its named ones: `shape.premises` in `security/lean-audit.json`
names this module, so the audit reads each abbrev here as the premise it unfolds to. -/

@[expose] public section

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- **The record is the live session's, at `J` tables** (`live-verifier`, as `RecordCustodyZK`): the record
`verify --zk --session-tables J` reads is the one the challenger wrote for the live session
(`RecordsLiveZKJ`: its sample, proofs and link points, and each table's root and rounds, are the live outcome's), and its
sample file the one `verity-verify-proof draw` printed for the verifier's own sample under the subset sampling design (`SampleFileZK` at
`subset kd` over the `n` units; `RecordCustodyZKJS` is its stratified form): `RecordCustodyZKJPLaw` at no public inputs
and `subset kd` at every sample (`recordCustodyZKJ_iff`). -/
abbrev RecordCustodyZKJ {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : Prop :=
  RecordCustodyZKJPLaw I none dj Jf wr (fun _ => .subset kd)

theorem recordCustodyZKJ_iff {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) :
    RecordCustodyZKJ I dj Jf wr kd ↔
      Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ Discharge.ZkLink.SampleFileZK dj (.subset kd) n :=
  Iff.rfl

theorem recordCustodyZKJP_none {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : RecordCustodyZKJP I none dj Jf wr kd ↔ RecordCustodyZKJ I dj Jf wr kd :=
  Iff.rfl

end FlockSoundness.Assumptions.Zk
