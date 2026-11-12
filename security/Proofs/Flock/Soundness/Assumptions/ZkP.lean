module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJP

/-!
# The `--zk` session's custody at the verifier's public inputs

`RecordCustodyZKJP`: `RecordCustodyZKJ` at flock-verify `--zk --public-inputs`, whose acceptance is `AcceptsZKP I own`
and whose session sets up at `setupOfP (zkInputs I) own`. `RecordCustodyZKJ` is stated at `setupOf (zkInputs I)`, which
refuses a circuit with public input ports (`HmIn.checkOwn` at `none`), so a session at public inputs needs this form.
`RecordCustodyZKJPLaw` is the same custody with each sample's file under the sampling design the verifier samples it by,
read per sample: at one sampling design for every sample its sample half is `DrawFileZK` at that sampling design
(`recordCustodyZKJPLaw_const`), and at the subset sampling design it is `RecordCustodyZKJP`
(`recordCustodyZKJPLaw_subset`); a member of a window's sample samples its part `S` by `subset |S|`. `RecordCustodyZKJ`
is its instance at no public inputs and the subset sampling design (`Proofs.Flock.Soundness.Premises`).
-/

@[expose] public section

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- **The record is the live session's, at `J` tables, at the verifier's public inputs** (`live-verifier`, as
`RecordCustodyZKJ`): the record `verify --zk --session-tables J --public-inputs` reads is the one the challenger wrote
for the live session (`RecordsLiveZKJA` at that command's acceptance and setup: its sample, proofs and link points, and
each table's root and rounds, are the live outcome's), and its sample file the one `flock-verify draw` printed for the
verifier's own sample under the subset sampling design (`DrawFileZK` at `subset kd` over the `n` proof blocks).
Provisional. -/
def RecordCustodyZKJP {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : Prop :=
  Discharge.Composed.RecordsLiveZKJA (Discharge.ZkHidden.AcceptsZKP I own)
      (Discharge.Exec.setupOfP (Discharge.ZkExec.zkInputs I) own) dj Jf wr ∧
    Discharge.ZkLink.DrawFileZK dj (.subset kd) n

/-- **The record is the live session's, at `J` tables, at the verifier's public inputs, under the sampling design of its sample**
(`live-verifier`, as `RecordCustodyZKJP`): the record `verify --zk --session-tables J --public-inputs` reads is the one
the challenger wrote for the live session (`RecordsLiveZKJA` at that command's acceptance and setup), and its sample file at
each sample `S` the one `flock-verify draw` printed for the verifier's own sample of `S` under the sampling design `law S` it drew `S` by
over the `n` proof blocks (`ZkLink.drawOf`): the registration's law at every sample (`recordCustodyZKJPLaw_const`), or
`subset |S|` for a member's part `S` of a window's sample. Provisional. -/
def RecordCustodyZKJPLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (law : Finset (Fin n) → Flock.Draw.Law) : Prop :=
  Discharge.Composed.RecordsLiveZKJA (Discharge.ZkHidden.AcceptsZKP I own)
      (Discharge.Exec.setupOfP (Discharge.ZkExec.zkInputs I) own) dj Jf wr ∧
    ∀ S D, Flock.Draw.UnitDraw.ofJson (dj S) = .ok D → D = Discharge.ZkLink.drawOf (law S) n S

/-- At one sampling design for every sample, `RecordCustodyZKJPLaw`'s sample half is `DrawFileZK` at that sampling design. -/
theorem recordCustodyZKJPLaw_const {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (law : Flock.Draw.Law) :
    RecordCustodyZKJPLaw I own dj Jf wr (fun _ => law) ↔
      Discharge.Composed.RecordsLiveZKJA (Discharge.ZkHidden.AcceptsZKP I own)
          (Discharge.Exec.setupOfP (Discharge.ZkExec.zkInputs I) own) dj Jf wr ∧
        Discharge.ZkLink.DrawFileZK dj law n :=
  Iff.rfl

/-- At the subset sampling design, `RecordCustodyZKJPLaw` is `RecordCustodyZKJP`. -/
theorem recordCustodyZKJPLaw_subset {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : RecordCustodyZKJPLaw I own dj Jf wr (fun _ => .subset kd) ↔ RecordCustodyZKJP I own dj Jf wr kd :=
  Iff.rfl

end FlockSoundness.Assumptions.Zk
