module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJP

/-! The `--zk` session's custody at the verifier's public inputs, a form of its own because `RecordCustodyZKJ`'s `setupOf` refuses a circuit with public input ports. -/

@[expose] public section

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- `live-verifier` as `RecordCustodyZKJ`, at the verifier's public inputs. Provisional. -/
def RecordCustodyZKJP {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : Prop :=
  Discharge.Composed.RecordsLiveZKJA (Discharge.ZkHidden.AcceptsZKP I own)
      (Discharge.Exec.setupOfP (Discharge.ZkExec.zkInputs I) own) dj Jf wr ∧
    Discharge.ZkLink.SampleFileZK dj (.subset kd) n

/-- **The record is the live session's, at `J` tables, at the verifier's public inputs, under the sampling design of its sample**
(`live-verifier`, as `RecordCustodyZKJP`): the record `verify --zk --session-tables J --public-inputs` reads is the one
the challenger wrote for the live session (`RecordsLiveZKJA` at that command's acceptance and setup), and its sample file at
each sample `S` the one `verity-verify-proof draw` printed for the verifier's own sample of `S` under the sampling design `samplingDesign S` it drew `S` by
over the `n` proof blocks (`ZkLink.drawOf`): the registration's sampling design at every sample (`recordCustodyZKJPLaw_const`), or
`subset |S|` for a member's part `S` of a window's sample. Provisional. -/
def RecordCustodyZKJPLaw {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (samplingDesign : Finset (Fin n) → Flock.Draw.SamplingDesign) : Prop :=
  Discharge.Composed.RecordsLiveZKJA (Discharge.ZkHidden.AcceptsZKP I own)
      (Discharge.Exec.setupOfP (Discharge.ZkExec.zkInputs I) own) dj Jf wr ∧
    ∀ S D, Flock.Draw.UnitSample.ofJson (dj S) = .ok D → D = Discharge.ZkLink.drawOf (samplingDesign S) n S

theorem recordCustodyZKJPLaw_const {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (samplingDesign : Flock.Draw.SamplingDesign) :
    RecordCustodyZKJPLaw I own dj Jf wr (fun _ => samplingDesign) ↔
      Discharge.Composed.RecordsLiveZKJA (Discharge.ZkHidden.AcceptsZKP I own)
          (Discharge.Exec.setupOfP (Discharge.ZkExec.zkInputs I) own) dj Jf wr ∧
        Discharge.ZkLink.SampleFileZK dj samplingDesign n :=
  Iff.rfl

theorem recordCustodyZKJPLaw_subset {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : Option ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : RecordCustodyZKJPLaw I own dj Jf wr (fun _ => .subset kd) ↔ RecordCustodyZKJP I own dj Jf wr kd :=
  Iff.rfl

end FlockSoundness.Assumptions.Zk
