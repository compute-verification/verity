import Proofs.Flock.Soundness.Discharge.ZkPubIn.Runs

/-!
# The custody assumption of a `--zk --public-inputs F` session

`RecordCustodyZKP` (`live-verifier`): `Assumptions.Zk.RecordCustodyZK` at flock-verify's acceptance with the verifier's
public inputs. `RecordCustodyZK` reads the record only at transcripts `verify --zk` accepts without `--public-inputs`,
and those are none for a circuit with public input ports (`Stmt.setupHidden` refuses it), so it says nothing there.
-/

namespace FlockSoundness.Assumptions.ZkPubIn

open FlockSoundness

/-- **The record is the live session's, with the verifier's public inputs `own`** (`live-verifier`, as
`RecordCustodyZK`): at every transcript `verify --zk --public-inputs F` accepts, the record's draw is `dj S`, its proofs
are the prover's, and its root, link points and rounds are the live outcome's (`RecordsLiveZKP`); and the draw file is
the one `flock-verify draw` printed for the verifier's own draw (`DrawFileZK`). -/
def RecordCustodyZKP {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript)
    (law : Flock.Draw.Law) (pop : ℕ) : Prop :=
  Discharge.ZkPubIn.RecordsLiveZKP I own dj wr ∧ Discharge.ZkLink.DrawFileZK dj law pop

end FlockSoundness.Assumptions.ZkPubIn
