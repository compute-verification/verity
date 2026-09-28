import Proofs.Flock.Soundness.Discharge.Composed.Coined
import Proofs.Flock.Soundness.ZK.Complete
import Proofs.Flock.Soundness.Discharge.ZkLink.DrawOk
import Proofs.Flock.Soundness.Discharge.Composed.DefsJ

/-!
# The `--zk` session's assumptions

The instances of `HashDerivedKey` and `PadNonvanishing` that `--zk` headlines take, stated as Prop constants so the
audit names them, and the record's custody (`live-verifier`) that `--zk`'s soundness takes: `RecordCustodyZK` at one
table, `RecordCustodyZKJ` at `J`.
-/

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- The salt hash keys Hm96 hiding at `2^-193` (`HashDerivedKey` at `ZkView.My`, `ZkView.saltHash`). -/
def HashDerivedKeyHm96 : Prop := HashDerivedKey Discharge.ZkView.My Discharge.ZkView.saltHash

/-- Every pad of at most 40 columns is nonvanishing under `execArith` at `fOf PAD_C`. -/
def PadNonvanishing40 : Prop :=
  ∀ cols ≤ 40, PadNonvanishing execArith cols (Refine.fOf Flock.Zk.PAD_C)

/-- **The record is the live session's** (`live-verifier`; the custody of the verifier's `PROTOCOL.md` §17.2, with A6,
`uniform/flock-coin-server`, for the coins in it). Nothing in a session record authenticates its writer, and
`flock-verify verify --zk` reads the draw and the proofs only from the record, so no check of the record establishes
either fact:

* `RecordsLiveZK I dj wr`: the record `verify --zk` reads is the one the challenger wrote for the live
  session. At every transcript it accepts, its draw is `dj S`, its proofs are the prover's, and its root, link points
  and rounds are the live outcome's.
* `DrawFileZK dj law pop`: the draw file is the one `flock-verify draw` printed for the verifier's own draw. Every draw
  file `dj S` that `UnitDraw.ofJson` decodes decodes to `drawOf law pop S`, the draw of `S` under `law` over
  `[0, pop)`.

Both are stated at what flock-verify parses, so the record's serialization is outside them. -/
def RecordCustodyZK {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → Refine.LiveOut mPts → Discharge.Exec.Transcript) (law : Flock.Draw.Law) (pop : ℕ) :
    Prop :=
  Discharge.ZkLink.RecordsLiveZK I dj wr ∧ Discharge.ZkLink.DrawFileZK dj law pop

/-- **The record is the live session's, at `J` tables** (`live-verifier`, as `RecordCustodyZK`): the record
`verify --zk --session-tables J` reads is the one the challenger wrote for the live session
(`RecordsLiveZKJ`: its draw, proofs and link points, and each table's root and rounds, are the live outcome's), and its
draw file the one `flock-verify draw` printed for the verifier's own draw under the subset law, the one law flock-verify
takes at more than one table (`DrawFileZK` at `subset kd` over the `n` units). -/
def RecordCustodyZKJ {n mPts : ℕ} {Reg : Type} (I : Discharge.Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Discharge.Tables.LiveOutJ (Jf S R) mPts → Discharge.Exec.Transcript)
    (kd : ℕ) : Prop :=
  Discharge.Composed.RecordsLiveZKJ I dj Jf wr ∧ Discharge.ZkLink.DrawFileZK dj (.subset kd) n

end FlockSoundness.Assumptions.Zk
