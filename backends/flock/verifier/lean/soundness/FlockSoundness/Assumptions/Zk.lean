import FlockSoundness.Discharge.Composed.Coined
import FlockSoundness.Discharge.ZkLink.DrawOk
import FlockSoundness.ZK.Complete

/-!
# The `--zk` session's assumptions, closed

The instances of `HashDerivedKey` and `PadNonvanishing` that `--zk` headlines take, stated as Prop constants so the
audit names them, and the record's custody (`RecordCustodyZK`, `live-verifier`) that `--zk`'s soundness takes.
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

* `RecordsLiveZK I dj wr`: the record `verify --zk` reads is the one the verifier's coin server wrote for the live
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

end FlockSoundness.Assumptions.Zk
