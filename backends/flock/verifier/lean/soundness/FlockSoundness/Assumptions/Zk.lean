import FlockSoundness.Discharge.Composed.Defs
import FlockSoundness.ZK.Complete

/-!
# The `--zk` session's assumptions, closed

The instances of `HashDerivedKey` and `PadNonvanishing` that `--zk` headlines take, stated as Prop constants so the
audit names them.
-/

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- The salt hash keys Hm96 hiding at `2^-193` (`HashDerivedKey` at `ZkView.My`, `ZkView.saltHash`). -/
def HashDerivedKeyHm96 : Prop := HashDerivedKey Discharge.ZkView.My Discharge.ZkView.saltHash

/-- Every pad of at most 40 columns is nonvanishing under `execArith` at `fOf PAD_C`. -/
def PadNonvanishing40 : Prop :=
  ∀ cols ≤ 40, PadNonvanishing execArith cols (Refine.fOf Flock.Zk.PAD_C)

end FlockSoundness.Assumptions.Zk
