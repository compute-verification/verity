module

public import Proofs.Flock.Soundness.Discharge.Composed.Coined
public import Proofs.Flock.Soundness.Discharge.Composed.DefsJ
public import Proofs.Flock.Soundness.Discharge.ZkLink.SampleOk
public import Proofs.Flock.Soundness.Discharge.ZkView.Defs
public import Proofs.Flock.Soundness.ZK.Complete
public import Proofs.Flock.Soundness.Premises
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
# The `--zk` session's assumptions

The instances of `HashDerivedKey` and `PadNonvanishing` that `--zk` headlines take, stated as Prop constants so the
audit names them. The record's custody (`live-verifier`) that `--zk`'s soundness takes, `RecordCustodyZKJ`, at `J`
tables under the subset sample, is `RecordCustodyZKJPLaw`'s instance at no public inputs (`Proofs.Flock.Soundness.Premises`), which this module imports so
its readers keep the name.
-/

@[expose] public section

namespace FlockSoundness.Assumptions.Zk

open FlockSoundness

/-- The salt hash keys Hm96 hiding at `2^-193` (`HashDerivedKey` at `ZkView.My`, `ZkView.saltHash`). -/
def HashDerivedKeyHm96 : Prop := HashDerivedKey Discharge.ZkView.My Discharge.ZkView.saltHash

/-- `HashDerivedKeyHm96` is `Claims.Hm96Hiding` at the deployed instance, by definition: it adds no assumption. -/
theorem hashDerivedKeyHm96_iff :
    HashDerivedKeyHm96 ↔
      Claims.Hm96Hiding Discharge.ZkView.My Discharge.ZkView.saltHash ((2 : ENNReal) ^ (-193 : ℤ)) :=
  Iff.rfl

/-- Every pad of at most 40 columns is nonvanishing under `execArith` at `fOf PAD_C`. -/
def PadNonvanishing40 : Prop :=
  ∀ cols ≤ 40, PadNonvanishing execArith cols (Refine.fOf Flock.Zk.PAD_C)

end FlockSoundness.Assumptions.Zk
