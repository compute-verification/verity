module

public import Proofs.Flock.ZeroKnowledgeHidden.Statement
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
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask

/-!
# Proof of Flock's zero-knowledge guarantee under hidden outputs

`Flock.SecurityProofs.ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden` is `zk_session_viewHJ`. With
`Flock.SecurityProofs.EndToEndHidden` at the same `dj` and `y₀` it is `zk_session_composedHJ`'s composition: sound and
zero knowledge on one plan.
-/

@[expose] public section

namespace Flock.SecurityProofs

/-- **Flock's one-stage audit under hidden outputs is zero knowledge**, at the pinned key's hiding gap, where hm96
hiding holds with no premise (`CROnly.hm96Hiding_gap`), and every table set up avoids the mask slot
(`FailClosed.DrawSetupZHJ.avoidsMask`). -/
theorem ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden :=
  fun _ _ P I ht _ dj _ _ y₀ S R ys =>
    FlockSoundness.Discharge.ZkHidden.zk_session_viewHJ P I dj y₀ ht
      (fun _ _ y => FlockSoundness.Discharge.FailClosed.DrawSetupZHJ.avoidsMask y)
      (FlockSoundness.CROnly.hm96Hiding_gap FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash)
      S R ys

end Flock.SecurityProofs
