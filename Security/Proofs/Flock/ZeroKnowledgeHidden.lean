import Proofs.Flock.ZeroKnowledgeHidden.Statement

/-!
# Proof of C-Flock's zero-knowledge guarantee under hidden outputs

`Flock.SecurityProofs.ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden` is `zk_session_viewHJ`. With
`Flock.SecurityProofs.EndToEndHidden` at the same `dj` and `y₀` it is `zk_session_composedHJ`'s composition: sound and
zero knowledge on one plan.
-/

namespace Flock.SecurityProofs

/-- **C-Flock's one-stage audit under hidden outputs is zero knowledge**, at the pinned key's hiding gap, where hm96
hiding holds with no premise (`CROnly.hm96Hiding_gap`). -/
theorem ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden :=
  fun _ _ P I ht _ dj _ _ y₀ hdm S R ys =>
    FlockSoundness.Discharge.ZkHidden.zk_session_viewHJ P I dj y₀ ht hdm
      (FlockSoundness.CROnly.hm96Hiding_gap FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash)
      S R ys

end Flock.SecurityProofs
