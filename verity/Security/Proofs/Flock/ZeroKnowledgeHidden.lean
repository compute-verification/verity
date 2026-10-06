import Specs.Flock.Guarantees.ZeroKnowledgeHidden

/-!
# Proof of C-Flock's zero-knowledge guarantee under hidden outputs

`Flock.SecurityProofs.ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden` is `zk_session_viewHJ`. With
`Flock.SecurityProofs.EndToEndHidden` at the same `dj` and `y₀` it is `zk_session_composedHJ`'s composition: sound and
zero knowledge on one plan.
-/

namespace Flock.SecurityProofs

/-- **C-Flock's one-stage audit under hidden outputs is zero knowledge.** -/
theorem ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden :=
  fun _ _ P I ht _ dj _ y₀ hdm hKey S R ys =>
    FlockSoundness.Discharge.ZkHidden.zk_session_viewHJ P I dj y₀ ht hdm hKey S R ys

end Flock.SecurityProofs
