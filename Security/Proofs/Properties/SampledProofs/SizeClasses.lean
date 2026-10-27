import Properties.SampledProofs.SizeClasses

/-! # Proof that sampled proofs' size classes are the claims' (`Properties.SampledProofs.SizeClasses`)

Each copy is the same expression over a widest class of the same value, so each side unfolds to the other.
-/

namespace Properties.SampledProofs.SecurityProofs

theorem sizeClassesAgree : SizeClassesAgree := by
  unfold SizeClassesAgree
  exact ⟨rfl, rfl⟩

end Properties.SampledProofs.SecurityProofs
