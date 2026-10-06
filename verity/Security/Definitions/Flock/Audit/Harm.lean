import Definitions.Flock.Audit.Law

/-! The definitions of `Proofs.Flock.Soundness.Audit.Harm` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal

variable {n : ℕ}

/-- **`harm_bound`'s specification**: every wrong-unit set that escapes with probability at least `δ` has harm at most
`H`. -/
def IsHarmBound (L : Law n) (h : Fin n → ℝ) (δ : ℝ≥0∞) (H : ℝ) : Prop :=
  ∀ B : Finset (Fin n), δ ≤ L.escape B → ∑ u ∈ B, h u ≤ H

end FlockSoundness.Audit
