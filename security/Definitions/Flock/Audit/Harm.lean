module

public import Definitions.Flock.Audit.SamplingDesign
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! Harm bounds: `IsHarmBound` specifies what `Stratified.harm_bound` returns, and this file does not prove that
implementation. -/

@[expose] public section

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal
variable {n : ℕ}

/-- **`harm_bound`'s specification**: every incorrect set that escapes with probability at least `δ` has harm at most
`H`. -/
def IsHarmBound (L : SamplingDesign n) (h : Fin n → ℝ) (δ : ℝ≥0∞) (H : ℝ) : Prop :=
  ∀ B : Finset (Fin n), δ ≤ L.missProb B → ∑ u ∈ B, h u ≤ H

end FlockSoundness.Audit
