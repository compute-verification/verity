module

public import Mathlib.Probability.ProbabilityMassFunction.Basic

/-! The probability of an event under a distribution (`prPMF`, `prPMFReal`). -/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

/-- The probability that a sample from `μ` satisfies `E`. -/
noncomputable def prPMF {α : Type} (μ : PMF α) (E : α → Prop) : ℝ≥0∞ :=
  μ.toOuterMeasure {x | E x}

/-- `prPMF` as a real number. -/
noncomputable abbrev prPMFReal {α : Type} (μ : PMF α) (E : α → Prop) : ℝ :=
  (prPMF μ E).toReal

end Verity.Game
