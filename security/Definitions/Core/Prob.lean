module

public import Mathlib.Probability.ProbabilityMassFunction.Basic

/-!
# The probability of an event under a distribution

The uniform average and probability over a finite type are `Game.avg` and `Game.prCoin` (`Definitions.Core.Game`), and
their real-valued forms are `Claims.Ex` and `Claims.Pr` (`Definitions.Core.Interaction`). A sample from any distribution
`μ : PMF α` has `prPMF μ E`, the probability that it satisfies `E`: Mathlib's outer measure of `{x | E x}` under `μ`, in
`[0, ∞]`. `prPMFReal` is the same probability as a real number. A randomized adversary is an average of deterministic
ones, so a statement over this probability quantifies over deterministic guesses only.
-/

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
