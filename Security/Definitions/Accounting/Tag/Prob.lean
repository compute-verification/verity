import Mathlib.Probability.ProbabilityMassFunction.Basic

/-!
# Probability of an event under a distribution (trusted)

The tag statements draw a key, a seed matrix or the words a prover did not read from a distribution, uniform
(`PMF.uniformOfFintype`) or any (`μ : PMF α`), and bound the probability of an event. `prob μ E` is that probability:
Mathlib's outer measure of `{x | E x}` under `μ`. A randomized adversary is an average of deterministic ones, so every
statement quantifies over deterministic guesses only.
-/

namespace Accounting.Tag

open scoped ENNReal

/-- The probability that a draw from `μ` satisfies `E`. -/
noncomputable def prob {α : Type} (μ : PMF α) (E : α → Prop) : ℝ≥0∞ :=
  μ.toOuterMeasure {x | E x}

end Accounting.Tag
