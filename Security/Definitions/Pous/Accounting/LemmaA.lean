import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Definitions.Pous.Game.Prob

/-!
# Lemma A's bound, the history union, and the A7′ certificate (sms5 ideal-model review, Track A)
-/

namespace Pous

open scoped ENNReal

/-- `|Hist| = Σ_{j<k} B^j = (B^k − 1)/(B − 1)`: the histories of fewer than `k` indices, over which A3′ takes its
union. -/
def histCount (B k : ℕ) : ℕ := ∑ j ∈ Finset.range k, B ^ j

/-- Lemma A in M1 (tradeoff §1.2, exact binomials; the A2 bound with a domain of `M` points):
`Γ(g) = 2^S · C(B, g) · Σ_{h ≤ g} C((g+1)·T, h) / M^g`. -/
noncomputable def gammaM1 (B S T g M : ℕ) : ℝ≥0∞ :=
  2 ^ S * (B.choose g) * (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ≥0∞)) / (M : ℝ≥0∞) ^ g

end Pous
