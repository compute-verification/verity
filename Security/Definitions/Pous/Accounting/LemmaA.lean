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

/-- `log₂(|Hist| · Γ(g))` for Lemma A in M1. -/
noncomputable def logHistGammaM1 (B S T g k M : ℕ) : ℝ :=
  Real.logb 2 (histCount B k) + S + Real.logb 2 (B.choose g) +
    Real.logb 2 (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ)) - g * Real.logb 2 M

/-- `log₂(|Hist| · Γ(g))` for Lemma A in M2 (two shared permutations): each hit count carries the identity factor
`g!/(g − h)!`, and the pool is `(M − g)!/M!`. -/
noncomputable def logHistGammaM2 (B S T g k M : ℕ) : ℝ :=
  Real.logb 2 (histCount B k) + S + Real.logb 2 (B.choose g) +
    Real.logb 2 (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h * g.descFactorial h : ℝ)) -
    Real.logb 2 (M.descFactorial g)

/-- One row of A7′: some `g ∈ [1, B]` has `((g − 1)/B)^k ≤ 1/100` and `log₂(|Hist| · Γ(g)) ≤ −128` for every domain
of `M ≥ 2^2048 − 2^2029` points (`log₂ M ≥ 2048 − 2.8·10⁻⁶`, which `φ(N)` meets for a 2048-bit `N` with its top 20
bits set). -/
def A7Row (logHG : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℝ) (B S T k : ℕ) : Prop :=
  ∃ g, 1 ≤ g ∧ g ≤ B ∧ (((g - 1 : ℕ) : ℚ) / B) ^ k ≤ 1 / 100 ∧
    ∀ M : ℕ, 2 ^ 2048 - 2 ^ 2029 ≤ M → logHG B S T g k M ≤ -128

end Pous
