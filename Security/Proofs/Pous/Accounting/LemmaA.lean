import Definitions.Pous.Accounting.LemmaA

/-! The proof-only part of `Definitions.Pous.Accounting.LemmaA`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pous

open scoped ENNReal

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
