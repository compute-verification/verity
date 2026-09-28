import Mathlib.Order.Bounds.Defs
import Mathlib.Tactic.NormNum
import Definitions.Pous.Model.Inc

/-!
# Certified numbers

The source report's operating point for Theorem 1(ii) (report §3): `S = (18/19)·|C|`, `β = (31/32)·|C|` and blocks of
`ℓ = 98,304` bits give `p₀ = 1 − 1/19 + 1/32 + 1/98304 ≈ 0.978630`, and `k = 214` is the least `k` with
`p₀^k ≤ 1/100`.
-/

namespace Pous.Accounting

/-- `p₀` at the report's operating point. -/
def reportP₀ : ℚ := 1 - 1 / 19 + 1 / 32 + 1 / 98304

/-- `p₀ β S B ℓ` is `reportP₀` at `S = (18/19)·B·ℓ`, `β = (31/32)·B·ℓ`, `ℓ = 98304`, whenever those are whole. -/
theorem p₀_at_report (B β S : ℕ) (hB : 0 < B) (hS : (S : ℝ) = 18 / 19 * (B * 98304))
    (hβ : (β : ℝ) = 31 / 32 * (B * 98304)) : p₀ β S B 98304 = reportP₀ := by
  have hB' : (B : ℝ) ≠ 0 := by exact_mod_cast hB.ne'
  simp only [p₀, hS, hβ, reportP₀]
  push_cast
  field_simp
  ring

theorem report_k214 : IsLeast {k : ℕ | reportP₀ ^ k ≤ 1 / 100} 214 := by
  refine ⟨by norm_num [reportP₀], fun k hk => ?_⟩
  by_contra h
  have hmono : reportP₀ ^ 213 ≤ reportP₀ ^ k :=
    pow_le_pow_of_le_one (by norm_num [reportP₀]) (by norm_num [reportP₀]) (by omega)
  have h213 : (1 / 100 : ℚ) < reportP₀ ^ 213 := by norm_num [reportP₀]
  exact absurd (h213.trans_le (hmono.trans hk)) (lt_irrefl _)

end Pous.Accounting
