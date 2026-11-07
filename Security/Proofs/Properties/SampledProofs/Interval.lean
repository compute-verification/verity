module

public import Properties.SampledProofs.Interval
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The interval bounds are sound

`Properties.SampledProofs.ExpLowerSound`, `ExpUpperSound`, `LnLowerSound`, `LnUpperSound`, `PowLowerSound`,
`PowUpperSound`, `BoundLnUpperSound` and `BoundLnBoundSound`. Each bound passes at a deployed precision, one step of
`2^-p` wide or less (`_honest`), and each conclusion fails at a bound on the wrong side (`_nontrivial`).
-/

@[expose] public section

namespace Properties.SampledProofs.SecurityProofs

open _root_.SampledProofs.Interval

theorem expLowerSound : ExpLowerSound := sorry

theorem expUpperSound : ExpUpperSound := sorry

theorem lnLowerSound : LnLowerSound := sorry

theorem lnUpperSound : LnUpperSound := sorry

theorem powLowerSound : PowLowerSound := sorry

theorem powUpperSound : PowUpperSound := sorry

theorem boundLnUpperSound : BoundLnUpperSound := sorry

theorem boundLnBoundSound : BoundLnBoundSound := sorry

/-! ## Witnesses -/

/-- At `p = 64`, `e^{-1}`'s bounds are one step of `2^-64` apart or less. -/
theorem expLowerSound_honest : expUpper (-1) 64 - expLower (-1) 64 ≤ 1 / 2 ^ 63 := sorry

/-- No rational at least `3` is below `e`. -/
theorem expLowerSound_nontrivial : ¬ ((3 : ℚ) : ℝ) ≤ Real.exp ((1 : ℚ) : ℝ) := sorry

/-- At `p = 64`, `e`'s bounds are one step of `2^-64` apart or less. -/
theorem expUpperSound_honest : expUpper 1 64 - expLower 1 64 ≤ 1 / 2 ^ 63 := sorry

/-- `2` is not above `e`. -/
theorem expUpperSound_nontrivial : ¬ Real.exp ((1 : ℚ) : ℝ) ≤ ((2 : ℚ) : ℝ) := sorry

/-- At `p = 64`, `ln(10^-30)`'s bounds are within `2^-62`. -/
theorem lnLowerSound_honest : lnUpper (1 / 10 ^ 30) 64 - lnLower (1 / 10 ^ 30) 64 ≤ 1 / 2 ^ 62 := sorry

/-- `1` is not below `ln 2`. -/
theorem lnLowerSound_nontrivial : ¬ ((1 : ℚ) : ℝ) ≤ Real.log ((2 : ℚ) : ℝ) := sorry

/-- At `p = 64`, `ln 2`'s bounds are within `2^-62`. -/
theorem lnUpperSound_honest : lnUpper 2 64 - lnLower 2 64 ≤ 1 / 2 ^ 62 := sorry

/-- `0` is not above `ln 2`. -/
theorem lnUpperSound_nontrivial : ¬ Real.log ((2 : ℚ) : ℝ) ≤ ((0 : ℚ) : ℝ) := sorry

/-- At `p = 64`, `2^-1`'s bounds are within `2^-62`. -/
theorem powLowerSound_honest : powUpper 2 (-1) 64 - powLower 2 (-1) 64 ≤ 1 / 2 ^ 62 := sorry

/-- `1` is not below `(1/2)^1`. -/
theorem powLowerSound_nontrivial : ¬ ((1 : ℚ) : ℝ) ≤ ((1 / 2 : ℚ) : ℝ) ^ ((1 : ℚ) : ℝ) := sorry

/-- At `p = 64`, `(1/2)^(1/2)`'s bounds are within `2^-62`. -/
theorem powUpperSound_honest : powUpper (1 / 2) (1 / 2) 64 - powLower (1 / 2) (1 / 2) 64 ≤ 1 / 2 ^ 62 := sorry

/-- `0` is not above `(1/2)^1`. -/
theorem powUpperSound_nontrivial : ¬ ((1 / 2 : ℚ) : ℝ) ^ ((1 : ℚ) : ℝ) ≤ ((0 : ℚ) : ℝ) := sorry

/-- `Bound.lnUpper (1/100)` is within `2^-63` of `ln 100` from above: `Bound.lnBound 100 1 false` is below it. -/
theorem boundLnUpperSound_honest :
    _root_.SampledProofs.Bound.lnUpper (1 / 100) - _root_.SampledProofs.Bound.lnBound 100 1 false < 1 / 2 ^ 63 := sorry

/-- `0` is not at least `ln 2 = ln(1/(1/2))`. -/
theorem boundLnUpperSound_nontrivial : ¬ Real.log (1 / ((1 / 2 : ℚ) : ℝ)) ≤ ((0 : ℚ) : ℝ) := sorry

/-- `Bound.lnBound 100 1`'s two sides are within `2^-100`. -/
theorem boundLnBoundSound_honest :
    _root_.SampledProofs.Bound.lnBound 100 1 true - _root_.SampledProofs.Bound.lnBound 100 1 false < 1 / 2 ^ 100 :=
  sorry

/-- `1` is not below `ln(2/1)`. -/
theorem boundLnBoundSound_nontrivial : ¬ ((1 : ℚ) : ℝ) ≤ Real.log (((2 : ℕ) : ℝ) / ((1 : ℕ) : ℝ)) := sorry

end Properties.SampledProofs.SecurityProofs
