module

public import Spot.Interval
public import Spot.Bound
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Interval bounds on `e^x`, `ln y` and `c^θ`: what they bound (`Spot.Interval`, `Spot.Bound`)

`Spot.Interval` computes rationals, each a multiple of `2^-p`, around `Real.exp`, `Real.log` and `Real.rpow`;
`Spot.Bound` bounds the logarithms its harm checks read (`densityBound`, `maxHarmCheck`). These properties say
each bound is on its side at every input and precision:

* `ExpLowerSound`, `ExpUpperSound`: `expLower x p ≤ e^x ≤ expUpper x p` for every rational `x`;
* `LnLowerSound`, `LnUpperSound`: `lnLower y p ≤ ln y ≤ lnUpper y p` for every rational `y`;
* `PowLowerSound`, `PowUpperSound`: `powLower c θ p ≤ c^θ ≤ powUpper c θ p` for every rational `c` and `θ`;
* `BoundLnUpperSound`: `ln(1/δ) ≤ Bound.lnUpper δ` for `0 < δ < 1`, the budget `densityBound` fills;
* `BoundLnBoundSound`: `Bound.lnBound a b false ≤ ln(a/b) ≤ Bound.lnBound a b true` for `0 < b ≤ a`.

`ln` is `Real.log` and `c^θ` is `Real.rpow`, with Mathlib's values at `y ≤ 0` and `c ≤ 0`. `Bound`'s helpers hold only
on their ranges, so those two statements range over them: `δ = (n + 1)/(n + k + 2)` is every rational in `(0, 1)`,
and `b = m + 1`, `a = m + 1 + k` every pair with `0 < b ≤ a`.

How tight a bound is (within `2^-p` of its value at the points the module's `#guard`s check) is not stated.
-/

@[expose] public section

namespace Properties.Spot

open _root_.Spot.Interval (expLower expUpper lnLower lnUpper powLower powUpper)

/-- **`expLower` is below `e^x`**, at every rational `x` and precision `p`. -/
def ExpLowerSound : Prop := ∀ (x : ℚ) (p : ℕ), ((expLower x p : ℚ) : ℝ) ≤ Real.exp x

/-- **`expUpper` is above `e^x`**, at every rational `x` and precision `p`. -/
def ExpUpperSound : Prop := ∀ (x : ℚ) (p : ℕ), Real.exp x ≤ ((expUpper x p : ℚ) : ℝ)

/-- **`lnLower` is below `ln y`**, at every rational `y` and precision `p`. -/
def LnLowerSound : Prop := ∀ (y : ℚ) (p : ℕ), ((lnLower y p : ℚ) : ℝ) ≤ Real.log y

/-- **`lnUpper` is above `ln y`**, at every rational `y` and precision `p`. -/
def LnUpperSound : Prop := ∀ (y : ℚ) (p : ℕ), Real.log y ≤ ((lnUpper y p : ℚ) : ℝ)

/-- **`powLower` is below `c^θ`**, at every rational `c` and `θ` and precision `p`. -/
def PowLowerSound : Prop := ∀ (c θ : ℚ) (p : ℕ), ((powLower c θ p : ℚ) : ℝ) ≤ (c : ℝ) ^ (θ : ℝ)

/-- **`powUpper` is above `c^θ`**, at every rational `c` and `θ` and precision `p`. -/
def PowUpperSound : Prop := ∀ (c θ : ℚ) (p : ℕ), (c : ℝ) ^ (θ : ℝ) ≤ ((powUpper c θ p : ℚ) : ℝ)

/-- **`Bound.lnUpper δ` is at least `ln(1/δ)`** at every rational `δ = (n + 1)/(n + k + 2)` in `(0, 1)`: the risk
budget `densityBound` fills is at least the one its harm bound needs. -/
def BoundLnUpperSound : Prop :=
  ∀ n k : ℕ, Real.log (1 / ((((n + 1 : ℕ) : ℚ) / ((n + k + 2 : ℕ) : ℚ) : ℚ) : ℝ)) ≤
    ((_root_.Spot.Bound.lnUpper (((n + 1 : ℕ) : ℚ) / ((n + k + 2 : ℕ) : ℚ)) : ℚ) : ℝ)

/-- **`Bound.lnBound` brackets `ln(a/b)`** at every `b = m + 1`, `a = m + 1 + k`: below it, and above it when `up`,
as `maxHarmCheck` reads it. -/
def BoundLnBoundSound : Prop :=
  ∀ m k : ℕ,
    ((_root_.Spot.Bound.lnBound (m + 1 + k) (m + 1) false : ℚ) : ℝ) ≤
        Real.log (((m + 1 + k : ℕ) : ℝ) / ((m + 1 : ℕ) : ℝ)) ∧
      Real.log (((m + 1 + k : ℕ) : ℝ) / ((m + 1 : ℕ) : ℝ)) ≤
        ((_root_.Spot.Bound.lnBound (m + 1 + k) (m + 1) true : ℚ) : ℝ)

end Properties.Spot
