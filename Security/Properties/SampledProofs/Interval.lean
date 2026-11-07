module

public import SampledProofs.Interval
public import SampledProofs.Bound
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Interval bounds on `e^x`, `ln y` and `c^θ`: what they bound (`SampledProofs.Interval`, `SampledProofs.Bound`)

`SampledProofs.Interval` computes rationals, each a multiple of `2^-p`, around `Real.exp`, `Real.log` and `Real.rpow`;
`SampledProofs.Bound` bounds the logarithms its harm checks read (`densityBound`, `maxHarmCheck`). These properties say
each bound is on its side at every input and precision:

* `ExpLowerSound`, `ExpUpperSound`: `expLower x p ≤ e^x ≤ expUpper x p` for every rational `x`;
* `LnLowerSound`, `LnUpperSound`: `lnLower y p ≤ ln y ≤ lnUpper y p` for `y > 0`;
* `PowLowerSound`, `PowUpperSound`: `powLower c θ p ≤ c^θ ≤ powUpper c θ p` for `c ≥ 0` and every rational `θ`;
* `BoundLnUpperSound`: `ln(1/δ) ≤ Bound.lnUpper δ` for `0 < δ < 1`, the budget `densityBound` fills;
* `BoundLnBoundSound`: `Bound.lnBound a b false ≤ ln(a/b) ≤ Bound.lnBound a b true` for `0 < b ≤ a`.

How tight a bound is (within `2^-p` of its value at the points the module's `#guard`s check) is not stated.
-/

@[expose] public section

namespace Properties.SampledProofs

open _root_.SampledProofs.Interval (expLower expUpper lnLower lnUpper powLower powUpper)

/-- **`expLower` is below `e^x`**, at every rational `x` and precision `p`. -/
def ExpLowerSound : Prop := ∀ (x : ℚ) (p : ℕ), ((expLower x p : ℚ) : ℝ) ≤ Real.exp x

/-- **`expUpper` is above `e^x`**, at every rational `x` and precision `p`. -/
def ExpUpperSound : Prop := ∀ (x : ℚ) (p : ℕ), Real.exp x ≤ ((expUpper x p : ℚ) : ℝ)

/-- **`lnLower` is below `ln y`**, at every rational `y > 0` and precision `p`. -/
def LnLowerSound : Prop := ∀ (y : ℚ) (p : ℕ), 0 < y → ((lnLower y p : ℚ) : ℝ) ≤ Real.log y

/-- **`lnUpper` is above `ln y`**, at every rational `y > 0` and precision `p`. -/
def LnUpperSound : Prop := ∀ (y : ℚ) (p : ℕ), 0 < y → Real.log y ≤ ((lnUpper y p : ℚ) : ℝ)

/-- **`powLower` is below `c^θ`**, at every rational `c ≥ 0`, rational `θ` and precision `p`. -/
def PowLowerSound : Prop := ∀ (c θ : ℚ) (p : ℕ), 0 ≤ c → ((powLower c θ p : ℚ) : ℝ) ≤ (c : ℝ) ^ (θ : ℝ)

/-- **`powUpper` is above `c^θ`**, at every rational `c ≥ 0`, rational `θ` and precision `p`. -/
def PowUpperSound : Prop := ∀ (c θ : ℚ) (p : ℕ), 0 ≤ c → (c : ℝ) ^ (θ : ℝ) ≤ ((powUpper c θ p : ℚ) : ℝ)

/-- **`Bound.lnUpper δ` is at least `ln(1/δ)`** for `0 < δ < 1`: the risk budget `densityBound` fills is at least the
one its harm bound needs. -/
def BoundLnUpperSound : Prop :=
  ∀ δ : ℚ, 0 < δ → δ < 1 → Real.log (1 / (δ : ℝ)) ≤ ((_root_.SampledProofs.Bound.lnUpper δ : ℚ) : ℝ)

/-- **`Bound.lnBound` brackets `ln(a/b)`** for `0 < b ≤ a`: below it, and above it when `up`, as `maxHarmCheck` reads
it. -/
def BoundLnBoundSound : Prop :=
  ∀ a b : ℕ, 0 < b → b ≤ a →
    ((_root_.SampledProofs.Bound.lnBound a b false : ℚ) : ℝ) ≤ Real.log ((a : ℝ) / b) ∧
      Real.log ((a : ℝ) / b) ≤ ((_root_.SampledProofs.Bound.lnBound a b true : ℚ) : ℝ)

end Properties.SampledProofs
