import CoreOnsite.Pooled
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Onsite's pooled figures meet the arithmetic of the statements they evaluate (`CoreOnsite.Pooled`)

`onsite bound` prints each resource's pooled figure over a monitoring period, computed by `Core.Onsite.Pooled` in exact naturals
with `ln` a certified bracket. Each property here says the figure satisfies the arithmetic the statement it evaluates
takes as its hypothesis, over the reals, so that statement applies at the figure:

* `LnBrackets`: `ln x`'s two bounds over `2^PREC` are below and above `Real.log x`, for a ratio `x ≥ 1`.
* `IntegrityHorizonCovers`: `ln(1/δ)/q ≤ faultyBlocks q δ + 1`, so `faultyBlocks q δ` is at least every total `w` of
  faulty proof blocks outside the event `Flock.Guarantees.IntegrityHorizon` bounds (`ln(1/δ)/q ≤ w`), for `q > 0` and
  `0 < δ ≤ 1`.
* `IntegrityPeriodCovers`: at `threshold n kd e`, when it is at most `n`, the sampling term
  `C(n − K₀, kd)/C(n, kd)` that `Flock.Guarantees.IntegrityPeriod` sums for a block is at most `e`.
* `PooledShortCovers`: `shortCount U p δ'`, when it is at most `U`, meets `Pous.Period.SecurityProofs.pooledShort_log`'s
  hypothesis `((z − 1)·U·p + ln(1/δ'))/ln z ≤ M` at some `z > 1`, so at least `U − M + 1` of `U` sampling periods
  certify (`certified`), `p` the per-period `δ + ε_max` PoUS publishes.
* `HorizonLeakCovers`: `leakBits`' `Fmax` meets `NetTiming.Guarantees.HorizonLeakAccepted`'s `hF`,
  `ln(1/δ) ≤ q·Fmax`; its faulty proof blocks' bits bound that statement's factor `(N + 1)^Fmax · 2^(b·Fmax)` at
  `b = BLOCK_BITS` at `N = B·units`, so at every `N ≤ B·units`; and its `K` is the blocks' own terms plus those bits.

`IntegrityHorizon`, `IntegrityPeriod` and `pooledShort_log` are in the proofs tree, which a property may not read, so
their arithmetic is restated here as each statement writes it rather than named. They take no premise and no
hypothesis: a positivity condition is a parametrization (`q = (a + 1)/(b + 1)`), and a figure's range is a disjunct.
That the figures' inputs (`q`, `n`, `kd`, `p`, `units`, `every`) are the subprotocols' published values is `CoreOnsite.Bound`'s
reading of the accepted answers, outside these statements.
-/

namespace Properties.Onsite

open Core Core.Onsite

/-- **`ln` brackets the logarithm.** For a ratio `x = (d + 1 + a)/(d + 1) ≥ 1` (every ratio at least `1` with a positive
denominator), `lo/2^PREC ≤ log x ≤ hi/2^PREC`, `(lo, hi) = ln x`. -/
def LnBrackets : Prop :=
  ∀ a d : ℕ,
    ((Pooled.ln ⟨d + 1 + a, d + 1⟩).1 : ℝ) / 2 ^ Pooled.PREC ≤ Real.log (((d + 1 + a : ℕ) : ℝ) / ((d + 1 : ℕ) : ℝ)) ∧
      Real.log (((d + 1 + a : ℕ) : ℝ) / ((d + 1 : ℕ) : ℝ)) ≤ ((Pooled.ln ⟨d + 1 + a, d + 1⟩).2 : ℝ) / 2 ^ Pooled.PREC

/-- **`IntegrityHorizon`'s figure.** For a draw covering at least a fraction `q = (a + 1)/(b + 1) > 0` of each block's
proof blocks and `δ = (c + 1)/(c + 1 + e) ∈ (0, 1]`, `ln(1/δ)/q ≤ faultyBlocks q δ + 1`: every total `w` of faulty proof
blocks with `¬ ln(1/δ)/q ≤ w`, outside the event `IntegrityHorizon` bounds, is at most `faultyBlocks q δ`. -/
def IntegrityHorizonCovers : Prop :=
  ∀ a b c e : ℕ,
    Real.log (1 / (((c + 1 : ℕ) : ℝ) / ((c + 1 + e : ℕ) : ℝ))) / (((a + 1 : ℕ) : ℝ) / ((b + 1 : ℕ) : ℝ)) ≤
      ((Pooled.faultyBlocks ⟨a + 1, b + 1⟩ ⟨c + 1, c + 1 + e⟩ + 1 : ℕ) : ℝ)

/-- **`IntegrityPeriod`'s threshold.** Either `threshold n kd e` exceeds `n`, or a block of `n` proof blocks drawing `kd`
has sampling term `C(n − K₀, kd)/C(n, kd)` at most `e = num/(d + 1)` at `K₀ = threshold n kd e`. -/
def IntegrityPeriodCovers : Prop :=
  ∀ n kd num d : ℕ, n < Pooled.threshold n kd ⟨num, d + 1⟩ ∨
    ((n - Pooled.threshold n kd ⟨num, d + 1⟩).choose kd : ℝ) / (n.choose kd : ℝ) ≤ (num : ℝ) / ((d + 1 : ℕ) : ℝ)

/-- **`pooledShort_log`'s count.** Either `shortCount U p δ'` exceeds `U`, or it meets `pooledShort_log`'s hypothesis
`((z − 1)·U·p + ln(1/δ'))/ln z ≤ M` at some `z > 1`, for `p = pn/(pd + 1)` and `δ' = (c + 1)/(c + 1 + e) ∈ (0, 1]`. -/
def PooledShortCovers : Prop :=
  ∀ U pn pd c e : ℕ, U < Pooled.shortCount U ⟨pn, pd + 1⟩ ⟨c + 1, c + 1 + e⟩ ∨
    ∃ z : ℝ, 1 < z ∧
      ((z - 1) * U * ((pn : ℝ) / ((pd + 1 : ℕ) : ℝ)) + Real.log (1 / (((c + 1 : ℕ) : ℝ) / ((c + 1 + e : ℕ) : ℝ)))) /
          Real.log z ≤ Pooled.shortCount U ⟨pn, pd + 1⟩ ⟨c + 1, c + 1 + e⟩

/-- **`HorizonLeakAccepted`'s count.** For `q = (a + 1)/(b + 1)` and `δ = (c + 1)/(c + 1 + e)`,
`leakBits B every units q δ = (K, Fmax, faulty)` has `ln(1/δ) ≤ q·Fmax` (the statement's `hF`); `faulty` bits cover
the statement's faulty-block factor `(N + 1)^Fmax · 2^(b·Fmax)` at `b = BLOCK_BITS` and `N = B·units`, so at every
`N ≤ B·units`; and `K = B·every + faulty`. -/
def HorizonLeakCovers : Prop :=
  ∀ B every units a b c e : ℕ,
    Real.log (1 / (((c + 1 : ℕ) : ℝ) / ((c + 1 + e : ℕ) : ℝ))) ≤
        (((a + 1 : ℕ) : ℝ) / ((b + 1 : ℕ) : ℝ)) *
          (Pooled.leakBits B every units ⟨a + 1, b + 1⟩ ⟨c + 1, c + 1 + e⟩).2.1 ∧
      (B * units + 1) ^ (Pooled.leakBits B every units ⟨a + 1, b + 1⟩ ⟨c + 1, c + 1 + e⟩).2.1 *
          2 ^ (Pooled.BLOCK_BITS * (Pooled.leakBits B every units ⟨a + 1, b + 1⟩ ⟨c + 1, c + 1 + e⟩).2.1) ≤
        2 ^ (Pooled.leakBits B every units ⟨a + 1, b + 1⟩ ⟨c + 1, c + 1 + e⟩).2.2 ∧
      (Pooled.leakBits B every units ⟨a + 1, b + 1⟩ ⟨c + 1, c + 1 + e⟩).1 =
        B * every + (Pooled.leakBits B every units ⟨a + 1, b + 1⟩ ⟨c + 1, c + 1 + e⟩).2.2

end Properties.Onsite
