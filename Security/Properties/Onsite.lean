import CoreOnsite.Pooled
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Onsite's pooled figures meet the arithmetic of the statements they evaluate (`CoreOnsite.Pooled`)

`onsite bound` prints each resource's pooled figure over a monitoring period, computed by `Core.Onsite.Pooled` in exact naturals
with `ln` a certified bracket. Each property here says the figure satisfies the arithmetic the statement it evaluates
takes as its hypothesis, over the reals, so that statement applies at the figure:

* `LnBrackets`: `ln x`'s two bounds over `2^PREC` are below and above `Real.log x`, for a ratio `x ≥ 1`.
* `IntegrityHorizonCovers`: `faultyBlocks q δ` is at least every total `w` of faulty proof blocks outside the event
  `Flock.Guarantees.IntegrityHorizon` bounds (`ln(1/δ)/q ≤ w`), for `q > 0` and `0 < δ ≤ 1`.
* `IntegrityPeriodCovers`: at `threshold n kd e`, when it is at most `n`, the sampling term
  `C(n − K₀, kd)/C(n, kd)` that `Flock.Guarantees.IntegrityPeriod` sums for a block is at most `e`.
* `PooledShortCovers`: `shortCount U p δ'`, when it is at most `U`, meets `Pous.Period.SecurityProofs.pooledShort_log`'s
  hypothesis `((z − 1)·U·p + ln(1/δ'))/ln z ≤ M` at some `z > 1`, so at least `U − M + 1` of `U` sampling periods
  certify (`certified`), `p` the per-period `δ + ε_max` PoUS publishes.
* `HorizonLeakCovers`: `leakBits`' `Fmax` meets `NetTiming.Guarantees.HorizonLeakAccepted`'s `hF`,
  `ln(1/δ) ≤ q·Fmax`; its faulty proof blocks' bits bound that statement's factor `(N + 1)^Fmax · 2^(b·Fmax)` at
  `b = BLOCK_BITS` for every `N ≤ B·units`; and its `K` is the blocks' own terms plus those bits.

`IntegrityHorizon`, `IntegrityPeriod` and `pooledShort_log` are in the proofs tree, which a property may not read, so
their arithmetic is restated here as each statement writes it rather than named. They take no premise. That the
figures' inputs (`q`, `n`, `kd`, `p`, `units`, `every`) are the subprotocols' published values is `CoreOnsite.Bound`'s
reading of the accepted answers, outside these statements.
-/

namespace Properties.Onsite

open Core Core.Onsite

/-- **`ln` brackets the logarithm.** For a ratio `x ≥ 1`, `lo/2^PREC ≤ log x ≤ hi/2^PREC`, `(lo, hi) = ln x`. -/
def LnBrackets : Prop :=
  ∀ x : Ratio, 0 < x.den → x.den ≤ x.num →
    ((Pooled.ln x).1 : ℝ) / 2 ^ Pooled.PREC ≤ Real.log ((x.num : ℝ) / x.den) ∧
      Real.log ((x.num : ℝ) / x.den) ≤ ((Pooled.ln x).2 : ℝ) / 2 ^ Pooled.PREC

/-- **`IntegrityHorizon`'s figure.** For a draw covering at least a fraction `q > 0` of each block's proof blocks and
`0 < δ ≤ 1`, every total `w` of faulty proof blocks with `¬ ln(1/δ)/q ≤ w`, outside the event `IntegrityHorizon`
bounds, is at most `faultyBlocks q δ`. -/
def IntegrityHorizonCovers : Prop :=
  ∀ q δ : Ratio, 0 < q.num → 0 < q.den → 0 < δ.num → δ.num ≤ δ.den → ∀ w : ℕ,
    ¬ Real.log (1 / ((δ.num : ℝ) / δ.den)) / ((q.num : ℝ) / q.den) ≤ w → w ≤ Pooled.faultyBlocks q δ

/-- **`IntegrityPeriod`'s threshold.** When `threshold n kd e` is at most `n`, a block of `n` proof blocks drawing `kd` has
sampling term `C(n − K₀, kd)/C(n, kd)` at most `e` at `K₀ = threshold n kd e`. -/
def IntegrityPeriodCovers : Prop :=
  ∀ (n kd : ℕ) (e : Ratio), 0 < e.den → Pooled.threshold n kd e ≤ n →
    ((n - Pooled.threshold n kd e).choose kd : ℝ) / (n.choose kd : ℝ) ≤ (e.num : ℝ) / e.den

/-- **`pooledShort_log`'s count.** When `shortCount U p δ'` is at most `U`, it meets `pooledShort_log`'s hypothesis
`((z − 1)·U·p + ln(1/δ'))/ln z ≤ M` at some `z > 1`, for `0 < δ' ≤ 1`. -/
def PooledShortCovers : Prop :=
  ∀ (U : ℕ) (p δ : Ratio), 0 < p.den → 0 < δ.num → δ.num ≤ δ.den → Pooled.shortCount U p δ ≤ U →
    ∃ z : ℝ, 1 < z ∧
      ((z - 1) * U * ((p.num : ℝ) / p.den) + Real.log (1 / ((δ.num : ℝ) / δ.den))) / Real.log z ≤
        Pooled.shortCount U p δ

/-- **`HorizonLeakAccepted`'s count.** `leakBits B every units q δ = (K, Fmax, faulty)` has `ln(1/δ) ≤ q·Fmax` (the
statement's `hF`); `faulty` bits cover the statement's faulty-block factor `(N + 1)^Fmax · 2^(b·Fmax)` at
`b = BLOCK_BITS`, for every `N ≤ B·units`; and `K = B·every + faulty`. -/
def HorizonLeakCovers : Prop :=
  ∀ (B every units : ℕ) (q δ : Ratio), 0 < q.num → 0 < q.den → 0 < δ.num → δ.num ≤ δ.den →
    Real.log (1 / ((δ.num : ℝ) / δ.den)) ≤ ((q.num : ℝ) / q.den) * (Pooled.leakBits B every units q δ).2.1 ∧
      (∀ N ≤ B * units, (N + 1) ^ (Pooled.leakBits B every units q δ).2.1 *
        2 ^ (Pooled.BLOCK_BITS * (Pooled.leakBits B every units q δ).2.1) ≤
          2 ^ (Pooled.leakBits B every units q δ).2.2) ∧
      (Pooled.leakBits B every units q δ).1 = B * every + (Pooled.leakBits B every units q δ).2.2

end Properties.Onsite
