import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Models.Accounting.Assumptions.WipePool

/-!
# The memory cap's lifetime pool (trusted)

Step 7 of the memory-cap restatement ("One delta"): one `δ` for a prover's whole life, by Ville's inequality over its
wipes. Wipe `t` fixes its un-held fraction `x_t ∈ [0, 1]` before its fresh coins, adaptively on everything before
them, and each of its `k` timed rounds catches it with probability at least `x_t` (`Assumptions.TimedRoundSound`), so it
passes with probability at most `(1 − x_t)^k ≤ exp(−k x_t)`.

* `WipePool`: `Pr[every wipe passes ∧ Σ_t x_t ≥ H] ≤ exp(−k H)`.
* `WipePoolDelta`: at `H = ln(1/δ)/k`, the pool holds except with probability `δ`: outside an event of probability at
  most `δ`, some wipe fails or the un-held fractions sum to less than `ln(1/δ)/k`.

Neither bound depends on the number of wipes: the run tree (`Accounting.WipePlay`) is finite, of any depth. At `k = 1`
each wipe is one read with its own un-held set, which gives the pool per read, adaptive within a wipe as well.
-/

namespace Accounting.Guarantees.Actual

/-- **The lifetime pool.** For any prover strategy over wipes of `k` timed rounds (`P : WipePlay k`, any finite number
of wipes, each fixing its un-held set before its coins) meeting timed-round soundness, and any `H`, the probability that
every wipe passes and the un-held fractions sum to at least `H` is at most `exp(−k H)`. -/
def WipePool : Prop :=
  ∀ (k : ℕ) (P : WipePlay k) (H : ℝ), Assumptions.TimedRoundSound P →
    P.prEv (fun x => H ≤ x) 0 ≤ ENNReal.ofReal (Real.exp (-((k : ℝ) * H)))

/-- **The lifetime pool at `δ`.** For `k > 0` rounds a wipe and `δ > 0`, under timed-round soundness, the probability
that every wipe passes and the un-held fractions sum to at least `ln(1/δ)/k` is at most `δ`. -/
def WipePoolDelta : Prop :=
  ∀ (k : ℕ) (P : WipePlay k) (δ : ℝ), 0 < k → 0 < δ → Assumptions.TimedRoundSound P →
    P.prEv (fun x => Real.log (1 / δ) / k ≤ x) 0 ≤ ENNReal.ofReal δ

end Accounting.Guarantees.Actual
