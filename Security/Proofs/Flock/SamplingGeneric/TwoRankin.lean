module

public import Definitions.Claims.SamplingTwoRankin
public import Proofs.Flock.SamplingGeneric.TwoServed

/-!
# Rankin's bound on the two-stage count

`AdmissibleTwoRankin` and `AdmissibleTwoRankinPatterns` (`Definitions.Claims.SamplingTwoRankin`): at a first stage
below independent catches at rates `q_ρ`, the admissible sets' count and their patterns' count are at most `δ^{−θ}`
times a product over the replay blocks.

* `bernoulli_Pa_le`: a Bernoulli first stage `bernoulli:NUM/DEN` on a statement in scope misses each set `T` of replay
  blocks with probability `(1 − NUM/DEN)^{|T|}`, so it meets the rates' hypothesis at `q_ρ = NUM/DEN ≤ 1`.
* `admissibleTwo_rankin`: **`AdmissibleTwoRankin`**.
* `admissibleTwo_rankin_patterns`: **`AdmissibleTwoRankinPatterns`**.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

/-- **A Bernoulli first stage meets the rates' hypothesis**: on a statement in scope whose first stage is
`bernoulli:NUM/DEN`, the first stage misses each set `T` of replay blocks with probability `∏_{ρ∈T} (1 − NUM/DEN)`, and
the rate is at most `1`. -/
theorem bernoulli_Pa_le {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) {num den : ℕ}
    (hb : d.replay.rules = [.bernoulli num den]) :
    (∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ _ρ ∈ T, (1 - (num : ℝ) / den)) ∧
      (num : ℝ) / den ≤ 1 := by
  sorry

/-- **`AdmissibleTwoRankin`**. -/
theorem admissibleTwo_rankin : AdmissibleTwoRankin := by
  sorry

/-- **`AdmissibleTwoRankinPatterns`**. -/
theorem admissibleTwo_rankin_patterns : AdmissibleTwoRankinPatterns := by
  sorry

end Claims.Sampling
