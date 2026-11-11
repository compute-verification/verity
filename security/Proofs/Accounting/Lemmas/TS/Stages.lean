module

public import Definitions.Accounting.Guarantees.TS.Stages

/-!
# Sampled proofs under per-stage coins: lemmas

The statements of `Definitions.Accounting.Guarantees.TS.Stages` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.

* `SampledBudget`: a sampled window with `s ≥ 2` samples a round whose share is at most `r` times the samples' mass on saved
  work, and whose samples are lucky at `p ∈ [1/e, 1)` (`Pr[hit ∧ ok] ≤ p·Pr[hit]`), has `Λ = r/ln(1/p)`
  (`RoundGame.BudgetAt`). The saving may depend on the coins before the samples, the luck's correlation across units is
  arbitrary, and no closure, LIN-STEER, STEER-1 or menu is read.
-/

@[expose] public section

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **The sampled budget.** For every sampled window with `s ≥ 2` samples a round, if every round's share is at most `r`
times the probability that a sample lands on saved work (`SampledWindow.Priced r`) and the samples are lucky at
`p ∈ [1/e, 1)` (`SampledWindow.LuckAt p`), then `Λ = r/ln(1/p)` bounds the window's saving per unit of detection. -/
def SampledBudget : Prop :=
  ∀ {A D : Type} [Fintype A] [Fintype D] {s n : ℕ} (W : SampledWindow A D s n) (p r : ℝ),
    2 ≤ s → Real.exp (-1) ≤ p → p < 1 → 0 < r → W.Priced r → W.LuckAt p →
    RoundGame.BudgetAt (r / Real.log (1 / p)) W.game

end Accounting.Guarantees.TS
