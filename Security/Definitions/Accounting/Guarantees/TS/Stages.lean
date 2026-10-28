import Definitions.Accounting.Guarantees.TS
import Definitions.Accounting.Assumptions.Stages

/-!
# Sampled proofs under per-stage coins (trusted)

STEER-PROOF's route A, each `def G : Prop` proved as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS.Stages`:

* `SampledBudget`: a sampled window with `s ≥ 2` draws a round whose share is at most `r` times the draws' mass on saved
  work, and whose draws are lucky at `p ∈ [1/e, 1)` (`Pr[hit ∧ ok] ≤ p·Pr[hit]`), has `Λ = r/ln(1/p)`
  (`RoundGame.BudgetAt`). The saving may depend on the coins before the draws, the luck's correlation across units is
  arbitrary, and no closure, LIN-STEER, STEER-1 or menu is read.
* `StageBudget`: the same for per-stage coins: under `StageLuck` at `p` and work weights at most `r` times the draw law,
  `Λ = r/ln(1/p)`.

`s ≥ 2` is needed: at `s = 1` and `Λ = 1/ln(1/p)`, skipping a small draw mass `m` whose units are lucky together with
probability `p` gives `E[e^{d/Λ}·1{pass}] = p^{−m}·(1 − (1 − p)·m) > 1`. PoUW's staged bound
(`Pouw.RelativeSecurity.RoundGame.Staged`) covers one draw a round, at `Λ = 1/(1 − p)`, for skips fixed before the
draw.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **The sampled budget.** For every sampled window with `s ≥ 2` draws a round, if every round's share is at most `r`
times the probability that a draw lands on saved work (`SampledWindow.Priced r`) and the draws are lucky at
`p ∈ [1/e, 1)` (`SampledWindow.LuckAt p`), then `Λ = r/ln(1/p)` bounds the window's saving per unit of detection. -/
def SampledBudget : Prop :=
  ∀ {A D : Type} [Fintype A] [Fintype D] {s n : ℕ} (W : SampledWindow A D s n) (p r : ℝ),
    2 ≤ s → Real.exp (-1) ≤ p → p < 1 → 0 < r → W.Priced r → W.LuckAt p →
    RoundGame.BudgetAt (r / Real.log (1 / p)) W.game

/-- **The budget under per-stage coins.** For every window of rounds of `M` stages with per-stage coins and `s ≥ 2`
draws a round over whole units (tickets `tk`, any law with rational weights), with work weights `w ≥ 0` at most `r`
times each unit's draw probability, (L) at `p ∈ [1/e, 1)` per unit at its stage's coin (`StageLuck`) gives
`Λ = r/ln(1/p)`. -/
def StageBudget : Prop :=
  ∀ {C V D : Type} [Fintype C] [Fintype V] [Fintype D] {M s n : ℕ} (W : StageWindow C V D M s n) (stg : V → ℕ)
    (tk : D → V) (w : V → ℝ) (p r : ℝ),
    2 ≤ s → Real.exp (-1) ≤ p → p < 1 → 0 < r → (∀ u, 0 ≤ w u) → (∀ u, w u ≤ r * prob (fun t => tk t = u)) →
    StageLuck p stg W → RoundGame.BudgetAt (r / Real.log (1 / p)) (W.game stg tk w)

end Accounting.Guarantees.TS
