module

public import Definitions.Pouw.Guarantees.TS.Stages

/-! Steps of `Definitions.Pouw.Guarantees.TS.Stages`'s proofs, each a `def G : Prop` proved by the `SecurityProofs`
theorem of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

def SampledBudget : Prop :=
  ∀ {A D : Type} [Fintype A] [Fintype D] {s n : ℕ} (W : SampledExecutionPhase A D s n) (p r : ℝ),
    2 ≤ s → Real.exp (-1) ≤ p → p < 1 → 0 < r → W.Priced r → W.LuckAt p →
    RoundGame.BudgetAt (r / Real.log (1 / p)) W.game

end ResourceAccounting.Guarantees.TS
