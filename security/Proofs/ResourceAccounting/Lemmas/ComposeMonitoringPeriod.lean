module

public import Definitions.Compose.Guarantees.ComposeMonitoringPeriod

/-! Steps of `Definitions.Compose.Guarantees.ComposeMonitoringPeriod`'s proofs, each a `def G : Prop` proved by the
`SecurityProofs` theorem of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.ComposeMonitoringPeriod

open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity

def MonitoringPeriodBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M n : ℕ} (stg : Fin N → ℕ) (w : (Fin N → Bool) → Fin N → ℝ) (a b : ℝ)
    (K : ℕ) (Λ : ℝ) (Wp : ExecutionPhasePlay C (Paths C M × (Equiv.Perm (Fin N) × Ξ)) (Fin N) M n),
    Wp.EveryExecutionPhase (fun live W => RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W (w live) a b K)) →
    RoundGame.BudgetAt Λ (monitoringPeriodGame stg w a b K Wp)

/-- A period credited up to its first failed window, a stopping time the cheater may steer, reads it at that window,
so it pays `ln(1/δ_s)` once, and needs neither a fixed `n` nor a rule against dropping windows. -/
def MonitoringPeriodTailMax : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {n : ℕ} (g : RoundGame Ω n) (Λ : ℝ), 0 < Λ → RoundGame.BudgetAt Λ g → ∀ t : ℝ,
    prob (fun x => ∃ j : ℕ, RoundGame.acceptsUpTo g j x ∧ Λ * t ≤ RoundGame.deficiencyUpTo g j x) ≤ Real.exp (-t)

end ResourceAccounting.Guarantees.ComposeMonitoringPeriod
