import Models.Accounting.Guarantees.ComposePeriod

/-!
# PoUW's credit over an accounting period: proofs

`Accounting.Guarantees.ComposePeriod.PeriodBudget` is proved here: a window's `serviceGame` is one round
`.round d pass (fun _ => .done)`, `periodGame` puts the next window's game in place of its end (`consGame`), and
`RoundGame.BudgetAt` of `.round d pass next` is the round's two conditions and `BudgetAt` of every `next ω`, so the
period's budget is its windows', by induction on the period.

`CircuitWorkPeriod` is stated only. Its proof is `CircuitWork`'s at `n` rounds
(note:compute-accounting/20261019T0739Z-draft-int-period): `luckAt_of_perTile` and `ServiceBudget` in every window at
`shareCone` and `a, b`, then `PeriodBudget`; `budget_tail` once at `t = ln(1/δ_s)`; `DrawnRightOff` at `n` rounds (its
event `n·(δ_bind + δ_ks)`); the period layer's `WriterReads` and `Covers` (`CoversProof.serviceCovers`' argument with a
window index: every position, cut and writer of window `w` is window `w`'s, so a late edge stays in its window); then
`UnitFixed`'s `/n` averages and `work_of_bounds` on the averaged shares.
-/

namespace Accounting.SecurityProofs.ComposePeriod

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

/-- A window's round then the rest has its budget when the round has and every continuation has. -/
theorem budgetAt_consGame {Ω : Type} [Fintype Ω] {n : ℕ} {Λ : ℝ} (g : RoundGame Ω 1) (next : Ω → RoundGame Ω n)
    (hg : RoundGame.BudgetAt Λ g) (hn : ∀ ω, RoundGame.BudgetAt Λ (next ω)) :
    RoundGame.BudgetAt Λ (consGame g next) := by
  cases g with
  | round d pass rest =>
    simp only [consGame, RoundGame.BudgetAt] at hg ⊢
    exact ⟨hg.1, hg.2.1, hn⟩

/-- `PeriodBudget`: by induction on the period, `budgetAt_consGame` at each window. -/
theorem PeriodBudget : Accounting.Guarantees.ComposePeriod.PeriodBudget := by
  intro C Ξ _ _ N M n stg w a b K Λ Wp h
  induction Wp with
  | done => simp only [periodGame, RoundGame.BudgetAt]
  | window W next ih =>
    simp only [Period.EveryWindow] at h
    simp only [periodGame]
    exact budgetAt_consGame _ _ h.1 fun ω => ih ω (h.2 ω)

end Accounting.SecurityProofs.ComposePeriod
