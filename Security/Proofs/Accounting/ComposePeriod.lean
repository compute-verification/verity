import Models.Accounting.Guarantees.ComposePeriod

/-!
# PoUW's work over an accounting period: proofs

`Accounting.Guarantees.ComposePeriod.PeriodBudget` is proved here: a window's `serviceGame` is one round
`.round d pass (fun _ => .done)`, `periodGame` puts the next window's game in place of its end (`consGame`), and
`RoundGame.BudgetAt` of `.round d pass next` is the round's two conditions and `BudgetAt` of every `next ω`, so the
period's budget is its windows', by induction on the period. The window's live tiles only pick its weights.

`CircuitWorkPeriod` is stated only. Its proof is `CircuitWork`'s at `n` rounds
(note:compute-accounting/20261019T0739Z-draft-int-period), on live tiles:
* per window, `ServiceBudget` at the detected live weights (`0` on a tile of no live words, else `liveCone`), which
  are at most `shareCone`, so `CircuitWork`'s `ρ`, stage-share and luck premises hold of them; its `LuckAt` from
  `PerLiveTile` (`luckAt_of_perTile` with luck `1` on a dead tile, of weight `0`); then `PeriodBudget`;
* `budget_tail` once at `t = ln(1/δ_s)`; `DrawnRightOff` at `n` rounds (its event `n·(δ_bind + δ_ks)`);
* the period layer's `WriterReads` and its `Covers` on live tiles (`CoversProof.serviceCovers`' argument with a window
  index: every position, cut and writer of window `w` is window `w`'s, so a late edge stays in its window). A live
  tile of words that isn't good under `periodMode` is reached by a skipped tile, which `DeadConeDead` makes live, so
  the game charges it in that tile's `liveCone`; the live tiles of no words carry at most `zeroCone`;
* `UnitFixed`'s `/n` averages, where `periodMode`'s good tiles are live and their words sum to `n·periodLive` less the
  covered ones, then `work_of_bounds` on the averaged shares, with `periodLive` in place of `N_w`.
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
  | window live W next ih =>
    simp only [Period.EveryWindow] at h
    simp only [periodGame]
    exact budgetAt_consGame _ _ h.1 fun ω => ih ω (h.2 ω)

end Accounting.SecurityProofs.ComposePeriod
