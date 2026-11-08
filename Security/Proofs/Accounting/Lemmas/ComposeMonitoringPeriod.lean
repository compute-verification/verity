module

public import Definitions.Accounting.Guarantees.ComposeMonitoringPeriod

/-!
# PoUW's work over a monitoring period of `n` windows: lemmas

The statements of `Definitions.Accounting.Guarantees.ComposeMonitoringPeriod` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.

* `PeriodBudget`: the composition lemma. If every window's `serviceGame` has its saving per unit of detection bounded by
  `Λ` (`RoundGame.BudgetAt`, which `ServiceBudget` gives per window), whatever the earlier windows did, so does the
  period's game (`periodGame`). Proved (`Proofs.Accounting.ComposeMonitoringPeriod`): a window's game is one round, and
  `BudgetAt` over rounds is that condition at each round given the past.
* `PeriodTailMax`: `budget_tail` at every prefix at once (Ville's inequality, finite form): when `Λ` bounds the play,
  some prefix of its rounds is accepted with share at least `Λ·t` with probability at most `e^{−t}`. So a prefix
  that ends at a stopping time the cheater steers pays `ln(1/δ_s)` once, as a fixed one does. Proved
  (`Proofs.Accounting.ComposeMonitoringPeriod`): by induction on the game, conditioning on the first round's coins as
  `expect_weight_le` does, with the bound for every threshold `D` in the induction.
-/

@[expose] public section

namespace Accounting.Guarantees.ComposeMonitoringPeriod

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The period's budget from its windows'.** For a period of `n` windows of one circuit of `M` rounds over `N` tiles,
any play (`WindowPlay`: each window's live tiles and split stages chosen after the earlier windows' coins), share
weights `w live` at each window's live tiles `live`, costs `a, b` and `K` samples a window: if every window's one round
(`serviceGame`) has its saving per unit of detection bounded by `Λ` (`RoundGame.BudgetAt`), whatever the past, then so
has the period's game (`periodGame`). With `budget_tail` at `n` rounds, an accepted period's total share is below
`Λ·ln(1/δ_s)` except with probability `δ_s`. -/
def PeriodBudget : Prop :=
  ∀ {C Ξ : Type} [Fintype C] [Fintype Ξ] {N M n : ℕ} (stg : Fin N → ℕ) (w : (Fin N → Bool) → Fin N → ℝ) (a b : ℝ)
    (K : ℕ) (Λ : ℝ) (Wp : WindowPlay C (Paths C M × (Equiv.Perm (Fin N) × Ξ)) (Fin N) M n),
    Wp.EveryWindow (fun live W => RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) stg W (w live) a b K)) →
    RoundGame.BudgetAt Λ (periodGame stg w a b K Wp)

/-- **The budget's tail at every prefix at once** (Ville's inequality, finite form): for any play of `n` rounds on
coins `Ω` with its saving per unit of detection bounded by `Λ > 0` (`RoundGame.BudgetAt`), the probability that some
prefix of its rounds passes (`RoundGame.acceptsUpTo`) with share at least `Λ·t` (`RoundGame.deficiencyUpTo`) is at most
`e^{−t}`. Its last prefix is `budget_tail`. A period credited up to its first failed window, a stopping time the
cheater may steer, reads it at that window, so it pays `ln(1/δ_s)` once, and neither a fixed `n` nor a rule against
dropping windows is needed for the tail. -/
def PeriodTailMax : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {n : ℕ} (g : RoundGame Ω n) (Λ : ℝ), 0 < Λ → RoundGame.BudgetAt Λ g → ∀ t : ℝ,
    prob (fun x => ∃ j : ℕ, RoundGame.acceptsUpTo g j x ∧ Λ * t ≤ RoundGame.deficiencyUpTo g j x) ≤ Real.exp (-t)

end Accounting.Guarantees.ComposeMonitoringPeriod
