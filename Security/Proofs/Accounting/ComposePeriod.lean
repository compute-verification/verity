import Models.Accounting.Guarantees.ComposePeriod
import Proofs.Accounting.TS.Rounds

/-!
# PoUW's work over an accounting period: proofs

`Accounting.Guarantees.ComposePeriod.PeriodBudget` is proved here: a window's `serviceGame` is one round
`.round d pass (fun _ => .done)`, `periodGame` puts the next window's game in place of its end (`consGame`), and
`RoundGame.BudgetAt` of `.round d pass next` is the round's two conditions and `BudgetAt` of every `next ω`, so the
period's budget is its windows', by induction on the period. The window's live tiles only pick its weights.

`Accounting.Guarantees.ComposePeriod.PeriodTailMax` is proved here too, by induction on the game for every threshold
`D` at once (`prob_prefix_le`): a prefix reaching `D > 0` passes its first round and has the rest reach `D − d`, so,
conditioning on the first round's coins as `expect_weight_le` does, its probability is at most
`E[1{pass}·e^{c·d}]·e^{−c·D} ≤ e^{−c·D}`. At `c = 1/Λ` the round bound is `BudgetAt`'s (`expBound_of_budgetAt`).

`CircuitWorkPeriod` is stated only. Its proof is `CircuitWork`'s at `n` rounds
(note:compute-accounting/20261019T0739Z-draft-int-period), on the accepted prefix's live tiles:
* per window, `ServiceBudget` at the detected live weights (`0` on a tile of no live words, else `liveCone`), which
  are at most `shareCone`, so `CircuitWork`'s `ρ`, stage-share and luck premises hold of them; its `LuckAt` from
  `PerLiveTile` (`luckAt_of_perTile` with luck `1` on a dead tile, of weight `0`); then `PeriodBudget`;
* `PeriodTailMax` once at `t = ln(1/δ_s)`, read at the prefix `j = prefixLen`; `DrawnRightOff` at `n` rounds (its event
  `n·(δ_bind + δ_ks)`), off which every drawn tile of a credited window is right at its committed values, so
  `PrefixOf` passes the game's first `j` rounds;
* the period layer's `WriterReads` and its `Covers` on the credited windows' live tiles (`CoversProof.serviceCovers`'
  argument with a window index: every position, cut and writer of window `w` is window `w`'s, so a late edge stays in
  its window, and a credited window's tiles have the modes of its own run). A live tile of words that isn't good
  under `periodMode` is reached by a skipped tile, which `DeadConeDead` makes live, so the game charges it in that
  tile's `liveCone` in the first `j` rounds; the live tiles of no words carry at most `zeroCone` a window;
* `UnitFixed`'s sums, where `periodMode`'s good tiles are live in credited windows and their words sum to
  `creditedLive` less the covered ones, then `work_of_bounds` on the prefix's total shares.
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

/-- **The exponential round bound at every prefix**: for `c ≥ 0`, some prefix of the play passes with share at least
`D` with probability at most `e^{−c·D}`. -/
theorem prob_prefix_le {Ω : Type} [Fintype Ω] {c : ℝ} (hc : 0 ≤ c) :
    ∀ {n : ℕ} (G : RoundGame Ω n), G.ExpBound c → ∀ D : ℝ,
      prob (fun x => ∃ j, RoundGame.acceptsUpTo G j x ∧ D ≤ RoundGame.deficiencyUpTo G j x) ≤ Real.exp (-(c * D))
  | _, .done, _, D => by
    rcases le_or_gt D 0 with hD | hD
    · exact (prob_le_one _).trans (Real.one_le_exp (neg_nonneg.mpr (mul_nonpos_of_nonneg_of_nonpos hc hD)))
    refine (prob_mono (q := fun _ => False) fun x hx => ?_).trans
      (Accounting.SecurityProofs.TS.Rounds.prob_false.trans_le (Real.exp_pos _).le)
    obtain ⟨j, -, hj⟩ := hx
    simp only [RoundGame.deficiencyUpTo] at hj
    linarith
  | _, .round d pass next, h, D => by
    classical
    obtain ⟨h1, h2⟩ := h
    rcases le_or_gt D 0 with hD | hD
    · exact (prob_le_one _).trans (Real.one_le_exp (neg_nonneg.mpr (mul_nonpos_of_nonneg_of_nonpos hc hD)))
    rw [prob_eq_expect, expect_prod]
    calc _ ≤ expect (fun a => Real.exp (-(c * D)) * if pass a then Real.exp (c * d a) else 0) :=
          expect_mono fun a => ?_
      _ = Real.exp (-(c * D)) * expect (fun a => if pass a then Real.exp (c * d a) else 0) :=
          expect_mul_left _ _
      _ ≤ Real.exp (-(c * D)) * 1 := mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le
      _ = Real.exp (-(c * D)) := mul_one _
    by_cases hpa : pass a
    · rw [ite_eq_left hpa]
      refine (Accounting.SecurityProofs.TS.Rounds.expect_ind_le_prob (Q := fun y => ∃ j,
        RoundGame.acceptsUpTo (next a) j y ∧ D - d a ≤ RoundGame.deficiencyUpTo (next a) j y)
        fun y hy => ?_).trans ?_
      · obtain ⟨j, hj1, hj2⟩ := hy
        cases j with
        | zero =>
          simp only [RoundGame.deficiencyUpTo] at hj2
          linarith
        | succ j =>
          simp only [RoundGame.acceptsUpTo, RoundGame.deficiencyUpTo] at hj1 hj2
          exact ⟨j, hj1.2, by linarith⟩
      · calc _ ≤ Real.exp (-(c * (D - d a))) := prob_prefix_le hc (next a) (h2 a) (D - d a)
          _ = Real.exp (-(c * D)) * Real.exp (c * d a) := by rw [← Real.exp_add]; ring_nf
    · rw [ite_eq_right hpa, mul_zero]
      refine (Accounting.SecurityProofs.TS.Rounds.expect_ind_le_prob (Q := fun _ => False) fun y hy => ?_).trans
        Accounting.SecurityProofs.TS.Rounds.prob_false.le
      obtain ⟨j, hj1, hj2⟩ := hy
      cases j with
      | zero =>
        simp only [RoundGame.deficiencyUpTo] at hj2
        linarith
      | succ j =>
        simp only [RoundGame.acceptsUpTo] at hj1
        exact hpa hj1.1

/-- `PeriodTailMax`: `prob_prefix_le` at `c = 1/Λ`, whose round bound `BudgetAt` gives. -/
theorem PeriodTailMax : Accounting.Guarantees.ComposePeriod.PeriodTailMax := by
  intro Ω _ n g Λ hΛ hB t
  have h := prob_prefix_le (c := 1 / Λ) (by positivity) g
    (Accounting.SecurityProofs.TS.Rounds.expBound_of_budgetAt g hB) (Λ * t)
  rwa [one_div, inv_mul_cancel_left₀ hΛ.ne'] at h

end Accounting.SecurityProofs.ComposePeriod
