import Definitions.Accounting.TS.Routes
import Definitions.Pouw.RelativeSecurity

/-!
# The two plays: full checking and sampling (trusted)

EPS1-SPEC §2.7 over PoUW's rounds (`Pouw.RelativeSecurity.RoundGame`, `Paths`, `prob`): a window of `S` rounds of
fresh coins on `Ω`, each round one live σ (and, when sampling, the round's draws).

* `Play`: the protocol's acceptance on each path of coins, the cheater's labelled run there, and the ideal game, whose
  round `w` has a share `d_w` and a check `pass`, its rounds' proofs as if every proof were sound. In 𝒫_full `d_w` is
  round `w`'s deficient share of its committed words; in 𝒫_samp(u, s, Q) it is its deficient, wrong and adaptive
  work share, and `pass` is the drawn units' proofs.
* `RoundGame.LuckAt p`: (L) at `p`, one bet a round: every round's share is in `[0, 1]`, and given the past the round
  is deficient and passes with probability at most `p`.
* `RoundGame.BudgetAt Λ`: `Λ` bounds the round's saving per unit of detection (Lemma C's `Λ`): given the past,
  `E[exp(d_w / Λ) · 1{pass}] ≤ 1`. When `d_w` is fixed before the round's draws this reads
  `d_w ≤ Λ · (−ln Pr[pass | past])`, SAMPLING's budget.
* `RoundGame.deficientRounds`: the rounds with a positive share along the realized coins.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **A play** of a window of `S` rounds on coins `Ω`: on each path the protocol's acceptance and the cheater's
labelled run, and the ideal game. -/
structure Play (Ω : Type) (S : ℕ) (K T L U : Type) where
  game : RoundGame Ω S
  accepts : Paths Ω S → Prop
  run : Paths Ω S → Anatomy K T L U

namespace RoundGame

variable {Ω : Type} [Fintype Ω]

/-- **(L) at `p`, one bet a round**: every round's share lies in `[0, 1]`, and at every round, whatever the past, the
round's coins make it deficient and passing with probability at most `p`. -/
def LuckAt (p : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧ prob (fun ω => pass ω ∧ 0 < d ω) ≤ p ∧ ∀ ω, LuckAt p (next ω)

/-- **`Λ` bounds the saving per unit of detection**: at every round, whatever the past, the shares are `≥ 0` and
`E[exp(d_w / Λ) · 1{pass}] ≤ 1` over the round's coins. -/
noncomputable def BudgetAt (Λ : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
      (∀ ω, 0 ≤ d ω) ∧
        (∑ ω, (by classical exact if pass ω then Real.exp (d ω / Λ) else 0)) ≤ (Fintype.card Ω : ℝ) ∧
        ∀ ω, BudgetAt Λ (next ω)

/-- The number of rounds with a positive share along the realized coins. -/
noncomputable def deficientRounds : {n : ℕ} → Pouw.RelativeSecurity.RoundGame Ω n → Paths Ω n → ℕ
  | _, .done, _ => 0
  | _, .round d _ next, x => (by classical exact if 0 < d x.1 then 1 else 0) + deficientRounds (next x.1) x.2

end RoundGame

end Accounting.TS
