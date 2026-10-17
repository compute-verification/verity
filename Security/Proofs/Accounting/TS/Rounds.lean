import Models.Accounting.Guarantees.TS
import Proofs.Pouw.RelativeSecurity

/-!
# The tensor–scalar model's round lemmas

The plays' tail bounds: one bet a round under (L), the deficiency against the deficient rounds, and SAMPLING's budget
(EPS1-SPEC §2.4, §2.7; gap list G7–G9: note:20261017T0801Z-draft-lean-stmt). They are PoUW's round bounds
(`Pouw.RelativeSecurity`): `luck_rounds` conditions on the first round's coins (`prob_eq_expect`, `expect_prod`) as
`expect_weight_le` does, and `budget_tail` is `prob_accepts_le` at `c = 1/Λ`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Accounting.Guarantees.TS Pouw.RelativeSecurity

/-- D-ideal's step 2, one bet a round: under (L) at `p`, the ideal game accepts with at least `k` deficient rounds
with probability at most `p^k`. -/
theorem luck_rounds {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (p : ℝ) (hp : 0 ≤ p)
    (hL : RoundGame.LuckAt p g) (k : ℕ) :
    prob (fun x => g.accepts x ∧ k ≤ Accounting.TS.RoundGame.deficientRounds g x) ≤ p ^ k := by
  sorry

/-- A round's share is at most 1, so the deficiency is at most the number of deficient rounds. -/
theorem deficiency_le_rounds {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (p : ℝ)
    (hL : RoundGame.LuckAt p g) (x : Paths Ω S) :
    g.deficiency x ≤ Accounting.TS.RoundGame.deficientRounds g x := by
  induction g with
  | done => simp [RoundGame.deficiency, Accounting.TS.RoundGame.deficientRounds]
  | round d pass next ih =>
    obtain ⟨hd, -, hnext⟩ := hL
    simp only [RoundGame.deficiency, Accounting.TS.RoundGame.deficientRounds, Nat.cast_add]
    refine add_le_add ?_ (ih x.1 (hnext x.1) x.2)
    split_ifs with h
    · simpa using (hd x.1).2
    · simpa using (not_lt.mp h)

/-- `Λ` bounding the play is PoUW's exponential round bound at `c = 1/Λ`. -/
theorem expBound_of_budgetAt {Ω : Type} [Fintype Ω] {Λ : ℝ} :
    ∀ {n : ℕ} (g : RoundGame Ω n), RoundGame.BudgetAt Λ g → g.ExpBound (1 / Λ)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨-, hsum, hnext⟩ := h
    refine ⟨?_, fun a => expBound_of_budgetAt (next a) (hnext a)⟩
    unfold expect
    rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
    · simp [h0]
    rw [div_le_one (by exact_mod_cast h0)]
    refine le_of_eq_of_le (Finset.sum_congr rfl fun ω _ => ?_) hsum
    dsimp only
    split_ifs
    · rw [one_div_mul_eq_div]
    · rfl

/-- Lemma C's step 1, SAMPLING's budget: when `Λ` bounds the play, the ideal game accepts with deficiency at least
`Λ·t` with probability at most `e^{−t}` (the product over rounds is a supermartingale). -/
theorem budget_tail {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (Λ : ℝ) (hΛ : 0 < Λ)
    (hB : RoundGame.BudgetAt Λ g) (t : ℝ) :
    prob (fun x => g.accepts x ∧ Λ * t ≤ g.deficiency x) ≤ Real.exp (-t) := by
  have h := g.prob_accepts_le (c := 1 / Λ) (by positivity) (expBound_of_budgetAt g hB) (Λ * t)
  rwa [one_div, inv_mul_cancel_left₀ hΛ.ne'] at h

end Accounting.SecurityProofs.TS
