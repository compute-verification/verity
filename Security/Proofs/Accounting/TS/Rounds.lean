import Models.Accounting.Guarantees.TS

/-!
# The tensor–scalar model's round lemmas (stubs)

The plays' tail bounds: one bet a round under (L), the deficiency against the deficient rounds, and SAMPLING's budget
(EPS1-SPEC §2.4, §2.7), each a `sorry` until a prover takes it (gap list G7–G9:
note:20261017T0801Z-draft-lean-stmt).
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
  sorry

/-- Lemma C's step 1, SAMPLING's budget: when `Λ` bounds the play, the ideal game accepts with deficiency at least
`Λ·t` with probability at most `e^{−t}` (the product over rounds is a supermartingale). -/
theorem budget_tail {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (Λ : ℝ) (hΛ : 0 < Λ)
    (hB : RoundGame.BudgetAt Λ g) (t : ℝ) :
    prob (fun x => g.accepts x ∧ Λ * t ≤ g.deficiency x) ≤ Real.exp (-t) := by
  sorry

end Accounting.SecurityProofs.TS
