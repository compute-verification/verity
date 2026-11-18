module

public import Definitions.Core.Scheme
import Mathlib.Tactic

/-!
# The accounting layer's theorem

`Scheme.accountingLayer` proves `Scheme.AccountingLayer`: the audit's bound at a scheme's accepted runs and incorrect
blocks, with the scheme's break event bounded by its own term, gives what the scheme gets back at every cutoff. An
accepted run that breaks nothing and leaves more than the missable cost plus the charges uncredited has incorrect
blocks costing more than the missable cost, so the sample misses them with probability at most the cutoff.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

/-- **The accounting layer's theorem.** -/
theorem Scheme.accountingLayer : Scheme.AccountingLayer := by
  intro S L pr ε hmono hunion haud hbr δ
  unfold Scheme.GetsBack
  refine (hmono _ (fun r => S.breaks r ∨ (S.accepts r ∧ L.missProb (S.incorrect r) ≤ δ)) ?_).trans ?_
  · rintro r ⟨ha, hl, hlt⟩
    by_cases hb : S.breaks r
    · exact Or.inl hb
    · refine Or.inr ⟨ha, ?_⟩
      by_contra hmiss
      have hmiss' : δ < L.missProb (S.incorrect r) := lt_of_not_ge hmiss
      have hc := S.charge r ha hl hb
      have hsup : ∑ b ∈ S.incorrect r, S.cost b ≤ S.missable L δ := by
        classical
        unfold Scheme.missable
        exact Finset.le_sup (f := fun T => ∑ b ∈ T, S.cost b) (by simp [hmiss'])
      omega
  · calc pr (fun r => S.breaks r ∨ (S.accepts r ∧ L.missProb (S.incorrect r) ≤ δ))
        ≤ pr S.breaks + pr (fun r => S.accepts r ∧ L.missProb (S.incorrect r) ≤ δ) := hunion _ _
      _ ≤ S.δscheme + (δ + ε) := add_le_add hbr (haud δ)
      _ = δ + ε + S.δscheme := by rw [add_comm]

end Verity
