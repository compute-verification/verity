module

public import Definitions.Core.Scheme
import Mathlib.Tactic

/-!
# The accounting layer's theorem

`Scheme.accountingLayer` proves `Scheme.AccountingLayer`: the audit's bound at a scheme's accepted runs and incorrect
blocks, with the scheme's break event bounded by its own term, gives what the scheme gets back at every cutoff. An
accepted run that breaks nothing and has `over` above `under` by more than the missable cost plus the charges has incorrect
blocks costing more than the missable cost, so the sample misses them with probability at most the cutoff.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

/-- **The accounting layer's theorem.** -/
theorem Scheme.accountingLayer : Scheme.AccountingLayer := by
  intro S L pr ε hmono hunion haud hbr δ
  unfold Scheme.GetsBack
  refine (hmono _ (fun r => S.breaks r ∨ (S.accepts r ∧ (L r).missProb (S.incorrect r) ≤ δ)) ?_).trans ?_
  · rintro r ⟨ha, hl, hlt⟩
    by_cases hb : S.breaks r
    · exact Or.inl hb
    · refine Or.inr ⟨ha, ?_⟩
      by_contra hmiss
      have hmiss' : δ < (L r).missProb (S.incorrect r) := lt_of_not_ge hmiss
      have hc := S.charge r ha hl hb
      have hsup : ∑ b ∈ S.incorrect r, S.cost b ≤ S.missable (L r) δ := by
        classical
        unfold Scheme.missable
        exact Finset.le_sup (f := fun T => ∑ b ∈ T, S.cost b) (by simp [hmiss'])
      omega
  · calc pr (fun r => S.breaks r ∨ (S.accepts r ∧ (L r).missProb (S.incorrect r) ≤ δ))
        ≤ pr S.breaks + pr (fun r => S.accepts r ∧ (L r).missProb (S.incorrect r) ≤ δ) := hunion _ _
      _ ≤ S.δscheme + (δ + ε) := add_le_add hbr (haud δ)
      _ = δ + ε + S.δscheme := by rw [add_comm]

/-- **A richer run.** The audit's bound read along a map `f` from a scheme's runs to the audit's outcomes, with the
probability that map's, is the audit's bound at the scheme's runs: the scheme's `Run` may carry more than the audit's
outcome (committed values, rows, a log). -/
theorem AuditBoundAt.comap {U Run Run' : Type} {L : Run → SamplingDesign U} {pr' : (Run' → Prop) → ℝ≥0∞}
    (f : Run' → Run) {accepts : Run → Prop} {incorrect : Run → Finset U} {ε : ℝ≥0∞} :
    AuditBoundAt L (fun p => pr' fun r => p (f r)) accepts incorrect ε ↔
      AuditBoundAt (fun r => L (f r)) pr' (fun r => accepts (f r)) (fun r => incorrect (f r)) ε :=
  Iff.rfl

/-- **A richer run, under one design.** -/
theorem AuditBound.comap {U Run Run' : Type} {L : SamplingDesign U} {pr' : (Run' → Prop) → ℝ≥0∞} (f : Run' → Run)
    {accepts : Run → Prop} {incorrect : Run → Finset U} {ε : ℝ≥0∞} :
    AuditBound L (fun p => pr' fun r => p (f r)) accepts incorrect ε ↔
      AuditBound L pr' (fun r => accepts (f r)) (fun r => incorrect (f r)) ε :=
  Iff.rfl

/-- **The accounting layer over a period.** -/
theorem Scheme.periodLayer : Scheme.PeriodLayer := by
  intro S L R T phase pr δ ε hmono hunion hempty hphase
  classical
  unfold Scheme.GetsBackPeriod
  set bad : Fin T → R → Prop := fun t r => S.accepts (phase t r) ∧ S.logOK (phase t r) ∧
    S.under (phase t r) + S.missable (L (phase t r)) δ + S.charges < S.over (phase t r) with hbad
  have key : ∀ s : Finset (Fin T), pr (fun r => ∃ t ∈ s, bad t r) ≤ s.card * (δ + ε + S.δscheme) := by
    intro s
    induction s using Finset.induction_on with
    | empty =>
      refine (hmono _ (fun _ => False) ?_).trans (by simp [hempty])
      rintro r ⟨t, ht, _⟩
      exact absurd ht (Finset.notMem_empty t)
    | insert a s ha ih =>
      refine (hmono _ (fun r => bad a r ∨ ∃ t ∈ s, bad t r) ?_).trans ?_
      · rintro r ⟨t, ht, hb⟩
        rcases Finset.mem_insert.1 ht with rfl | ht
        · exact Or.inl hb
        · exact Or.inr ⟨t, ht, hb⟩
      · refine (hunion _ _).trans ?_
        rw [Finset.card_insert_of_notMem ha, Nat.cast_add, Nat.cast_one, add_mul, one_mul, add_comm]
        exact add_le_add ih (hphase a)
  have h := key Finset.univ
  rw [Finset.card_univ, Fintype.card_fin] at h
  refine (hmono _ _ ?_).trans h
  rintro r ⟨t, hb⟩
  exact ⟨t, Finset.mem_univ t, hb⟩

end Verity
