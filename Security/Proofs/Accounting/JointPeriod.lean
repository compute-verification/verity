import Definitions.Accounting.Guarantees.JointPeriod
import Proofs.Pouw.Copy

/-!
# Proof of the joint guarantee over a period and over a horizon of periods

An audit's per-period guarantee (`PeriodRun.Sound`) bounds its event over its own coordinate of the coins at every
value of the others. Splitting the coins at that coordinate (`Equiv.piSplitAt`) and averaging over the others
(`pr_prod_le`, read through `pr_equiv`, as `Run.pr_computeLt_le` does for a run's three audits) bounds it over all of
them (`pr_le_of_own`, `PeriodRun.pr_bad_le`). A union bound over the five audits sums their errors (`JointPeriod`), and
over a horizon the union runs over `Fin m × Audit`, every period's audits (`JointHorizon`).
-/

namespace Accounting

open Pouw

/-- Updating coordinate `i` of the coins `(x₀, g)`, split at `i`, to `x` gives `(x, g)`. -/
theorem update_piSplitAt_symm {I : Type} [DecidableEq I] {Ω : I → Type} (i : I) (x₀ x : Ω i)
    (g : (j : { j // j ≠ i }) → Ω j) :
    Function.update ((Equiv.piSplitAt i Ω).symm (x₀, g)) i x = (Equiv.piSplitAt i Ω).symm (x, g) := by
  funext j
  by_cases hj : j = i
  · subst hj
    simp [Equiv.piSplitAt_symm_apply]
  · simp [Function.update_of_ne hj, Equiv.piSplitAt_symm_apply, hj]

/-- **A bound over one coordinate bounds the event.** If, at every value of the other coins, the probability over
coordinate `i` of `E` is at most `δ`, so is the probability over all the coins. -/
theorem pr_le_of_own {I : Type} [Fintype I] [DecidableEq I] {Ω : I → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (i : I) (E : ((j : I) → Ω j) → Prop) {δ : ℝ}
    (h : ∀ ξ, pr (fun x : Ω i => E (Function.update ξ i x)) ≤ δ) : pr E ≤ δ := by
  obtain ⟨x₀⟩ : Nonempty (Ω i) := inferInstance
  let e : ((j : I) → Ω j) ≃ ((j : { j // j ≠ i }) → Ω j) × Ω i :=
    (Equiv.piSplitAt i Ω).trans (Equiv.prodComm _ _)
  have key := pr_prod_le (fun y => E (e.symm y)) δ ((pr_nonneg _).trans (h (Classical.arbitrary _))) fun g => by
    have hg : ∀ x, e.symm (g, x) = Function.update ((Equiv.piSplitAt i Ω).symm (x₀, g)) i x := fun x => by
      rw [update_piSplitAt_symm]
      rfl
    simpa only [hg] using h ((Equiv.piSplitAt i Ω).symm (x₀, g))
  rw [← pr_equiv e] at key
  simpa only [Equiv.symm_apply_apply] using key

/-- The union bound over a finite family of events. -/
theorem pr_exists_le_sum_univ {X K : Type} [Fintype X] [Fintype K] (Bad : K → X → Prop) :
    pr (fun x => ∃ k, Bad k x) ≤ ∑ k, pr (Bad k) :=
  (pr_mono fun _ ⟨k, hk⟩ => ⟨k, Finset.mem_univ k, hk⟩).trans (Pouw.Proofs.pr_exists_le_sum Finset.univ Bad)

namespace PeriodRun

/-- **An audit's per-period guarantee bounds its event over all the period's coins**: it accepts while its claim
fails with probability at most its error. -/
theorem pr_bad_le {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} [Fintype I]
    [DecidableEq I] {Ω : I → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (r : PeriodRun d α C Ω)
    {c : Certificates} {p : Price} {a : Audit} {δ : ℝ} (h : r.Sound c p a δ) :
    pr (fun ξ => r.accepts a ξ ∧ ¬ r.Holds c p a ξ) ≤ δ :=
  pr_le_of_own (r.own a) _ h

end PeriodRun

namespace SecurityProofs

/-- **The joint guarantee over a period** (`Accounting.Guarantees.JointPeriod`). -/
theorem JointPeriod : Accounting.Guarantees.JointPeriod := by
  intro ι α I _ _ _ Ω _ _ d C r c p δ h
  exact (pr_exists_le_sum_univ _).trans (Finset.sum_le_sum fun a _ => r.pr_bad_le (h a))

/-- **The joint guarantee over a horizon of periods** (`Accounting.Guarantees.JointHorizon`). -/
theorem JointHorizon : Accounting.Guarantees.JointHorizon := by
  intro ι α I _ _ _ Ω _ _ d C m r c p δ h
  calc pr (fun ξ => ∃ j a, (r j).accepts a ξ ∧ ¬ (r j).Holds (c j) p a ξ)
      ≤ pr (fun ξ => ∃ q : Fin m × Audit, (r q.1).accepts q.2 ξ ∧ ¬ (r q.1).Holds (c q.1) p q.2 ξ) :=
        pr_mono fun _ ⟨j, a, hb⟩ => ⟨(j, a), hb⟩
    _ ≤ ∑ q : Fin m × Audit, δ q.1 q.2 :=
        (pr_exists_le_sum_univ _).trans (Finset.sum_le_sum fun q _ => (r q.1).pr_bad_le (h q.1 q.2))
    _ = ∑ j, ∑ a, δ j a := Fintype.sum_prod_type _

end SecurityProofs

end Accounting
