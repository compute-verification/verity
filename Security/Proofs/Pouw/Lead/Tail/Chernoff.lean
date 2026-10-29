import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Definitions.Pouw.Basic.Prob

/-!
# Sequentially dominated events are rarely many

`count_tail`: over a uniform product of finite coordinates, let each event `E i` read only coordinates `≤ i`, and let
it hold on at most a `p i` share of every cylinder that fixes the coordinates `< i`. Then at least `r` of the events
hold on at most a `(Σ p)^r / r!` share of the coins. This is the multiplicative (Poisson-type) Chernoff bound for
sequentially dominated indicators: `Pr[N ≥ r] ≤ E[C(N, r)] ≤ e_r(p) ≤ (Σ p)^r / r!`, each `r`-set of events
bounded by the product of its `p`s by conditioning on its last event's cylinder.

The lead's coin-branch tail (`Tail.Repeats`) reads it. An additive Hoeffding bound (Mathlib's
`ProbabilityTheory.measure_sum_ge_le_of_iIndepFun`) doesn't serve: its deviation grows as the square root of the
number of blocks, which at T 512 × 64 exceeds the coin-branch slack, and it needs independent blocks, which the lead's
coins don't give (`Tail.Repeats`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Classical in
/-- **Sequentially dominated events are rarely many**: `r` or more of them hold on at most a `(Σ p)^r / r!` share of
the coins. -/
theorem count_tail {B : ℕ} {X : Fin B → Type} [∀ i, Fintype (X i)] (E : Fin B → (∀ i, X i) → Prop)
    (p : Fin B → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hread : ∀ i (ω ω' : ∀ i, X i), (∀ j, j ≤ i → ω j = ω' j) → (E i ω ↔ E i ω'))
    (hdom : ∀ i (ω : ∀ i, X i),
      Pouw.pr (fun ω' : ∀ i, X i => (∀ j, j < i → ω' j = ω j) ∧ E i ω') ≤
        p i * Pouw.pr fun ω' : ∀ i, X i => ∀ j, j < i → ω' j = ω j)
    (r : ℕ) :
    Pouw.pr (fun ω : ∀ i, X i => r ≤ (Finset.univ.filter fun i => E i ω).card) ≤
      (∑ i, p i) ^ r / r.factorial := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
