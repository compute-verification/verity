import Proofs.Pouw.Prob

/-!
# Truncation: a sure-cost guarantee over a class closed under budget-abort is per-run

Fix an adversary `a` with a nonnegative realized cost. Suppose every adversary of a class whose cost is surely at most
`T ≥ 0` is accepted with `T < c` with probability at most `ε`, and that for every budget `b ≥ 0` the class has an
adversary within `b` that is accepted wherever `a` is accepted at cost at most `b`. Then `a` is accepted at a realized
cost below `c` with probability at most `ε`: the coins are finite, so the costs below `c` have a largest, `b < c`, and
the adversary truncated at `b` is accepted on every such run.
-/

namespace Accounting

open Pouw

variable {Ω A : Type} [Fintype Ω]

/-- The realized costs below `c > 0` of an adversary with nonnegative costs on finitely many coins are at most one
budget `b` with `0 ≤ b < c`. -/
theorem exists_budget (cost : Ω → ℝ) (hnn : ∀ ω, 0 ≤ cost ω) {c : ℝ} (hc : 0 < c) :
    ∃ b, 0 ≤ b ∧ b < c ∧ ∀ ω, cost ω < c → cost ω ≤ b := by
  classical
  by_cases hs : (Finset.univ.filter fun ω => cost ω < c).Nonempty
  · obtain ⟨ω₀, h₀, hmax⟩ := Finset.exists_max_image _ cost hs
    refine ⟨cost ω₀, hnn ω₀, (Finset.mem_filter.1 h₀).2, fun ω hω => hmax ω ?_⟩
    simpa using hω
  · refine ⟨0, le_rfl, hc, fun ω hω => absurd ⟨ω, by simpa using hω⟩ hs⟩

/-- **Truncation.** A guarantee on every adversary within a sure budget (`hsure`) bounds the probability that `a` is
accepted at a realized cost below `c`, when the class has, for every budget `b ≥ 0`, an adversary within `b` accepted
wherever `a` is accepted at cost at most `b` (`habort`). -/
theorem pr_lt_of_abort (acc : A → Ω → Prop) (Bdd : A → ℝ → Prop) (a : A) (cost : Ω → ℝ) (c ε : ℝ)
    (hnn : ∀ ω, 0 ≤ cost ω)
    (hsure : ∀ a' T, 0 ≤ T → Bdd a' T → pr (fun ω => acc a' ω ∧ T < c) ≤ ε)
    (habort : ∀ b, 0 ≤ b → ∃ a', Bdd a' b ∧ ∀ ω, cost ω ≤ b → acc a ω → acc a' ω) :
    pr (fun ω => acc a ω ∧ cost ω < c) ≤ ε := by
  by_cases hc : 0 < c
  · obtain ⟨b, hb0, hbc, hb⟩ := exists_budget cost hnn hc
    obtain ⟨a', ha', hacc⟩ := habort b hb0
    exact (pr_mono fun ω h => ⟨hacc ω (hb ω h.2) h.1, hbc⟩).trans (hsure a' b hb0 ha')
  · obtain ⟨a', ha', -⟩ := habort 0 le_rfl
    rw [pr_eq_zero fun ω h => hc ((hnn ω).trans_lt h.2)]
    exact (pr_nonneg _).trans (hsure a' 0 le_rfl ha')

end Accounting
