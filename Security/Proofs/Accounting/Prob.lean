import Proofs.Pouw.Prob
import Definitions.Pous.Game.Prob

/-!
# A sure-bound guarantee against advice, over the product of the coins

An audit's guarantee in PoUW's shape holds at every value `o` of the advice (the other audits' coins): if its account's
charge is at most `B` on every one of its own coins, it accepts with `B` below the certificate with probability at most
`ε`. Taking `B` the largest charge over its own coins bounds, over both, the probability that it accepts and its
account is short of the certificate on every one of its own coins.

PoUS's and the bandwidth audit's probability (`Pous.pr`, in `ℝ≥0∞`) is the same fraction as PoUW's (`Pouw.pr`, in `ℝ`):
`pr_eq_toReal`.
-/

namespace Accounting

open Pouw

variable {X O : Type} [Fintype X] [Fintype O]

theorem pr_eq_toReal (P : X → Prop) : pr P = (Pous.pr P).toReal := by
  unfold Pouw.pr Pous.pr
  rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem pr_short_le [Nonempty X] [Nonempty O] (acc : X → O → Prop) (m : X → O → ℕ) (c ε : ℝ)
    (h : ∀ (o : O) (B : ℝ), 0 ≤ B → (∀ x, (m x o : ℝ) ≤ B) → pr (fun x => acc x o ∧ B < c) ≤ ε) :
    pr (fun y : O × X => acc y.2 y.1 ∧ ∀ x, (m x y.1 : ℝ) < c) ≤ ε := by
  classical
  have top : ∀ o x, (m x o : ℝ) ≤ ((Finset.univ.sup fun x => m x o : ℕ) : ℝ) := fun o x => by
    exact_mod_cast Finset.le_sup (f := fun x => m x o) (Finset.mem_univ x)
  have hε : 0 ≤ ε := by
    obtain ⟨o⟩ := ‹Nonempty O›
    exact (pr_nonneg _).trans (h o _ (Nat.cast_nonneg _) (top o))
  refine pr_prod_le _ ε hε fun o => ?_
  by_cases hs : ∀ x, (m x o : ℝ) < c
  · obtain ⟨x₀, -, hx₀⟩ := Finset.exists_mem_eq_sup Finset.univ Finset.univ_nonempty fun x => m x o
    refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (h o _ (Nat.cast_nonneg _) (top o))
    rw [hx₀]
    exact hs x₀
  · rw [pr_eq_zero fun x hx => hs hx.2]
    exact hε

end Accounting
