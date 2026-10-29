import Proofs.Pouw.Lead.Tail.Chernoff

/-!
# Sequentially dominated events, one coordinate at a time

`count_tail_fiber` is `count_tail` with its domination read off one coordinate: if event `i` reads only coordinates
`≤ i`, and whatever the other coordinates, at most a `p i` share of coordinate `i`'s values make it hold, then at
least `r` of the events hold on at most a `(Σ p)^r / r!` share of the coins.

On a cylinder that fixes the coordinates `< i`, event `i` reads only coordinate `i` among the free ones, so the
cylinder's points where it holds are, over each value of the coordinates `> i`, at most `p i · |X i|` of coordinate
`i`'s values: `count_tail`'s `hdom`. The lead's three counts (`Tail.across_tail`, `Tail.columns_tail`,
`Tail.slots_tail`) read it, each with its own fiber count.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Classical in
/-- **Sequentially dominated events, one coordinate at a time**: `r` or more of them hold on at most a
`(Σ p)^r / r!` share of the coins. -/
theorem count_tail_fiber {B : ℕ} {X : Fin B → Type} [∀ i, Fintype (X i)] (E : Fin B → (∀ i, X i) → Prop)
    (p : Fin B → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hread : ∀ i (ω ω' : ∀ i, X i), (∀ j, j ≤ i → ω j = ω' j) → (E i ω ↔ E i ω'))
    (hfib : ∀ i (ω : ∀ i, X i),
      ((Finset.univ.filter fun v : X i => E i (Function.update ω i v)).card : ℝ) ≤ p i * Fintype.card (X i))
    (r : ℕ) :
    Pouw.pr (fun ω : ∀ i, X i => r ≤ (Finset.univ.filter fun i => E i ω).card) ≤
      (∑ i, p i) ^ r / r.factorial := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
