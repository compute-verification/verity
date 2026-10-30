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

open Finset

theorem pr_congr {α : Type} [Fintype α] {P P' : α → Prop} (h : ∀ x, P x ↔ P' x) : Pouw.pr P = Pouw.pr P' :=
  congrArg Pouw.pr (funext fun x => propext (h x))

open Classical in
/-- On a product, an event that is a rectangle has the product of its sides' counts. -/
theorem pr_rect {α β : Type} [Fintype α] [Fintype β] (V : Finset α) (W : Finset β) :
    Pouw.pr (fun y : α × β => y.1 ∈ V ∧ y.2 ∈ W) = (#V * #W : ℝ) / (Fintype.card α * Fintype.card β) := by
  rw [Pouw.pr_eq_card _ (V ×ˢ W) fun y => mem_product, card_product, Fintype.card_prod]
  push_cast
  rfl

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
  refine count_tail E p hp hread (fun i ω => ?_) r
  let e := Equiv.piSplitAt i X
  let V := univ.filter fun v : X i => E i (Function.update ω i v)
  let W := univ.filter fun f : (∀ j : {j // j ≠ i}, X j) => ∀ j (h : j < i), f ⟨j, h.ne⟩ = ω j
  have hcyl : ∀ ω' : ∀ i, X i, (∀ j, j < i → ω' j = ω j) ↔ (e ω').2 ∈ W := fun ω' => by
    simp only [W, mem_filter, mem_univ, true_and]
    rfl
  have hupd : ∀ ω' : ∀ i, X i, (∀ j, j < i → ω' j = ω j) →
      ∀ j, j ≤ i → Function.update ω i (ω' i) j = ω' j := fun ω' h j hj => by
    rcases lt_or_eq_of_le hj with hj | rfl
    · rw [Function.update_of_ne hj.ne]
      exact (h j hj).symm
    · exact Function.update_self _ _ _
  have hev : ∀ ω' : ∀ i, X i, ((∀ j, j < i → ω' j = ω j) ∧ E i ω') ↔ (e ω').1 ∈ V ∧ (e ω').2 ∈ W := fun ω' => by
    rw [← hcyl]
    simp only [V, mem_filter, mem_univ, true_and]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨(hread i _ _ (hupd ω' h1)).2 h2, h1⟩
    · rintro ⟨h2, h1⟩
      exact ⟨h1, (hread i _ _ (hupd ω' h1)).1 h2⟩
  have hl : Pouw.pr (fun ω' : ∀ i, X i => (∀ j, j < i → ω' j = ω j) ∧ E i ω') =
      Pouw.pr (fun y : X i × (∀ j : {j // j ≠ i}, X j) => y.1 ∈ V ∧ y.2 ∈ W) :=
    (pr_congr hev).trans (Pouw.pr_equiv e fun y => y.1 ∈ V ∧ y.2 ∈ W)
  have hr : Pouw.pr (fun ω' : ∀ i, X i => ∀ j, j < i → ω' j = ω j) =
      Pouw.pr (fun y : X i × (∀ j : {j // j ≠ i}, X j) => y.1 ∈ univ ∧ y.2 ∈ W) :=
    (pr_congr fun ω' => (hcyl ω').trans (and_iff_right (mem_univ _)).symm).trans
      (Pouw.pr_equiv e fun y => y.1 ∈ univ ∧ y.2 ∈ W)
  rw [hl, hr, pr_rect, pr_rect, card_univ, mul_div_assoc']
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  calc (#V * #W : ℝ) ≤ p i * Fintype.card (X i) * #W :=
        mul_le_mul_of_nonneg_right (hfib i ω) (Nat.cast_nonneg _)
    _ = p i * (Fintype.card (X i) * #W) := by ring

end Pouw.SecurityProofs.Lead.Tail
