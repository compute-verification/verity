import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Sampling.Cost

/-!
# Binding of the trace's commitment, at the openings the extractor checks

The claims block's `Binding` is `CommitmentScheme.Binds` at every `H`: two openings that pass at one position under one
roots, of different values, make `H` calls that collide. The extractor checks every opening it keeps with the
commitment's own check (`checkRevealed`), its calls the extractor's own, so two kept openings that disagree are a
collision among those calls, and `findColl` finds one.

* `findColl_isSome`: `findColl` finds a collision whenever its list holds two different inputs with one output.
* `disagree_collision`: under `CommitmentScheme.Binds`, two passing openings of one position to different values make calls
  that collide in any log holding both checks' calls.
* `Pr_union_le`: the union bound.
-/

namespace Claims.Sampling

open Claims

/-- `findColl` finds a collision whenever the list holds two different inputs with one output. -/
theorem findColl_isSome :
    ∀ l : List (Msg × Msg), (∃ p ∈ l, ∃ p' ∈ l, p.1 ≠ p'.1 ∧ p.2 = p'.2) → (findColl l).isSome := by
  intro l
  induction l with
  | nil => rintro ⟨p, hp, _⟩; simp at hp
  | cons hd tl ih =>
    rintro ⟨p, hp, p', hp', hne, heq⟩
    obtain ⟨q, a⟩ := hd
    simp only [findColl]
    cases hf : tl.find? (fun p => decide (p.1 ≠ q) && decide (p.2 = a)) with
    | some p'' => rfl
    | none =>
      apply ih
      have hno : ∀ y ∈ tl, ¬ (y.1 ≠ q ∧ y.2 = a) := by
        intro y hy hy'
        have := List.find?_eq_none.1 hf y hy
        simp [hy'.1, hy'.2] at this
      rcases List.mem_cons.1 hp with rfl | hp <;> rcases List.mem_cons.1 hp' with rfl | hp'
      · exact absurd rfl hne
      · exact absurd ⟨fun h => hne h.symm, heq.symm⟩ (hno _ hp')
      · exact absurd ⟨hne, heq⟩ (hno _ hp)
      · exact ⟨p, hp, p', hp', hne, heq⟩

/-- **Disagreeing openings collide**: under `CommitmentScheme.Binds`, two openings that pass at one position under one
roots, of different values, make calls that collide in any log `l` that holds both checks' calls. -/
theorem disagree_collision {H : Msg → Msg} {Cm : CommitmentScheme Bool} (hb : Cm.Binds H) {roots : List Msg}
    {o o' : Opening Bool} (hpos : o.pos = o'.pos) (hne : o.val ≠ o'.val) (h : Cm.Opens H roots o = true)
    (h' : Cm.Opens H roots o' = true) {l : List (Msg × Msg)} (hl : ∀ q ∈ ((Cm.opens roots o).eval H).2, q ∈ l)
    (hl' : ∀ q ∈ ((Cm.opens roots o').eval H).2, q ∈ l) : LogColl l :=
  (hb roots o o' hpos hne h h').mono fun q hq => (List.mem_append.1 hq).elim (hl q) (hl' q)

theorem Pr_union_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  unfold Pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  exact_mod_cast (Finset.card_le_card fun a ha => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at ha ⊢; exact ha).trans
    (Finset.card_union_le _ _)

end Claims.Sampling
