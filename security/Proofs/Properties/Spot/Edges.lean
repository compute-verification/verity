module

public import Properties.Spot.Edges

/-! # Proof that a root `Spot.Edges.edgesOf` accepts is within the parts bound (`Properties.Spot.Edges`)

`argPartsOk`'s set of spaces, built by inserting each part's `partKey`, has as many members as the parts' keys as a
`Finset` (`foldl_insert_size`), so an argument it accepts has at most one part more than those keys; `edgesOf` refuses
first unless every argument of every root node is accepted (`edgesOf_partsOk`).
-/

@[expose] public section

open _root_.Spot
open Flock.Program (Prog Part)

namespace Properties.Spot.Edges

private theorem foldl_insert_size (l : List Part) (h : Std.HashSet (Option (Bool × ℕ)))
    (S : Finset (Option (Bool × ℕ))) (hs : h.size = S.card) (hm : ∀ x, x ∈ h ↔ x ∈ S) :
    (l.foldl (fun h q => h.insert (Edges.partKey q)) h).size = (S ∪ (l.map Edges.partKey).toFinset).card := by
  induction l generalizing h S with
  | nil => simp [hs]
  | cons q l ih =>
    rw [List.foldl_cons]
    have hs' : (h.insert (Edges.partKey q)).size = (insert (Edges.partKey q) S).card := by
      rw [Std.HashSet.size_insert]
      by_cases hq : Edges.partKey q ∈ S
      · rw [if_pos ((hm _).mpr hq), Finset.card_insert_of_mem hq, hs]
      · rw [if_neg (fun h' => hq ((hm _).mp h')), Finset.card_insert_of_notMem hq, hs]
    have hm' : ∀ x, x ∈ h.insert (Edges.partKey q) ↔ x ∈ insert (Edges.partKey q) S := by
      intro x
      rw [Std.HashSet.mem_insert, Finset.mem_insert, hm, beq_iff_eq]
      exact ⟨fun h' => h'.elim (fun e => Or.inl e.symm) Or.inr, fun h' => h'.elim (fun e => Or.inl e.symm) Or.inr⟩
    rw [ih _ _ hs' hm']
    congr 1
    ext x
    simp only [Finset.mem_union, Finset.mem_insert, List.mem_toFinset, List.map_cons, List.mem_cons]
    tauto

theorem argPartsOk_card {k np : ℕ} {s : Flock.Program.Seq} (h : Edges.argPartsOk k np s = true) :
    s.parts.size ≤ (s.parts.toList.map Edges.partKey).toFinset.card + 1 := by
  unfold Edges.argPartsOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  have := h.2
  rw [← Array.foldl_toList, foldl_insert_size s.parts.toList {} ∅ (by simp) (by simp)] at this
  simpa using this

theorem edgesOf_partsOk {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {W : Spot.Wires.Wiring}
    {ws : List Spot.Wires.Wire} (h : Edges.edgesOf P rs W = .ok ws) : Edges.partsOk P = true := by
  cases hP : Edges.partsOk P
  · unfold Edges.edgesOf at h
    simp [hP, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · rfl

/-- **A root `edgesOf` accepts is within the parts bound.** -/
theorem edgesOf_partsWithin {P : Prog} {rs : Array Flock.TemplateQuery.TRange} {W : Spot.Wires.Wiring}
    {ws : List Spot.Wires.Wire} (h : Edges.edgesOf P rs W = .ok ws) : PartsWithin P := by
  have hp := edgesOf_partsOk h
  unfold Edges.partsOk at hp
  intro k hk j hj
  have hk' := List.all_eq_true.mp hp k (List.mem_range.mpr hk)
  rw [Array.all_eq_true] at hk'
  exact argPartsOk_card (hk' j hj)

end Properties.Spot.Edges
