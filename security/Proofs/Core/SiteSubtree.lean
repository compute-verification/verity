module

public import Definitions.Core.SiteSubtree
public import Mathlib.Data.Finset.Union

/-!
# A subtree of the Site: its places and its parts

* `Site.mem_places`: the places are the ids the Site knows (`Site.Known`), as a finite set.
* `Site.under_eq`: a subtree is its root and its root's children's subtrees.
* `Site.placesIn_union`, `Site.placesIn_biUnion`: the places of a union of sets of ids are the union of their places.
-/

@[expose] public section

namespace Verity.Site

open Finset

variable (S : Site)

/-- The places are the ids the Site knows. -/
theorem mem_places {i : String} : i ∈ S.places ↔ S.Known i := by
  unfold places Known
  rw [Finset.mem_insert, List.mem_toFinset]

/-- A location registered under another is a place. -/
theorem mem_places_of_child {i c : String} (h : S.Child i c) : c ∈ S.places := by
  obtain ⟨x, hx, hxc, -⟩ := h
  exact Finset.mem_insert_of_mem (List.mem_toFinset.2 (List.mem_map.2 ⟨x, hx, hxc⟩))

/-- **A subtree is its root and its root's children's subtrees.** -/
theorem under_eq (i : String) (hi : i ∈ S.places) :
    S.under i = insert i ((S.places.filter (S.Child i)).biUnion S.under) := by
  ext j
  simp only [under, mem_filter, mem_insert, mem_biUnion]
  constructor
  · rintro ⟨hj, h⟩
    rcases Relation.ReflTransGen.cases_head h with rfl | ⟨c, hc, hcj⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨c, ⟨S.mem_places_of_child hc, hc⟩, hj, hcj⟩
  · rintro (rfl | ⟨c, ⟨-, hc⟩, hj, hcj⟩)
    · exact ⟨hi, Relation.ReflTransGen.refl⟩
    · exact ⟨hj, Relation.ReflTransGen.head hc hcj⟩

/-- The places of a union of two sets of ids. -/
theorem placesIn_union (X Y : Finset String) : S.placesIn (X ∪ Y) = S.placesIn X ∪ S.placesIn Y := by
  ext p
  simp only [placesIn, mem_filter, mem_univ, true_and, mem_union]

/-- The places of a union of sets of ids. -/
theorem placesIn_biUnion {ι : Type*} (P : Finset ι) (X : ι → Finset String) :
    S.placesIn (P.biUnion X) = P.biUnion fun p => S.placesIn (X p) := by
  ext q
  simp only [placesIn, mem_filter, mem_univ, true_and, mem_biUnion]

end Verity.Site
