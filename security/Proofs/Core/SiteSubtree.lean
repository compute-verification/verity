module

public import Definitions.Core.SiteSubtree
public import Mathlib.Data.Finset.Union

/-!
# A subtree of the Site: its places

* `Site.mem_placesIn`: a place is among the places of a set of ids when its id is in the set.
* `Site.placesIn_under`: a subtree's places are its root, where that is a place, and its root's children's subtrees'
  places.
* `Site.under_eq_of_closed`: a set of places that holds `i`, holds each location registered under one of its own and
  is reached from `i` is the subtree at `i`, which reads a subtree off a Site given in full.
-/

@[expose] public section

namespace Verity.Site

open Finset

variable (S : Site)

/-- A place is among the places of a set of ids exactly when its id is in the set. -/
theorem mem_placesIn {X : Finset String} {p : S.Place} : p ∈ S.placesIn X ↔ p.1 ∈ X := by
  simp only [placesIn, mem_filter, mem_univ, true_and]

/-- A location registered under another is a place. -/
theorem mem_places_of_child {i c : String} (h : S.Child i c) : c ∈ S.places := by
  obtain ⟨x, hx, hxc, -⟩ := h
  exact Finset.mem_insert_of_mem (List.mem_toFinset.2 (List.mem_map.2 ⟨x, hx, hxc⟩))

/-- **A subtree's places are its root and its root's children's subtrees' places**; at an id that names no place,
its children's subtrees' alone. -/
theorem placesIn_under (i : String) :
    S.placesIn (S.under i) =
      S.placesIn {i} ∪ (S.places.filter (S.Child i)).biUnion fun c => S.placesIn (S.under c) := by
  ext p
  simp only [placesIn, under, mem_filter, mem_univ, true_and, mem_union, mem_singleton, mem_biUnion]
  constructor
  · rintro ⟨-, h⟩
    rcases Relation.ReflTransGen.cases_head h with h | ⟨c, hc, hcj⟩
    · exact Or.inl h.symm
    · exact Or.inr ⟨c, ⟨S.mem_places_of_child hc, hc⟩, p.2, hcj⟩
  · rintro (rfl | ⟨c, ⟨-, hc⟩, hj, hcj⟩)
    · exact ⟨p.2, Relation.ReflTransGen.refl⟩
    · exact ⟨hj, Relation.ReflTransGen.head hc hcj⟩

/-- **A subtree read off a Site given in full**: a set `T` of places that holds `i`, holds every location registered
under one of its own, and each of whose places is reached from `i`, is the subtree at `i`. The first three are
decided on a Site given in full. -/
theorem under_eq_of_closed {i : String} {T : Finset String} (hT : ∀ j ∈ T, j ∈ S.places)
    (hc : ∀ x ∈ S.locs, ∀ j ∈ T, x.parent = some j → x.id ∈ T) (hi : i ∈ T)
    (hr : ∀ j ∈ T, Relation.ReflTransGen S.Child i j) : S.under i = T := by
  ext j
  simp only [under, mem_filter]
  constructor
  · rintro ⟨-, h⟩
    induction h with
    | refl => exact hi
    | tail _ hbc ih =>
      obtain ⟨x, hx, rfl, hp⟩ := hbc
      exact hc x hx _ ih hp
  · exact fun hj => ⟨hT j hj, hr j hj⟩

end Verity.Site
