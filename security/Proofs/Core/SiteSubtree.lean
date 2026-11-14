module

public import Definitions.Core.SiteSubtree
public import Mathlib.Data.Finset.Union

/-!
# A subtree of the Site: its places

* `Site.mem_placesIn`, `Site.mem_under`: a place is among the places of a set of ids when its id is in the set, and
  in a subtree when it is reached from the subtree's root.
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

/-- A place is in the subtree at `i` exactly when it is reached from `i` through the locations registered under
one another. -/
theorem mem_under {i j : String} : j ∈ S.under i ↔ j ∈ S.places ∧ Relation.ReflTransGen S.Child i j := by
  classical
  unfold under
  exact mem_filter

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
  constructor
  · intro h
    rcases Relation.ReflTransGen.cases_head (S.mem_under.1 (S.mem_placesIn.1 h)).2 with h | ⟨c, hc, hcj⟩
    · exact mem_union_left _ (S.mem_placesIn.2 (mem_singleton.2 h.symm))
    · exact mem_union_right _ (mem_biUnion.2
        ⟨c, mem_filter.2 ⟨S.mem_places_of_child hc, hc⟩, S.mem_placesIn.2 (S.mem_under.2 ⟨p.2, hcj⟩)⟩)
  · intro h
    refine S.mem_placesIn.2 (S.mem_under.2 ⟨p.2, ?_⟩)
    rcases mem_union.1 h with h | h
    · have hp : p.1 = i := mem_singleton.1 (S.mem_placesIn.1 h)
      subst hp
      exact Relation.ReflTransGen.refl
    · obtain ⟨c, hc, hj⟩ := mem_biUnion.1 h
      exact Relation.ReflTransGen.head (mem_filter.1 hc).2 (S.mem_under.1 (S.mem_placesIn.1 hj)).2

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
