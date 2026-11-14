module

public import Definitions.Core.SiteSubtree
public import Proofs.Core.Site
public import Mathlib.Data.Finset.Union

/-!
# A subtree of the Site: its places and its auditor's devices

* `Site.mem_placesIn`: a place is among the places of a set of ids when its id is in the set.
* `Site.mem_places`: the places are the ids the Site knows (`Site.Known`), as a finite set.
* `Site.placesIn_under`: a subtree's places are its root, where that is a place, and its root's children's subtrees'
  places.
* `Site.under_eq_of_closed`: a set of places that holds `i`, holds each location registered under one of its own and
  is reached from `i` is the subtree at `i`, which reads a subtree off a Site given in full.
* `Site.outside_notMem_under`, `Site.outside_mem_compl`: the outside is in no subtree but its own, and is outside a
  subtree's location and its auditor's devices.
* `Site.crossing`, `Site.uncovered`: the registered links with exactly one end in a set of ids, and those of them with
  neither end in a covering set, as lists of links.
* `Site.mem_far_iff`: where no link joins a place to itself, as on a valid log's Site, a far end of a place's links is
  the other end of a registered link that ends at it.
* `Site.challengers_sealed`: at a Site whose links join kinds a link may join, every link of a memory challenger linked
  only into a subtree ends at one of the subtree's prover places (`Site.mem_far_of_link`, `Site.host_of_challenger`);
  so such challengers, as sinks, add no uncovered link (`Site.uncovered_sealed`).
* `Site.within_of_not_shared`, `Site.units_union_sinks`: a subtree's challenger that is not shared is linked only into
  it, and the subtree's certifiers, a set of its challengers and its other challengers are its auditor's devices.
-/

@[expose] public section

namespace Verity.Site

open Finset

variable (S : Site)

/-- A place is among the places of a set of ids exactly when its id is in the set. -/
theorem mem_placesIn {X : Finset String} {p : S.Place} : p ∈ S.placesIn X ↔ p.1 ∈ X := by
  simp only [placesIn, mem_filter, mem_univ, true_and]

/-- The places are the ids the Site knows. -/
theorem mem_places {i : String} : i ∈ S.places ↔ S.Known i := by
  unfold places Known
  rw [Finset.mem_insert, List.mem_toFinset]

/-- The place the outside's id names is the outside. -/
theorem place_outside : (S.place outside).1 = outside := by
  have h : outside ∈ S.places := Finset.mem_insert_self outside _
  rw [place, dif_pos h]

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

/-- **The outside is in no subtree but its own**: where no location has the outside's id, the outside is not in the
subtree at any other place. -/
theorem outside_notMem_under (hno : ∀ x ∈ S.locs, x.id ≠ outside) {i : String} (hi : i ≠ outside) :
    outside ∉ S.under i := by
  intro h
  unfold under at h
  rcases Relation.ReflTransGen.cases_tail (Finset.mem_filter.1 h).2 with h | ⟨c, -, x, hx, hxo, -⟩
  · exact hi h.symm
  · exact hno x hx hxo

/-- **The outside is outside a subtree's location and its auditor's devices**, with no challenger pinned: where no
location has the outside's id and the outside is no certifier, at a subtree at any other place the outside is none of
the subtree's prover places, its memory challengers or its certifiers. -/
theorem outside_mem_compl (hno : ∀ x ∈ S.locs, x.id ≠ outside) (hk : S.kind? outside ≠ some .certifier)
    {i : String} (hi : i ≠ outside) :
    S.place outside ∈
      (S.placesIn (S.proverUnder i ∪ (S.challengersOf ∅ i \ ∅)) ∪ S.placesIn (S.certifiersOf i ∪ ∅))ᶜ := by
  classical
  rw [Finset.mem_compl, Finset.mem_union, S.mem_placesIn, S.mem_placesIn, S.place_outside]
  rintro (h | h)
  · rcases Finset.mem_union.1 h with h | h
    · unfold proverUnder at h
      exact S.outside_notMem_under hno hi (Finset.mem_filter.1 h).1
    · unfold challengersOf at h
      exact Finset.notMem_empty _ (Finset.mem_filter.1 (Finset.mem_sdiff.1 h).1).2.2.1
  · rcases Finset.mem_union.1 h with h | h
    · unfold certifiersOf at h
      exact hk (Finset.mem_filter.1 h).2.1
    · exact Finset.notMem_empty _ h

/-- The registered links with exactly one end in `X`. -/
def crossing (X : Finset String) : List Link :=
  S.links.filter fun k => decide (k.ends.1 ∈ X ↔ k.ends.2 ∉ X)

/-- **The uncovered links of `X`** under the covering set `cov`: the registered links with exactly one end in `X` and
neither end in `cov`. -/
def uncovered (cov X : Finset String) : List Link :=
  (S.crossing X).filter fun k => decide (k.ends.1 ∉ cov ∧ k.ends.2 ∉ cov)

/-- **A subtree's memory challenger is a pinned, registered challenger at a link's far end**: it is registered as a
memory challenger, is among `pinned`, and is a far end of a link of one of the subtree's prover places. -/
theorem mem_challengersOf {pinned : Finset String} {i y : String} (h : y ∈ S.challengersOf pinned i) :
    S.kind? y = some .challenger ∧ y ∈ pinned ∧ ∃ x ∈ S.proverUnder i, y ∈ S.far x := by
  classical
  unfold challengersOf at h
  exact (Finset.mem_filter.1 h).2

/-- **A link's two ends are each a far end of the other's**, when they differ. -/
theorem mem_far_of_link {k : Link} (hk : k ∈ S.links) :
    (k.ends.2 ≠ k.ends.1 → k.ends.2 ∈ S.far k.ends.1) ∧ (k.ends.1 ≠ k.ends.2 → k.ends.1 ∈ S.far k.ends.2) := by
  unfold far
  constructor
  · intro h
    refine List.mem_flatMap.2 ⟨k, List.mem_filter.2 ⟨hk, by simp⟩, ?_⟩
    simp [h]
  · intro h
    refine List.mem_flatMap.2 ⟨k, List.mem_filter.2 ⟨hk, by simp⟩, ?_⟩
    simp [h]

/-- **Where no link joins a place to itself**, as on a valid log's Site (`Registration.Valid.ends_ne`), `y` is a far
end of `x`'s links exactly when a registered link has its two ends at `x` and `y`. On any Site the far ends of `x`
leave `x` itself out, so a link from `x` to `x` gives it no far end. -/
theorem mem_far_iff (hne : ∀ k ∈ S.links, k.ends.1 ≠ k.ends.2) {x y : String} :
    y ∈ S.far x ↔ ∃ k ∈ S.links, (k.ends.1 = x ∧ k.ends.2 = y) ∨ (k.ends.1 = y ∧ k.ends.2 = x) := by
  constructor
  · intro h
    unfold far at h
    obtain ⟨k, hk, hy⟩ := List.mem_flatMap.1 h
    obtain ⟨hk, hx⟩ := List.mem_filter.1 hk
    have hx : k.ends.1 = x ∨ k.ends.2 = x := by simpa using hx
    have hy : (y = k.ends.1 ∨ y = k.ends.2) ∧ y ≠ x := by simpa using hy
    refine ⟨k, hk, ?_⟩
    rcases hx with hx | hx <;> rcases hy with ⟨hy | hy, hyx⟩
    · exact absurd (hy.trans hx) hyx
    · exact Or.inl ⟨hx, hy.symm⟩
    · exact Or.inr ⟨hy.symm, hx⟩
    · exact absurd (hy.trans hx) hyx
  · rintro ⟨k, hk, ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩⟩
    · exact (S.mem_far_of_link hk).1 (hne k hk).symm
    · exact (S.mem_far_of_link hk).2 (hne k hk)

/-- **The memory challengers linked only into a subtree are sealed by the registration**: at a Site whose links join
kinds a link may join, every link with an end at a memory challenger of `sink`, each linked only into the subtree at
`i`, has its other end at one of the subtree's prover places: a host at or under `i`. Nothing is assumed of the links
but the entry check. -/
theorem challengers_sealed (hk : S.LinkKinds) (i : String) (sink : Finset String)
    (hs : ∀ y ∈ sink, S.kind? y = some .challenger ∧ S.Within i y) :
    ∀ k ∈ S.links, (k.ends.1 ∈ sink → k.ends.2 ∈ S.proverUnder i) ∧
      (k.ends.2 ∈ sink → k.ends.1 ∈ S.proverUnder i) := by
  classical
  intro k hmem
  have hh := host_of_challenger hk hmem
  have hfar := S.mem_far_of_link hmem
  constructor
  · intro h1
    obtain ⟨hc, hw⟩ := hs _ h1
    have hhost := hh.1 hc
    have hne : k.ends.2 ≠ k.ends.1 := by
      intro h
      rw [h, hc] at hhost
      exact absurd hhost (by decide)
    unfold proverUnder
    refine Finset.mem_filter.2 ⟨hw _ (hfar.1 hne), ?_⟩
    intro k' hk'
    rw [hhost, Option.mem_def, Option.some.injEq] at hk'
    subst hk'
    rfl
  · intro h2
    obtain ⟨hc, hw⟩ := hs _ h2
    have hhost := hh.2 hc
    have hne : k.ends.1 ≠ k.ends.2 := by
      intro h
      rw [h, hc] at hhost
      exact absurd hhost (by decide)
    unfold proverUnder
    refine Finset.mem_filter.2 ⟨hw _ (hfar.2 hne), ?_⟩
    intro k' hk'
    rw [hhost, Option.mem_def, Option.some.injEq] at hk'
    subst hk'
    rfl

/-- **Sealed sinks add nothing to the uncovered links**: where every registered link with an end at a place of
`sink` has its other end in `X`, the uncovered links of the location `X ∪ sink` are those of `X` with the sinks
covering beside `cert`: no link of a sink is among them. -/
theorem uncovered_sealed (cert X sink : Finset String)
    (h : ∀ k ∈ S.links, (k.ends.1 ∈ sink → k.ends.2 ∈ X) ∧ (k.ends.2 ∈ sink → k.ends.1 ∈ X)) :
    S.uncovered cert (X ∪ sink) = S.uncovered (cert ∪ sink) X := by
  unfold uncovered crossing
  rw [List.filter_filter, List.filter_filter]
  refine List.filter_congr fun k hk => ?_
  have h1 := (h k hk).1
  have h2 := (h k hk).2
  by_cases a1 : k.ends.1 ∈ X <;> by_cases a2 : k.ends.2 ∈ X <;> by_cases b1 : k.ends.1 ∈ sink <;>
    by_cases b2 : k.ends.2 ∈ sink <;> by_cases c1 : k.ends.1 ∈ cert <;> by_cases c2 : k.ends.2 ∈ cert <;>
    simp_all

/-- A memory challenger of the subtree at `i` that is not shared is linked only into the subtree. -/
theorem within_of_not_shared {pinned : Finset String} {i y : String}
    (h : y ∈ S.challengersOf pinned i \ S.sharedOf pinned i) : S.Within i y := by
  classical
  obtain ⟨h1, h2⟩ := Finset.mem_sdiff.1 h
  by_contra hw
  exact h2 (by unfold sharedOf; exact Finset.mem_filter.2 ⟨h1, hw⟩)

/-- The certifiers of the subtree at `i`, the challengers `units` and the subtree's other challengers are together the
auditor's devices at the subtree. -/
theorem units_union_sinks {pinned units : Finset String} {i : String} (hu : units ⊆ S.challengersOf pinned i) :
    S.certifiersOf i ∪ units ∪ (S.challengersOf pinned i \ units) = S.auditorsOf pinned i := by
  ext y
  simp only [auditorsOf, Finset.mem_union, Finset.mem_sdiff]
  constructor
  · rintro ((h | h) | ⟨h, -⟩)
    · exact Or.inl h
    · exact Or.inr (hu h)
    · exact Or.inr h
  · rintro (h | h)
    · exact Or.inl (Or.inl h)
    · by_cases hy : y ∈ units
      · exact Or.inl (Or.inr hy)
      · exact Or.inr ⟨h, hy⟩

end Verity.Site
