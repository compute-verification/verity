module

public import Definitions.Core.SiteSubtree
public import Proofs.Core.Site
public import Mathlib.Data.Finset.Union

/-! A subtree of the Site: its places and its auditor's devices. -/

@[expose] public section

namespace Verity.Site

open Finset

variable (S : Site)

theorem mem_placesIn {X : Finset String} {p : S.Place} : p ∈ S.placesIn X ↔ p.1 ∈ X := by
  simp only [placesIn, mem_filter, mem_univ, true_and]

theorem mem_places {i : String} : i ∈ S.places ↔ S.Known i := by
  unfold places Known
  rw [Finset.mem_insert, List.mem_toFinset]

theorem place_outside : (S.place outside).1 = outside := by
  have h : outside ∈ S.places := Finset.mem_insert_self outside _
  rw [place, dif_pos h]

theorem mem_under {i j : String} : j ∈ S.under i ↔ j ∈ S.places ∧ Relation.ReflTransGen S.Child i j := by
  classical
  unfold under
  exact mem_filter


theorem mem_places_of_child {i c : String} (h : S.Child i c) : c ∈ S.places := by
  obtain ⟨x, hx, hxc, -⟩ := h
  exact Finset.mem_insert_of_mem (List.mem_toFinset.2 (List.mem_map.2 ⟨x, hx, hxc⟩))

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

theorem outside_notMem_under (hno : ∀ x ∈ S.locs, x.id ≠ outside) {i : String} (hi : i ≠ outside) :
    outside ∉ S.under i := by
  intro h
  rcases Relation.ReflTransGen.cases_tail (S.mem_under.1 h).2 with h | ⟨c, -, x, hx, hxo, -⟩
  · exact hi h.symm
  · exact hno x hx hxo

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

theorem mem_challengersOf {pinned : Finset String} {i y : String} (h : y ∈ S.challengersOf pinned i) :
    S.kind? y = some .challenger ∧ y ∈ pinned ∧ ∃ x ∈ S.proverUnder i, y ∈ S.far x := by
  classical
  unfold challengersOf at h
  exact (Finset.mem_filter.1 h).2

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

/-- Nothing is assumed of the links but the entry check. -/
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

theorem within_of_not_shared {pinned : Finset String} {i y : String}
    (h : y ∈ S.challengersOf pinned i \ S.sharedOf pinned i) : S.Within i y := by
  classical
  obtain ⟨h1, h2⟩ := Finset.mem_sdiff.1 h
  by_contra hw
  exact h2 (by unfold sharedOf; exact Finset.mem_filter.2 ⟨h1, hw⟩)

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
