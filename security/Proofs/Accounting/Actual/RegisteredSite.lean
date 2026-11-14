module

public import Definitions.Accounting.Guarantees.RegisteredSite
public import Proofs.Accounting.Actual.Parts
public import Proofs.Accounting.Actual.Physical
public import Proofs.Core.SiteSubtree
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# A subtree's limits, read off the registration

* `linkCapacity_union_le`, `linkCapacity_biUnion_le`: in the model, the capacity across the boundary of a union of
  parts is at most the parts' capacities summed, since a link across the union's boundary crosses a part's.
* `Site.egress_le_registered`: `egress_le` at the Site's places and links and a subtree.
* `Site.egressLimit_le_parts`: a subtree's limit is at most its root's alone plus its root's children's subtrees'
  (`Site.under_eq`).
* `EgressRegistered`: the two together, the guarantee.
* `Site.uncertified_registered_eq`: where every registered link ends at places, the model's `β°` at the Site's links is
  the sum of `β` over the uncovered links (`Site.uncovered`).
* `Site.leak_le_registered`: `leak_le_sum_K` at a subtree, its certifiers and a set of its memory challengers the
  trusted devices and its other challengers sinks, which the registration seals (`Site.challengers_sealed`,
  `Site.uncovered_sealed`).
* `LeakPartsRegistered`: that at a valid log's Site (`Registration.Valid.linkKinds`, `Registration.Valid.ends_known`),
  the guarantee.
-/

@[expose] public section

namespace Accounting

open Verity Finset

section Parts

variable {L E ι : Type*} [Fintype E] [DecidableEq L]

/-- A link across the boundary of two parts' union crosses the boundary of one of them. -/
theorem linkCapacity_union_le (ends : E → L × L) (β : E → ℕ) (X Y : Finset L) :
    linkCapacity ends β (X ∪ Y) ≤ linkCapacity ends β X + linkCapacity ends β Y := by
  classical
  have hsub : boundary ends (X ∪ Y) ⊆ boundary ends X ∪ boundary ends Y := by
    intro e he
    simp only [boundary, mem_filter, mem_univ, true_and, mem_union] at he ⊢
    by_cases h1 : (ends e).1 ∈ X <;> by_cases h2 : (ends e).2 ∈ X <;>
      by_cases h3 : (ends e).1 ∈ Y <;> by_cases h4 : (ends e).2 ∈ Y <;> simp_all
  unfold linkCapacity
  exact (Finset.sum_le_sum_of_subset hsub).trans (Nat.le.intro Finset.sum_union_inter)

/-- **Limits add over parts, at most**: the capacity across the boundary of a union of parts is at most the parts'
capacities summed. -/
theorem linkCapacity_biUnion_le (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L) :
    linkCapacity ends β (P.biUnion X) ≤ ∑ p ∈ P, linkCapacity ends β (X p) := by
  classical
  induction P using Finset.induction_on with
  | empty => simp [linkCapacity, boundary]
  | insert p P hp ih =>
    rw [Finset.biUnion_insert, Finset.sum_insert hp]
    exact (linkCapacity_union_le ends β (X p) (P.biUnion X)).trans (Nat.add_le_add_left ih _)

end Parts

end Accounting

namespace Verity.Site

open Finset

variable (S : Site)

/-- **A subtree's egress is at most its registered limit**: for an actual circuit on the Site's places whose wires run
on the registered links within their capacity, what leaves the subtree at `i` over the ticks `[a, b)` is at most the
capacity of the registered links with exactly one end in it, a tick. -/
theorem egress_le_registered (i : String) (A : Accounting.Placed S.Place)
    (link : Fin A.N → Fin A.N → Fin S.links.length) (h : A.Links S.placeEnds S.linkβ link) (a b : ℕ) :
    A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤
      Accounting.linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) * (b - a) :=
  Accounting.SecurityProofs.Actual.egress_le A h (S.placesIn (S.under i)) a b

/-- **A subtree's limit is at most its root's and its root's children's subtrees', summed**: limits add over the parts
of a subtree of the Site, at most. -/
theorem egressLimit_le_parts (i : String) (hi : i ∈ S.places) :
    Accounting.linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) ≤
      Accounting.linkCapacity S.placeEnds S.linkβ (S.placesIn {i}) +
        ∑ c ∈ S.places.filter (S.Child i), Accounting.linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under c)) := by
  rw [S.under_eq i hi, Finset.insert_eq, S.placesIn_union, S.placesIn_biUnion]
  exact (Accounting.linkCapacity_union_le _ _ _ _).trans
    (Nat.add_le_add_left (Accounting.linkCapacity_biUnion_le _ _ _ _) _)

/-- **The uncovered term is read off the registered links**: where every registered link ends at places of the Site,
the model's `β°` at the registered links, a covering set `cov` and a location `X` is the sum of `β` over the
registered links with exactly one end in `X` and neither end in `cov` (`uncovered`). -/
theorem uncertified_registered_eq (cov X : Finset String)
    (hends : ∀ k ∈ S.links, k.ends.1 ∈ S.places ∧ k.ends.2 ∈ S.places) :
    Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.placesIn cov) (S.placesIn X) =
      ((S.uncovered cov X).map (·.β)).sum := by
  classical
  have hl : ∀ (l : List Link) (p : Link → Bool),
      ((l.filter p).map (·.β)).sum = (l.map fun k => if p k = true then k.β else 0).sum := by
    intro l p
    induction l with
    | nil => rfl
    | cons k l ih =>
      rw [List.filter_cons]
      by_cases hk : p k = true
      · rw [ite_eq_left hk, List.map_cons, List.sum_cons, List.map_cons, List.sum_cons, ite_eq_left hk, ih]
      · rw [ite_eq_right hk, List.map_cons, List.sum_cons, ite_eq_right hk, ih, Nat.zero_add]
  have hp : ∀ (e : Fin S.links.length) (Y : Finset String),
      ((S.placeEnds e).1 ∈ S.placesIn Y ↔ (S.links[e.1]).ends.1 ∈ Y) ∧
        ((S.placeEnds e).2 ∈ S.placesIn Y ↔ (S.links[e.1]).ends.2 ∈ Y) := by
    intro e Y
    have h := hends (S.links[e.1]) (List.getElem_mem e.2)
    simp [placeEnds, placesIn, place, h.1, h.2]
  unfold Accounting.uncertifiedCapacity Accounting.boundary uncovered crossing
  rw [List.filter_filter, hl, Finset.sum_filter, Finset.sum_filter, ← Fin.sum_univ_fun_getElem]
  refine Finset.sum_congr rfl fun e _ => ?_
  have hX := hp e X
  have hC := hp e cov
  by_cases h : ((S.links[e.1]).ends.1 ∈ X ↔ (S.links[e.1]).ends.2 ∉ X) <;>
    by_cases h1 : (S.links[e.1]).ends.1 ∈ cov <;> by_cases h2 : (S.links[e.1]).ends.2 ∈ cov <;>
    simp [linkβ, hX, hC, h, h1, h2]

/-- **The leak limit at a subtree, each memory challenger a sealed sink or a trusted device.** At a Site whose links
join kinds a link may join and end at places, for the subtree at `i` and a set `units` of its memory challengers that
holds every shared one: the trusted devices are the subtree's certifiers with `units`, the location the subtree's
prover places with the other challengers. With `B4`'s premises there and each trusted device's record within its `K`,
what the outside does during `[a, b)` takes at most `2 ^ (∑ K + jitter · buckets · # + β° (b − a))` values, `β°` the
model's at the Site's links, the subtree's auditor's devices covering its prover places. That a challenger outside
`units` is sealed is concluded: it is not shared, so it is linked only into the subtree, and its links end at hosts. -/
theorem leak_le_registered {Ω : Type*} (hkinds : S.LinkKinds)
    (hends : ∀ k ∈ S.links, k.ends.1 ∈ S.places ∧ k.ends.2 ∈ S.places) (pinned units : Finset String) (i : String)
    (R : Ω → Accounting.Actual.Run S.Place) (ℓ : S.Place → S.Place → ℕ)
    (link : ∀ r, Fin (R r).A.N → Fin (R r).A.N → Fin S.links.length) (𝒱 : Finset S.Place) (spec : ∀ r, (R r).A.Spec)
    (record : S.Place → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ) (K : S.Place → ℕ)
    (hu : units ⊆ S.challengersOf pinned i) (hsh : S.sharedOf pinned i ⊆ units)
    (hcs : Accounting.CausalStrategy R ℓ) (hdl : Accounting.DeclaredLinks R S.placeEnds S.linkβ link)
    (hT : Accounting.TrustedUnits R 𝒱 (S.placesIn (S.certifiersOf i ∪ units)) spec record bucket jitter)
    (hsec : Accounting.Secrets R (S.placesIn (S.certifiersOf i ∪ units))
      (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units))) a)
    (hK : Accounting.RecordsWithin R (S.placesIn (S.certifiersOf i ∪ units)) record a b K) :
    (Set.range fun r =>
        (R r).acts (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units)) ∪
          S.placesIn (S.certifiersOf i ∪ units))ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ S.placesIn (S.certifiersOf i ∪ units), K y) +
        jitter * ((Finset.Ico a b).image bucket).card * (S.placesIn (S.certifiersOf i ∪ units)).card +
        Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.placesIn (S.auditorsOf pinned i))
          (S.placesIn (S.proverUnder i)) * (b - a)) : ℕ) : ℕ∞) := by
  have hin : ∀ y ∈ S.challengersOf pinned i \ units, S.Within i y := fun y hy =>
    S.within_of_not_shared (Finset.mem_sdiff.2
      ⟨(Finset.mem_sdiff.1 hy).1, fun h => (Finset.mem_sdiff.1 hy).2 (hsh h)⟩)
  have h := Accounting.leak_le_sum_K R ℓ S.placeEnds S.linkβ link 𝒱 (S.placesIn (S.certifiersOf i ∪ units))
    (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units))) spec record bucket jitter a b K hcs hdl hT
    hsec hK
  rw [S.uncertified_registered_eq _ _ hends,
    S.uncovered_sealed (S.certifiersOf i ∪ units) (S.proverUnder i) (S.challengersOf pinned i \ units)
      (S.challengers_sealed hkinds i (S.challengersOf pinned i \ units)
        fun y hy => ⟨(S.mem_challengersOf (Finset.mem_sdiff.1 hy).1).1, hin y hy⟩),
    S.units_union_sinks hu, ← S.uncertified_registered_eq _ _ hends] at h
  exact h

end Verity.Site

namespace Accounting.SecurityProofs

theorem EgressRegistered : Guarantees.EgressRegistered := by
  intro S i
  exact ⟨fun A link h a b => S.egress_le_registered i A link h a b, S.egressLimit_le_parts i⟩

theorem LeakPartsRegistered : Guarantees.LeakPartsRegistered := by
  intro D A c l S Ω pinned units i R ℓ link 𝒱 spec record bucket jitter a b K hv hu hsh hcs hdl hT hsec hK
  exact S.leak_le_registered hv.linkKinds
    (fun k hk => ⟨S.mem_places.2 (hv.ends_known k hk).1, S.mem_places.2 (hv.ends_known k hk).2⟩) pinned units i R
    ℓ link 𝒱 spec record bucket jitter a b K hu hsh hcs hdl hT hsec hK

end Accounting.SecurityProofs
