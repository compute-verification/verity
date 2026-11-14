module

public import Definitions.Accounting.Guarantees.RegisteredSite
public import Proofs.Accounting.Actual.Physical
public import Proofs.Core.SiteSubtree
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# A subtree's egress limit, read off the registered links

* `linkCapacity_union_le`, `linkCapacity_biUnion_le`: in the model, the capacity across the boundary of a union of
  parts is at most the parts' capacities summed, since a link across the union's boundary crosses a part's.
* `Site.egress_le_registered`: `egress_le` at the Site's places and links and a subtree.
* `Site.egressLimit_le_parts`: a subtree's limit is at most its root's alone plus its root's children's subtrees'
  (`Site.under_eq`).
* `EgressRegistered`: the two together, the guarantee.
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

end Verity.Site

namespace Accounting.SecurityProofs

theorem EgressRegistered : Guarantees.EgressRegistered := by
  intro S i
  exact ⟨fun A link h a b => S.egress_le_registered i A link h a b, S.egressLimit_le_parts i⟩

end Accounting.SecurityProofs
