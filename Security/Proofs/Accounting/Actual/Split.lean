import Definitions.Accounting.Actual
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The split: what a region reads is what its location carried in plus what crossed its boundary

A gate that `A[X, [t, t'))` reads from outside itself is placed outside `X`, and counts in `ingress X [t, t')`, or in
`X`. In `X` it is an input gate, or a computed gate placed before `t`: one placed at or after `t'` would be read by a gate
of the region, which runs before `t'`, before its own tick, which `Forward X` refuses. Either way `A[X, [t, ∞))` reads it
from outside itself, so it counts in `state X t`. The same holds of the reads in any set `G` of gates, with
`stateIn X t G` in place of `state X t` (`width_into_region_inter_le`).
-/

namespace Accounting.SecurityProofs.Actual

open Verity Finset

variable {L : Type*} (A : Placed L)

theorem mem_region {X : Finset L} {I : Set ℕ} {g : Fin A.N} :
    g ∈ A.region X I ↔ A.isInput g = false ∧ A.x g ∈ X ∧ A.t g ∈ I := by
  simp [Placed.region, Circuit.computed]

theorem mem_into {S : Finset (Fin A.N)} {a : Fin A.N} :
    a ∈ A.into S ↔ a ∉ S ∧ ∃ g ∈ S, a ∈ A.reads g := by
  simp [Circuit.into]

theorem width_mono {S T : Finset (Fin A.N)} (h : S ⊆ T) : A.width S ≤ A.width T :=
  Finset.sum_le_sum_of_subset h

theorem width_union_le (S T : Finset (Fin A.N)) : A.width (S ∪ T) ≤ A.width S + A.width T := by
  classical
  have := Finset.sum_union_inter (s₁ := S) (s₂ := T) (f := A.w)
  unfold Circuit.width
  omega

theorem stateIn_univ (X : Finset L) (t : ℕ) : A.stateIn X t univ = A.state X t := by
  unfold Placed.stateIn Placed.state
  rw [inter_univ]

/-- **The split in `G`**: what `A[X, [t, t'))` reads from outside itself in the gates `G` is at most the part in `G` of
what `X` carries across `t` plus what crossed `X`'s boundary during `[t, t')`, when wires run forward in time inside
`X`. -/
theorem width_into_region_inter_le (X : Finset L) (t t' : ℕ) (G : Finset (Fin A.N)) (h : A.Forward X) :
    A.width (A.into (A.region X (Set.Ico t t')) ∩ G) ≤ A.stateIn X t G + A.ingress X (Set.Ico t t') := by
  classical
  unfold Placed.stateIn Placed.ingress
  refine (width_mono A fun a ha => ?_).trans (width_union_le A _ _)
  obtain ⟨hai, haG⟩ := mem_inter.1 ha
  rw [mem_union, mem_inter, mem_filter, mem_filter]
  obtain ⟨ha, g, hg, hag⟩ := (mem_into A).1 hai
  by_cases hx : A.x a ∈ X
  · refine Or.inl ⟨⟨(mem_into A).2 ⟨fun ha' => ha ?_, g, ?_, hag⟩, hx⟩, haG⟩
    · obtain ⟨hai, -, hat⟩ := (mem_region A).1 ha'
      obtain ⟨-, hgx, -, hgt⟩ := (mem_region A).1 hg
      exact (mem_region A).2 ⟨hai, hx, hat, (h g a hag hai hx hgx).trans_lt hgt⟩
    · obtain ⟨hgi, hgx, hgt, -⟩ := (mem_region A).1 hg
      exact (mem_region A).2 ⟨hgi, hgx, hgt⟩
  · exact Or.inr ⟨hai, hx⟩

/-- **The split**: what `A[X, [t, t'))` reads from outside itself is at most what `X` carries across `t` plus what
crossed `X`'s boundary during `[t, t')`, when wires run forward in time inside `X`. -/
theorem width_into_region_le (X : Finset L) (t t' : ℕ) (h : A.Forward X) :
    A.width (A.into (A.region X (Set.Ico t t'))) ≤ A.state X t + A.ingress X (Set.Ico t t') := by
  have := width_into_region_inter_le A X t t' univ h
  rwa [inter_univ, stateIn_univ] at this

end Accounting.SecurityProofs.Actual
