import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# The physical assumptions' first consequences

* `forward_of_causal`: causality at any latency makes wires run forward in time inside every location, the form the
  split needs.
* `ingress_le`, `egress_le`: under `Links`, `in_X(I), out_X(I) ≤ β_X |I|`. A gate that `A[X, I]` reads from outside `X`
  reaches it on a link across `X`'s boundary, read during `I`; a gate of `A[X, I]` read outside `X` leaves on one, sent
  during `I`.
* `computeRate_one`: compute rate at a pricing of at least one per computed gate bounds the number of gates, compute
  rate at one per gate.
-/

namespace Accounting.SecurityProofs.Actual

open Verity Finset

variable {L : Type*} (A : Placed L)

theorem forward_of_causal {ℓ : L → L → ℕ} (h : A.Causal ℓ) (X : Finset L) : A.Forward X :=
  fun g a ha _ _ _ => (Nat.le_add_right _ _).trans (h g a ha)

theorem causal_of_feasible {E : Type*} {H : Hardware L E} {ann : Annotation A E} (h : Feasible H A ann) :
    A.Causal H.ℓ := h.1

theorem width_biUnion_le {ι : Type*} (s : Finset ι) (f : ι → Finset (Fin A.N)) :
    A.width (s.biUnion f) ≤ ∑ i ∈ s, A.width (f i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Circuit.width]
  | insert i s hi ih =>
    rw [biUnion_insert, sum_insert hi]
    exact (width_union_le A _ _).trans (Nat.add_le_add_left ih _)

theorem mem_boundary {E : Type*} [Fintype E] {ends : E → L × L} {X : Finset L} {e : E} {x y : L} (hx : x ∉ X)
    (hy : y ∈ X) (he : ends e = (x, y) ∨ ends e = (y, x)) : e ∈ boundary ends X := by
  classical
  simp only [boundary, mem_filter, mem_univ, true_and]
  rcases he with he | he <;> rw [he] <;> simp [hx, hy]

/-- **Ingress is at most `β_X` a tick**: `in_X([a, b)) ≤ β_X (b − a)`. -/
theorem ingress_le {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) :
    A.ingress X (Set.Ico a b) ≤ linkCapacity ends β X * (b - a) := by
  classical
  unfold Placed.ingress linkCapacity
  rw [Finset.sum_mul]
  refine (width_mono A ?_).trans ((width_biUnion_le A _ _).trans (Finset.sum_le_sum fun e _ => (h.2 e a b).1))
  intro c hc
  obtain ⟨hc, hcX⟩ := mem_filter.1 hc
  obtain ⟨-, g, hg, hcg⟩ := (mem_into A).1 hc
  obtain ⟨-, hgX, hgt⟩ := (mem_region A).1 hg
  have hne : A.x c ≠ A.x g := fun h' => hcX (h' ▸ hgX)
  refine mem_biUnion.2 ⟨link g c, mem_boundary hcX hgX (h.1 g c hcg hne), ?_⟩
  simp only [Placed.readOn, mem_filter, mem_univ, true_and]
  exact ⟨g, hcg, hne, rfl, hgt⟩

/-- **Egress is at most `β_X` a tick**: `out_X([a, b)) ≤ β_X (b − a)`. -/
theorem egress_le {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) :
    A.egress X (Set.Ico a b) ≤ linkCapacity ends β X * (b - a) := by
  classical
  unfold Placed.egress linkCapacity
  rw [Finset.sum_mul]
  refine (width_mono A ?_).trans ((width_biUnion_le A _ _).trans (Finset.sum_le_sum fun e _ => (h.2 e a b).2))
  intro g hg
  obtain ⟨hg, k, hkX, hgk⟩ := mem_filter.1 hg
  obtain ⟨-, hgX, hgt⟩ := (mem_region A).1 hg
  have hne : A.x g ≠ A.x k := fun h' => hkX (h' ▸ hgX)
  have he := h.1 k g hgk hne
  refine mem_biUnion.2 ⟨link k g, mem_boundary hkX hgX he.symm, ?_⟩
  simp only [Placed.sentOn, mem_filter, mem_univ, true_and]
  exact ⟨hgt, k, hgk, hne, rfl⟩

/-- **Pricing at least one per gate bounds the gates**: compute rate at a pricing `π ≥ 1` on computed gates is compute
rate at one per gate. -/
theorem computeRate_one {𝒳 : Set (Finset L)} {π : Fin A.N → ℕ} {compute : Finset L → ℕ}
    (h : A.ComputeRate 𝒳 π compute) (hπ : ∀ g, A.isInput g = false → 1 ≤ π g) :
    A.ComputeRate 𝒳 (fun _ => 1) compute := fun X hX a b =>
  (Finset.sum_le_sum fun g hg => hπ g ((mem_region A).1 hg).1).trans (h X hX a b)

end Accounting.SecurityProofs.Actual
