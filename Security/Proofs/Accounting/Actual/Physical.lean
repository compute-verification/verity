import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# The physical assumptions' first consequences

* `forward_of_causal`: causality at any latency makes wires run forward in time inside every location, the form the
  split needs.
* `ingress_le`, `egress_le`: under `Links`, `in_X(I), out_X(I) ≤ β_X |I|`. A gate that `A[X, I]` reads from outside `X`
  reaches it on a link across `X`'s boundary, read during `I` at the link's end inside `X` (`inner`); a gate of `X` that
  a gate outside `X` reads during `I` leaves on one, read during `I` at its end outside (`outer`). Each counts one way
  of each link, at the readers' ticks.
* `computeRate_one`: compute rate at a pricing of at least one per computed gate bounds the number of gates, compute
  rate at one per gate.
* `state_eq_stateIn_add_compl`: the state is its part in `G` plus its part outside `G`, so memory at a tick bounds
  the part in any `G` there (`stateIn_le_of_memoryAt`).
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

open Classical in
/-- The end of link `e` in `X` (its second end, when its first isn't), and the other: for a link across `X`'s boundary,
the end inside and the end outside. -/
noncomputable def inner {E : Type*} (ends : E → L × L) (X : Finset L) (e : E) : L :=
  if (ends e).1 ∈ X then (ends e).1 else (ends e).2

open Classical in
@[inherit_doc inner]
noncomputable def outer {E : Type*} (ends : E → L × L) (X : Finset L) (e : E) : L :=
  if (ends e).1 ∈ X then (ends e).2 else (ends e).1

theorem inner_outer {E : Type*} {ends : E → L × L} {X : Finset L} {e : E} {x y : L} (hx : x ∉ X) (hy : y ∈ X)
    (he : ends e = (x, y) ∨ ends e = (y, x)) : inner ends X e = y ∧ outer ends X e = x := by
  classical
  unfold inner outer
  rcases he with he | he <;> rw [he] <;> simp [hx, hy]

/-- **Ingress is at most `β_X` a tick**: `in_X([a, b)) ≤ β_X (b − a)`. -/
theorem ingress_le {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) :
    A.ingress X (Set.Ico a b) ≤ linkCapacity ends β X * (b - a) := by
  classical
  unfold Placed.ingress linkCapacity
  rw [Finset.sum_mul]
  refine (width_mono A ?_).trans ((width_biUnion_le A _ _).trans
    (Finset.sum_le_sum fun e _ => h.2 e (inner ends X e) a b))
  intro c hc
  obtain ⟨hc, hcX⟩ := mem_filter.1 hc
  obtain ⟨-, g, hg, hcg⟩ := (mem_into A).1 hc
  obtain ⟨-, hgX, hgt⟩ := (mem_region A).1 hg
  have hne : A.x c ≠ A.x g := fun h' => hcX (h' ▸ hgX)
  have he := (h.1 g c hcg hne).2
  refine mem_biUnion.2 ⟨link g c, mem_boundary hcX hgX he, ?_⟩
  simp only [Placed.readOn, mem_filter, mem_univ, true_and]
  exact ⟨g, hcg, hne, rfl, (inner_outer hcX hgX he).1.symm, hgt⟩

/-- **Egress is at most `β_X` a tick**: `out_X([a, b)) ≤ β_X (b − a)`. -/
theorem egress_le {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X : Finset L) (a b : ℕ) :
    A.egress X (Set.Ico a b) ≤ linkCapacity ends β X * (b - a) := by
  classical
  unfold Placed.egress linkCapacity
  rw [Finset.sum_mul]
  refine (width_mono A ?_).trans ((width_biUnion_le A _ _).trans
    (Finset.sum_le_sum fun e _ => h.2 e (outer ends X e) a b))
  intro g hg
  simp only [mem_filter, mem_univ, true_and] at hg
  obtain ⟨hgX, k, hkX, hkt, hgk⟩ := hg
  have hne : A.x g ≠ A.x k := fun h' => hkX (h' ▸ hgX)
  have he := (h.1 k g hgk hne).2.symm
  refine mem_biUnion.2 ⟨link k g, mem_boundary hkX hgX he, ?_⟩
  simp only [Placed.readOn, mem_filter, mem_univ, true_and]
  exact ⟨k, hgk, hne, rfl, (inner_outer hkX hgX he).2.symm, hkt⟩

theorem width_inter_add_compl (S G : Finset (Fin A.N)) : A.width S = A.width (S ∩ G) + A.width (S ∩ Gᶜ) := by
  have hc : S ∩ Gᶜ = S \ G := by
    ext a
    simp
  rw [hc]
  unfold Circuit.width
  exact (sum_inter_add_sum_sdiff S G A.w).symm

/-- **The state splits along any set of gates**: `mem_X(t)` is its part in `G` plus its part outside `G`. -/
theorem state_eq_stateIn_add_compl (X : Finset L) (t : ℕ) (G : Finset (Fin A.N)) :
    A.state X t = A.stateIn X t G + A.stateIn X t Gᶜ := by
  unfold Placed.stateIn Placed.state
  exact width_inter_add_compl A _ G

theorem stateIn_le_state (X : Finset L) (t : ℕ) (G : Finset (Fin A.N)) : A.stateIn X t G ≤ A.state X t := by
  rw [state_eq_stateIn_add_compl A X t G]
  exact Nat.le_add_right _ _

theorem memoryAt_of_memory {𝒳 : Set (Finset L)} {M : Finset L → ℕ} (h : A.Memory 𝒳 M) (T : Set ℕ) :
    A.MemoryAt 𝒳 M T :=
  fun X hX t _ => h X hX t

/-- Memory at a tick bounds the state's part in any set of gates there. -/
theorem stateIn_le_of_memoryAt {𝒳 : Set (Finset L)} {M : Finset L → ℕ} {T : Set ℕ} (h : A.MemoryAt 𝒳 M T)
    {X : Finset L} (hX : X ∈ 𝒳) {t : ℕ} (ht : t ∈ T) (G : Finset (Fin A.N)) : A.stateIn X t G ≤ M X :=
  (stateIn_le_state A X t G).trans (h X hX t ht)

/-- **Pricing at least one per gate bounds the gates**: compute rate at a pricing `π ≥ 1` on computed gates is compute
rate at one per gate. -/
theorem computeRate_one {𝒳 : Set (Finset L)} {π : Fin A.N → ℕ} {compute : Finset L → ℕ}
    (h : A.ComputeRate 𝒳 π compute) (hπ : ∀ g, A.isInput g = false → 1 ≤ π g) :
    A.ComputeRate 𝒳 (fun _ => 1) compute := fun X hX a b =>
  (Finset.sum_le_sum fun g hg => hπ g ((mem_region A).1 hg).1).trans (h X hX a b)

end Accounting.SecurityProofs.Actual
