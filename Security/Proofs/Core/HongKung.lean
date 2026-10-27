import Mathlib

/-!
# Loomis–Whitney, and Hong–Kung's count of a product's segments

A part of the SAK lower bound (`Pouw.SakLb`): the combinatorial core of the data-movement lower bound for an exact
matrix product.

* `loomis_whitney`: a finite set `S` of triples `(i, j, l)` has `|S|² ≤ |S_ij| · |S_jl| · |S_il|`, its three
  projections (the discrete Loomis–Whitney inequality, by Cauchy–Schwarz on the fibres over `(i, j)`).
* `segment_le`: a set of MACs `A_ij B_jl → C_il` that touches at most `h` entries of each of `A`, `B` and `C` has at
  most `h^{3/2}` of them: `|S|² ≤ h³`.
* `parts_le`: if the `m k n` MACs of a product are done in `T` segments, each touching at most `h` entries of each
  operand and of the result, then `(m k n)² ≤ T² h³`.

Hong–Kung's argument reads `parts_le` at `h = 2M` (scaled by the words an entry takes): a run with `M` words of fast
memory, cut into segments of `M` words moved, touches in each segment at most the `M` words it holds at the segment's
start and the `M` it moves, so it moves at least `M (T − 1) ≥ m k n / (2√2 √M) − M` words. At the served shapes and the
cards' rates this is at most 2.8% of the tensor units' time, so the bound does not bind: P1's floor on the tensor
units does.
-/

namespace Verity.Accel.HongKung

open Finset

variable {X Y Z : Type*} [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

/-- The `(i, j)` projection: the entries of `A` a set of MACs reads. -/
def pxy (S : Finset (X × Y × Z)) : Finset (X × Y) := S.image fun p => (p.1, p.2.1)

/-- The `(j, l)` projection: the entries of `B`. -/
def pyz (S : Finset (X × Y × Z)) : Finset (Y × Z) := S.image fun p => (p.2.1, p.2.2)

/-- The `(i, l)` projection: the entries of `C`. -/
def pxz (S : Finset (X × Y × Z)) : Finset (X × Z) := S.image fun p => (p.1, p.2.2)

theorem fibre_le_yz (S : Finset (X × Y × Z)) (q : X × Y) :
    #{p ∈ S | (p.1, p.2.1) = q} ≤ #{r ∈ pyz S | r.1 = q.2} := by
  apply card_le_card_of_injOn (fun p => (p.2.1, p.2.2))
  · intro p hp
    simp only [coe_filter, Set.mem_ofPred_eq] at hp
    simp only [coe_filter, Set.mem_ofPred_eq, pyz, mem_image]
    refine ⟨⟨p, hp.1, rfl⟩, ?_⟩
    rw [← hp.2]
  · intro p hp p' hp' h
    simp only [coe_filter, Set.mem_ofPred_eq] at hp hp'
    simp only [Prod.mk.injEq] at h
    have h1 : p.1 = p'.1 := by
      have := congrArg Prod.fst hp.2; have := congrArg Prod.fst hp'.2; simp_all
    exact Prod.ext h1 (Prod.ext h.1 h.2)

theorem fibre_le_xz (S : Finset (X × Y × Z)) (q : X × Y) :
    #{p ∈ S | (p.1, p.2.1) = q} ≤ #{r ∈ pxz S | r.1 = q.1} := by
  apply card_le_card_of_injOn (fun p => (p.1, p.2.2))
  · intro p hp
    simp only [coe_filter, Set.mem_ofPred_eq] at hp
    simp only [coe_filter, Set.mem_ofPred_eq, pxz, mem_image]
    refine ⟨⟨p, hp.1, rfl⟩, ?_⟩
    rw [← hp.2]
  · intro p hp p' hp' h
    simp only [coe_filter, Set.mem_ofPred_eq] at hp hp'
    simp only [Prod.mk.injEq] at h
    have h2 : p.2.1 = p'.2.1 := by
      have := congrArg Prod.snd hp.2; have := congrArg Prod.snd hp'.2; simp_all
    exact Prod.ext h.1 (Prod.ext h2 h.2)

/-- **Discrete Loomis–Whitney**: `|S|² ≤ |S_ij| · |S_jl| · |S_il|`. -/
theorem loomis_whitney (S : Finset (X × Y × Z)) : #S ^ 2 ≤ #(pxy S) * #(pyz S) * #(pxz S) := by
  set a : Y → ℕ := fun y => #{r ∈ pyz S | r.1 = y}
  set b : X → ℕ := fun x => #{r ∈ pxz S | r.1 = x}
  set Xs := S.image fun p => p.1
  set Ys := S.image fun p => p.2.1
  have hS : #S = ∑ q ∈ pxy S, #{p ∈ S | (p.1, p.2.1) = q} :=
    card_eq_sum_card_fiberwise fun p hp => mem_image_of_mem _ hp
  have hfib : ∀ q, #{p ∈ S | (p.1, p.2.1) = q} ^ 2 ≤ a q.2 * b q.1 := by
    intro q
    rw [sq]
    exact Nat.mul_le_mul (fibre_le_yz S q) (fibre_le_xz S q)
  have hsub : pxy S ⊆ Xs ×ˢ Ys := by
    intro q hq
    simp only [pxy, mem_image] at hq
    obtain ⟨p, hp, rfl⟩ := hq
    exact mem_product.mpr ⟨mem_image_of_mem _ hp, mem_image_of_mem _ hp⟩
  have ha : ∑ y ∈ Ys, a y = #(pyz S) := by
    refine (card_eq_sum_card_fiberwise fun r hr => ?_).symm
    simp only [pyz, coe_image, Set.mem_image, mem_coe] at hr
    obtain ⟨p, hp, rfl⟩ := hr
    exact mem_image_of_mem _ hp
  have hb : ∑ x ∈ Xs, b x = #(pxz S) := by
    refine (card_eq_sum_card_fiberwise fun r hr => ?_).symm
    simp only [pxz, coe_image, Set.mem_image, mem_coe] at hr
    obtain ⟨p, hp, rfl⟩ := hr
    exact mem_image_of_mem _ hp
  calc #S ^ 2 = (∑ q ∈ pxy S, #{p ∈ S | (p.1, p.2.1) = q}) ^ 2 := by rw [hS]
    _ ≤ #(pxy S) * ∑ q ∈ pxy S, #{p ∈ S | (p.1, p.2.1) = q} ^ 2 := sq_sum_le_card_mul_sum_sq
    _ ≤ #(pxy S) * ∑ q ∈ pxy S, a q.2 * b q.1 := Nat.mul_le_mul_left _ (sum_le_sum fun q _ => hfib q)
    _ ≤ #(pxy S) * ∑ q ∈ Xs ×ˢ Ys, a q.2 * b q.1 :=
        Nat.mul_le_mul_left _ (sum_le_sum_of_subset hsub)
    _ = #(pxy S) * ((∑ y ∈ Ys, a y) * ∑ x ∈ Xs, b x) := by
        rw [sum_product, sum_mul_sum, sum_comm]
    _ = #(pxy S) * #(pyz S) * #(pxz S) := by rw [ha, hb, mul_assoc]

/-- **A segment's MACs**: touching at most `h` entries of each of `A`, `B` and `C`, a set of MACs has at most `h^{3/2}`
of them. -/
theorem segment_le (S : Finset (X × Y × Z)) (h : ℕ) (h1 : #(pxy S) ≤ h) (h2 : #(pyz S) ≤ h) (h3 : #(pxz S) ≤ h) :
    #S ^ 2 ≤ h ^ 3 := by
  calc #S ^ 2 ≤ #(pxy S) * #(pyz S) * #(pxz S) := loomis_whitney S
    _ ≤ h * h * h := Nat.mul_le_mul (Nat.mul_le_mul h1 h2) h3
    _ = h ^ 3 := by ring

/-- **Hong–Kung's count of segments**: if the `m k n` MACs of a product (every triple of `X × Y × Z`) are done in `T`
segments, each touching at most `h` entries of each operand and of the result, then `(m k n)² ≤ T² h³`. -/
theorem parts_le [Fintype X] [Fintype Y] [Fintype Z] {T : ℕ} (P : Fin T → Finset (X × Y × Z))
    (hcover : ∀ p, ∃ t, p ∈ P t) (h : ℕ) (hseg : ∀ t, #(pxy (P t)) ≤ h ∧ #(pyz (P t)) ≤ h ∧ #(pxz (P t)) ≤ h) :
    (Fintype.card X * Fintype.card Y * Fintype.card Z) ^ 2 ≤ T ^ 2 * h ^ 3 := by
  have hc : Fintype.card X * Fintype.card Y * Fintype.card Z ≤ ∑ t, #(P t) := by
    have : (univ : Finset (X × Y × Z)) ⊆ univ.biUnion P := by
      intro p _; obtain ⟨t, ht⟩ := hcover p; exact mem_biUnion.mpr ⟨t, mem_univ _, ht⟩
    calc _ = #(univ : Finset (X × Y × Z)) := by simp [Fintype.card_prod, mul_assoc]
      _ ≤ #(univ.biUnion P) := card_le_card this
      _ ≤ ∑ t, #(P t) := card_biUnion_le
  calc _ ≤ (∑ t, #(P t)) ^ 2 := Nat.pow_le_pow_left hc 2
    _ ≤ #(univ : Finset (Fin T)) * ∑ t, #(P t) ^ 2 := sq_sum_le_card_mul_sum_sq
    _ ≤ T * ∑ _t : Fin T, h ^ 3 := by
        rw [card_univ, Fintype.card_fin]
        exact Nat.mul_le_mul_left _ (sum_le_sum fun t _ =>
          segment_le (P t) h (hseg t).1 (hseg t).2.1 (hseg t).2.2)
    _ = T ^ 2 * h ^ 3 := by simp [sum_const, card_univ]; ring

/-- **Honest**: the cube `[a] × [a] × [a]` meets Loomis–Whitney with equality, `(a³)² = (a²)³`. -/
theorem loomis_whitney_cube (a : ℕ) :
    #(univ : Finset (Fin a × Fin a × Fin a)) ^ 2 =
      #(pxy (univ : Finset (Fin a × Fin a × Fin a))) * #(pyz (univ : Finset (Fin a × Fin a × Fin a))) *
        #(pxz (univ : Finset (Fin a × Fin a × Fin a))) := by
  have e1 : pxy (univ : Finset (Fin a × Fin a × Fin a)) = univ := by
    ext q; simp only [pxy, mem_image, mem_univ, true_and, iff_true]
    rcases Nat.eq_zero_or_pos a with rfl | ha
    · exact q.1.elim0
    · exact ⟨(q.1, q.2, ⟨0, ha⟩), rfl⟩
  have e2 : pyz (univ : Finset (Fin a × Fin a × Fin a)) = univ := by
    ext q; simp only [pyz, mem_image, mem_univ, true_and, iff_true]
    rcases Nat.eq_zero_or_pos a with rfl | ha
    · exact q.1.elim0
    · exact ⟨(⟨0, ha⟩, q.1, q.2), rfl⟩
  have e3 : pxz (univ : Finset (Fin a × Fin a × Fin a)) = univ := by
    ext q; simp only [pxz, mem_image, mem_univ, true_and, iff_true]
    rcases Nat.eq_zero_or_pos a with rfl | ha
    · exact q.1.elim0
    · exact ⟨(q.1, ⟨0, ha⟩, q.2), rfl⟩
  rw [e1, e2, e3]
  simp only [card_univ, Fintype.card_prod, Fintype.card_fin]
  ring

end Verity.Accel.HongKung
