import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Defs

/-!
# Hiding, from events to tests

`Hides.avg_le`: a registration that hides every event up to `ε` hides every test with values in `[0, 1]` up to `ε`: at
the set `A` of views more likely at `p` than at `p'`, the test's gain is at most `Pr_p[A] - Pr_{p'}[A]`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open FlockSoundness.Game
open scoped ENNReal

theorem prCoin_le_one' {C : Type} [Fintype C] (p : C → Prop) : prCoin p ≤ 1 := by
  classical
  rw [prCoin_eq_card, ENNReal.div_le_iff_le_mul (Or.inr ENNReal.one_ne_top) (Or.inl (ENNReal.natCast_ne_top _)),
    one_mul]
  exact_mod_cast Finset.card_le_univ _

theorem avg_const_le {C : Type} [Fintype C] (x : ℝ≥0∞) : avg (fun _ : C => x) ≤ x := by
  rcases isEmpty_or_nonempty C with hC | hC
  · simp [avg]
  · exact (avg_const x).le

theorem avg_eq_mul_sum {C : Type} [Fintype C] (f : C → ℝ≥0∞) : avg f = (Fintype.card C : ℝ≥0∞)⁻¹ * ∑ c, f c := by
  rw [avg, Finset.mul_sum]

/-- A sum over the salts, regrouped by the view. -/
theorem sum_fiber {R V : Type} [Fintype R] [DecidableEq V] (vx : R → V) (S : Finset V) (hS : ∀ r, vx r ∈ S)
    (g : V → ℝ≥0∞) :
    ∑ r, g (vx r) = ∑ v ∈ S, ((Finset.univ.filter fun r => vx r = v).card : ℝ≥0∞) * g v := by
  rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (t := S) (g := vx) (fun r _ => hS r)]
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [Finset.sum_congr rfl fun r hr => by rw [(Finset.mem_filter.1 hr).2], Finset.sum_const, nsmul_eq_mul]

/-- The salts whose view lies in `A ⊆ S`, counted by view. -/
theorem card_in {R V : Type} [Fintype R] [DecidableEq V] (vx : R → V) (A : Finset V) :
    ((Finset.univ.filter fun r => vx r ∈ A).card : ℝ≥0∞) =
      ∑ v ∈ A, ((Finset.univ.filter fun r => vx r = v).card : ℝ≥0∞) := by
  rw [Finset.card_eq_sum_card_fiberwise (s := Finset.univ.filter fun r => vx r ∈ A) (f := vx) (t := A)
    (fun r hr => (Finset.mem_filter.1 hr).2)]
  push_cast
  refine Finset.sum_congr rfl fun v hv => ?_
  congr 2
  ext r
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  exact ⟨fun h => h.2, fun h => ⟨h ▸ hv, h⟩⟩

/-- **Hiding every event is hiding every test**: for `f` with values in `[0, 1]`, `E_p[f] ≤ E_{p'}[f] + ε`. -/
theorem Hides.avg_le {P R V : Type} [Fintype R] {reg : P → R → V} {ε : ℝ≥0∞} (h : Hides reg ε) (p p' : P)
    (f : V → ℝ≥0∞) (hf : ∀ v, f v ≤ 1) :
    avg (fun r => f (reg p r)) ≤ avg (fun r => f (reg p' r)) + ε := by
  classical
  rcases isEmpty_or_nonempty R with hR | hR
  · simp [avg]
  set vx := reg p
  set vy := reg p'
  set S : Finset V := Finset.univ.image vx ∪ Finset.univ.image vy
  have hXS : ∀ r, vx r ∈ S := fun r => Finset.mem_union_left _ (Finset.mem_image_of_mem _ (Finset.mem_univ r))
  have hYS : ∀ r, vy r ∈ S := fun r => Finset.mem_union_right _ (Finset.mem_image_of_mem _ (Finset.mem_univ r))
  set nX : V → ℝ≥0∞ := fun v => ((Finset.univ.filter fun r => vx r = v).card : ℝ≥0∞)
  set nY : V → ℝ≥0∞ := fun v => ((Finset.univ.filter fun r => vy r = v).card : ℝ≥0∞)
  set A : Finset V := S.filter fun v => (Finset.univ.filter fun r => vy r = v).card <
    (Finset.univ.filter fun r => vx r = v).card
  -- per view: `nX·f + nY·[A] ≤ nY·f + nX·[A]`
  have hv : ∀ v ∈ S, nX v * f v + (if v ∈ A then nY v else 0) ≤ nY v * f v + (if v ∈ A then nX v else 0) := by
    intro v hvS
    by_cases hA : v ∈ A
    · simp only [hA, ↓reduceIte]
      obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_lt (Finset.mem_filter.1 hA).2
      have e : nX v = nY v + (d + 1 : ℕ) := by simp only [nX, nY, hd]; push_cast; ring
      rw [e, add_mul]
      calc nY v * f v + ((d + 1 : ℕ) : ℝ≥0∞) * f v + nY v
          = nY v * f v + (((d + 1 : ℕ) : ℝ≥0∞) * f v + nY v) := by ring
        _ ≤ nY v * f v + (((d + 1 : ℕ) : ℝ≥0∞) + nY v) := by gcongr; exact mul_le_of_le_one_right' (hf v)
        _ = nY v * f v + (nY v + ((d + 1 : ℕ) : ℝ≥0∞)) := by ring
    · simp only [hA, ↓reduceIte, add_zero]
      have hle : (Finset.univ.filter fun r => vx r = v).card ≤ (Finset.univ.filter fun r => vy r = v).card :=
        not_lt.1 fun hlt => hA (Finset.mem_filter.2 ⟨hvS, hlt⟩)
      gcongr
      exact Nat.cast_le.2 hle
  have hsum := Finset.sum_le_sum hv
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.sum_filter, ← Finset.sum_filter] at hsum
  -- the event `A`
  have hE := h p p' fun v => v ∈ A
  rw [prCoin_eq_card, prCoin_eq_card, ENNReal.div_eq_inv_mul, ENNReal.div_eq_inv_mul, card_in vx A, card_in vy A] at hE
  have hS2 : (Finset.filter (fun v => v ∈ A) S) = A := by
    ext v; simp only [Finset.mem_filter, and_iff_right_iff_imp]; exact fun hv => (Finset.mem_filter.1 hv).1
  rw [hS2] at hsum
  set N := (Fintype.card R : ℝ≥0∞)⁻¹
  have hfin : N * ∑ v ∈ A, nY v ≠ ⊤ := by
    refine ne_top_of_le_ne_top ENNReal.one_ne_top ?_
    have := prCoin_le_one' (fun r => vy r ∈ A)
    rwa [prCoin_eq_card, ENNReal.div_eq_inv_mul, card_in vy A] at this
  rw [avg_eq_mul_sum, avg_eq_mul_sum, sum_fiber vx S hXS, sum_fiber vy S hYS]
  refine ENNReal.le_of_add_le_add_right hfin ?_
  calc N * ∑ v ∈ S, nX v * f v + N * ∑ v ∈ A, nY v
      = N * (∑ v ∈ S, nX v * f v + ∑ v ∈ A, nY v) := by rw [mul_add]
    _ ≤ N * (∑ v ∈ S, nY v * f v + ∑ v ∈ A, nX v) := by gcongr
    _ = N * ∑ v ∈ S, nY v * f v + N * ∑ v ∈ A, nX v := by rw [mul_add]
    _ ≤ N * ∑ v ∈ S, nY v * f v + (N * ∑ v ∈ A, nY v + ε) := by gcongr
    _ = N * ∑ v ∈ S, nY v * f v + ε + N * ∑ v ∈ A, nY v := by ring

end FlockSoundness.Discharge.PrivateCircuit
