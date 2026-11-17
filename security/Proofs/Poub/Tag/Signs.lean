module

public import Proofs.Poub.Tag.Basic
public import Definitions.Poub.Guarantees.Sketch

/-! ±1 rows do not collide: at most half the rows have a given inner product with a nonzero integer vector. -/

@[expose] public section

open Verity.Game renaming prPMF → prob

namespace ResourceAccounting.TagProofs

open ResourceAccounting.Tag Finset
open scoped ENNReal

theorem prob_uniform_le_inv_pow {α : Type} [Fintype α] [Nonempty α] (E : α → Prop) [DecidablePred E] (k : ℕ)
    (h : 2 ^ k * (univ.filter E).card ≤ Fintype.card α) :
    prob (PMF.uniformOfFintype α) E ≤ (2 ^ k : ℝ≥0∞)⁻¹ := by
  refine (prob_uniform_le E _ le_rfl).trans ?_
  rw [ENNReal.div_le_iff (by simp [Fintype.card_ne_zero]) (by simp)]
  calc ((univ.filter E).card : ℝ≥0∞) = (2 ^ k)⁻¹ * (2 ^ k * (univ.filter E).card) := by
        rw [← mul_assoc, ENNReal.inv_mul_cancel (by simp) (by simp), one_mul]
    _ ≤ (2 ^ k)⁻¹ * Fintype.card α := by gcongr; exact_mod_cast h

theorem card_level_le {R : Type} [Fintype R] (inner : R → ℤ) (f : R → R) (hf : Function.Involutive f)
    (hne : ∀ r, inner (f r) ≠ inner r) (t : ℤ) :
    2 * (univ.filter fun r => inner r = t).card ≤ Fintype.card R := by
  classical
  set A := univ.filter fun r => inner r = t with hA
  have hdisj : Disjoint A (A.image f) := by
    rw [Finset.disjoint_left]
    intro r hr hr'
    obtain ⟨r', hr'A, rfl⟩ := Finset.mem_image.1 hr'
    simp only [hA, mem_filter, mem_univ, true_and] at hr hr'A
    exact hne r' (hr.trans hr'A.symm)
  have hcard : (A.image f).card = A.card := Finset.card_image_of_injective _ hf.injective
  calc 2 * A.card = (A ∪ A.image f).card := by rw [card_union_of_disjoint hdisj, hcard]; ring
    _ ≤ Fintype.card R := card_le_univ _

theorem card_rows_le {R : Type} [Fintype R] {k : ℕ} (inner : R → ℤ) (f : R → R) (hf : Function.Involutive f)
    (hne : ∀ r, inner (f r) ≠ inner r) (t : Fin k → ℤ) :
    2 ^ k * (univ.filter fun S : Fin k → R => ∀ i, inner (S i) = t i).card ≤ Fintype.card (Fin k → R) := by
  classical
  have hset : (univ.filter fun S : Fin k → R => ∀ i, inner (S i) = t i) =
      Fintype.piFinset fun i => univ.filter fun r => inner r = t i := by
    ext S
    simp
  rw [hset, Fintype.card_piFinset, Fintype.card_fun, Fintype.card_fin]
  calc 2 ^ k * ∏ i, (univ.filter fun r => inner r = t i).card
      = ∏ i : Fin k, 2 * (univ.filter fun r => inner r = t i).card := by
        rw [prod_mul_distrib, prod_const, card_univ, Fintype.card_fin]
    _ ≤ ∏ _i : Fin k, Fintype.card R :=
        prod_le_prod fun i _ => card_level_le inner f hf hne (t i)
    _ = Fintype.card R ^ k := by rw [prod_const, card_univ, Fintype.card_fin]

theorem sgn_not_sub (b : Bool) : sgn (!b) - sgn b = if b then 2 else -2 := by
  cases b <;> simp [sgn]

theorem inner_flip {ι : Type} [Fintype ι] [DecidableEq ι] (d : ι → ℤ) (r : ι → Bool) (j : ι) :
    ∑ i, sgn (Function.update r j (!r j) i) * d i - ∑ i, sgn (r i) * d i = (sgn (!r j) - sgn (r j)) * d j := by
  rw [← Finset.sum_sub_distrib, Finset.sum_eq_single j]
  · rw [Function.update_self, sub_mul]
  · intro i _ hi
    rw [Function.update_of_ne hi, sub_self]
  · simp

theorem flip_moves {ι : Type} [Fintype ι] [DecidableEq ι] (d : ι → ℤ) (j : ι) (hj : d j ≠ 0) :
    Function.Involutive (fun r : ι → Bool => Function.update r j (!r j)) ∧
      ∀ r : ι → Bool, ∑ i, sgn (Function.update r j (!r j) i) * d i ≠ ∑ i, sgn (r i) * d i := by
  refine ⟨fun r => ?_, fun r h => ?_⟩
  · simp
  · have e := inner_flip d r j
    rw [h, sub_self, sgn_not_sub] at e
    cases hb : r j <;> simp [hb] at e <;> exact hj (by omega)

theorem nonCollision : ResourceAccounting.Guarantees.Sketch.NonCollision := by
  intro k n x y
  classical
  apply prob_uniform_le_inv_pow
  by_cases hxy : x = y
  · have : (univ.filter fun S : Signs k n => x ≠ y ∧ project S x = project S y) = ∅ := by
      ext S
      simp [hxy]
    rw [this, card_empty, mul_zero]
    exact Nat.zero_le _
  · obtain ⟨j, hj⟩ : ∃ j, x j - y j ≠ 0 := by
      by_contra h
      push Not at h
      exact hxy (funext fun j => sub_eq_zero.1 (h j))
    obtain ⟨hf, hne⟩ := flip_moves (fun i => x i - y i) j hj
    have hset : (univ.filter fun S : Signs k n => x ≠ y ∧ project S x = project S y) =
        univ.filter fun S : Signs k n => ∀ i, ∑ l, sgn (S i l) * (x l - y l) = (fun _ => (0 : ℤ)) i := by
      ext S
      simp only [mem_filter, mem_univ, true_and, ne_eq, hxy, not_false_eq_true]
      rw [funext_iff]
      refine forall_congr' fun i => ?_
      simp only [project, mul_sub, Finset.sum_sub_distrib, sub_eq_zero]
    rw [hset]
    exact card_rows_le (fun r : Fin n → Bool => ∑ l, sgn (r l) * (x l - y l)) _ hf hne _

theorem commitThenProject : ResourceAccounting.Guarantees.Sketch.CommitThenProject := by
  intro k m n p A B C
  classical
  apply prob_uniform_le_inv_pow
  by_cases hC : C = A * B
  · have : (univ.filter fun S : Signs k m => C ≠ A * B ∧ signMatrix S * C = signMatrix S * A * B) = ∅ := by
      ext S
      simp [hC]
    rw [this, card_empty, mul_zero]
    exact Nat.zero_le _
  · obtain ⟨r, q, hrq⟩ : ∃ r q, (C - A * B) r q ≠ 0 := by
      by_contra h
      push Not at h
      exact hC (sub_eq_zero.1 (Matrix.ext h))
    obtain ⟨hf, hne⟩ := flip_moves (fun j => (C - A * B) j q) r hrq
    refine le_trans (Nat.mul_le_mul_left _ (card_le_card fun S hS => ?_))
      (card_rows_le (fun row : Fin m → Bool => ∑ j, sgn (row j) * (C - A * B) j q) _ hf hne fun _ => 0)
    simp only [mem_filter, mem_univ, true_and] at hS ⊢
    have h0 : signMatrix S * (C - A * B) = 0 := by
      rw [Matrix.mul_sub, hS.2, Matrix.mul_assoc, sub_self]
    intro i
    simpa [Matrix.mul_apply, signMatrix] using congrFun (congrFun h0 i) q

end ResourceAccounting.TagProofs
