import Proofs.Flock.Soundness.Discharge.ZkCoins.Basic

/-!
# Product laws on committed rounds

The hybrid of `Hybrid.lean` runs on `ω : I → Z` drawn from a product of per-round laws `μ j` (`W μ ω`, the weight of
`ω`). `resample` redraws one coordinate `k` from its own law inside the sum, and `reweight` changes `k`'s law under a
function that does not read `ω k`.
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Finset Function
open scoped ENNReal

section Weights

variable {I Z : Type} [Fintype I] [DecidableEq I] [Fintype Z]

/-- The weight of `ω` under the product of the laws `μ j`. -/
noncomputable def W (μ : I → Z → ℝ≥0∞) (ω : I → Z) : ℝ≥0∞ := ∏ j, μ j (ω j)

/-- The weight of `ω`'s coordinates other than `k`. -/
noncomputable def Wo (μ : I → Z → ℝ≥0∞) (k : I) (ω : I → Z) : ℝ≥0∞ := ∏ j, if j = k then 1 else μ j (ω j)

theorem sum_W (μ : I → Z → ℝ≥0∞) : ∑ ω, W μ ω = ∏ j, ∑ z, μ j z := by
  unfold W
  rw [Fintype.prod_sum]

theorem sum_W_one {μ : I → Z → ℝ≥0∞} (hμ : ∀ j, ∑ z, μ j z = 1) : ∑ ω, W μ ω = 1 := by
  rw [sum_W]
  exact Finset.prod_eq_one fun j _ => hμ j

theorem W_eq (μ : I → Z → ℝ≥0∞) (k : I) (ω : I → Z) : W μ ω = μ k (ω k) * Wo μ k ω := by
  unfold W Wo
  rw [← Finset.mul_prod_erase univ (fun j => μ j (ω j)) (mem_univ k),
    ← Finset.mul_prod_erase univ (fun j => if j = k then (1 : ℝ≥0∞) else μ j (ω j)) (mem_univ k)]
  rw [if_pos rfl, one_mul]
  refine congrArg _ (Finset.prod_congr rfl fun j hj => ?_)
  rw [if_neg (Finset.ne_of_mem_erase hj)]

theorem Wo_update (μ : I → Z → ℝ≥0∞) (k : I) (ω : I → Z) (z : Z) : Wo μ k (update ω k z) = Wo μ k ω := by
  unfold Wo
  refine Finset.prod_congr rfl fun j _ => ?_
  by_cases hj : j = k
  · rw [if_pos hj, if_pos hj]
  · rw [if_neg hj, if_neg hj, update_of_ne hj]

theorem Wo_congr {μ μ' : I → Z → ℝ≥0∞} {k : I} (h : ∀ j, j ≠ k → μ j = μ' j) (ω : I → Z) :
    Wo μ k ω = Wo μ' k ω := by
  unfold Wo
  refine Finset.prod_congr rfl fun j _ => ?_
  by_cases hj : j = k
  · rw [if_pos hj, if_pos hj]
  · rw [if_neg hj, if_neg hj, h j hj]

/-- Moving the fiber `ω k = z'` onto the fiber `ω k = z` by setting coordinate `k`. -/
theorem fiber_shift [DecidableEq Z] (k : I) (z z' : Z) (Φ : (I → Z) → ℝ≥0∞) :
    ∑ ω, (if ω k = z then Φ ω else 0) = ∑ ω, (if ω k = z' then Φ (update ω k z) else 0) := by
  have hinv : Involutive fun ω : I → Z => update ω k (Equiv.swap z z' (ω k)) := by
    intro ω
    show update (update ω k (Equiv.swap z z' (ω k))) k (Equiv.swap z z' (update ω k (Equiv.swap z z' (ω k)) k)) = ω
    rw [update_self, update_idem, Equiv.swap_apply_self, update_eq_self]
  symm
  refine Fintype.sum_equiv hinv.toPerm _ _ fun ω => ?_
  show (if ω k = z' then Φ (update ω k z) else 0) =
    if update ω k (Equiv.swap z z' (ω k)) k = z then Φ (update ω k (Equiv.swap z z' (ω k))) else 0
  rw [update_self]
  by_cases h : ω k = z'
  · have hs : Equiv.swap z z' (ω k) = z := by rw [h]; exact Equiv.swap_apply_right z z'
    rw [if_pos h, if_pos hs, hs]
  · have hs : Equiv.swap z z' (ω k) ≠ z := by
      intro h'
      apply h
      rw [← Equiv.swap_apply_self z z' (ω k), h']
      exact Equiv.swap_apply_left z z'
    rw [if_neg h, if_neg hs]

/-- Grouping a sum by coordinate `k`. -/
theorem sum_group [DecidableEq Z] (k : I) (h : Z → ℝ≥0∞) (Φ : (I → Z) → ℝ≥0∞) :
    ∑ ω, h (ω k) * Φ ω = ∑ z, h z * ∑ ω, (if ω k = z then Φ ω else 0) := by
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun ω _ => ?_
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq univ (ω k) (fun z => h z * Φ ω)]
  simp

theorem sum_fiber_update [DecidableEq Z] (μ : I → Z → ℝ≥0∞) (k : I) (z z' : Z) (F : (I → Z) → ℝ≥0∞) :
    ∑ ω, (if ω k = z' then Wo μ k ω * F (update ω k z) else 0) =
      ∑ ω, (if ω k = z then Wo μ k ω * F ω else 0) := by
  rw [fiber_shift k z z' (fun ω => Wo μ k ω * F ω)]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [Wo_update]

/-- **Redrawing coordinate `k` from its own law** leaves the sum unchanged. -/
theorem resample {μ : I → Z → ℝ≥0∞} {k : I} (hμ : ∑ z, μ k z = 1) (F : (I → Z) → ℝ≥0∞) :
    ∑ ω, W μ ω * F ω = ∑ ω, W μ ω * ∑ z, μ k z * F (update ω k z) := by
  classical
  set T : Z → ℝ≥0∞ := fun z => ∑ ω, (if ω k = z then Wo μ k ω * F ω else 0) with hT
  have lhs : ∑ ω, W μ ω * F ω = ∑ z, μ k z * T z := by
    simp only [W_eq μ k, mul_assoc]
    exact sum_group k (μ k) _
  have inner : ∀ z, ∑ ω, W μ ω * F (update ω k z) = T z := by
    intro z
    simp only [W_eq μ k, mul_assoc]
    rw [sum_group k (μ k)]
    simp only [sum_fiber_update μ k z]
    rw [← Finset.sum_mul, hμ, one_mul]
  rw [lhs]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun z _ => ?_
  rw [← inner z, Finset.mul_sum]
  refine Finset.sum_congr rfl fun ω _ => ?_
  ring

/-- **Changing coordinate `k`'s law** under a function that does not read `ω k`. -/
theorem reweight {μ μ' : I → Z → ℝ≥0∞} {k : I} (hoff : ∀ j, j ≠ k → μ j = μ' j) (hμ : ∑ z, μ k z = 1)
    (hμ' : ∑ z, μ' k z = 1) (G : (I → Z) → ℝ≥0∞) (hG : ∀ ω z, G (update ω k z) = G ω) :
    ∑ ω, W μ ω * G ω = ∑ ω, W μ' ω * G ω := by
  classical
  obtain ⟨z₀, -, -⟩ := Finset.exists_ne_zero_of_sum_ne_zero (s := univ) (f := μ k) (by rw [hμ]; exact one_ne_zero)
  have hT : ∀ (ν : I → Z → ℝ≥0∞) z, ∑ ω, (if ω k = z then Wo ν k ω * G ω else 0) =
      ∑ ω, (if ω k = z₀ then Wo ν k ω * G ω else 0) := by
    intro ν z
    rw [← sum_fiber_update ν k z z₀ G]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [hG]
  have key : ∀ ν : I → Z → ℝ≥0∞, ∑ z, ν k z = 1 →
      ∑ ω, W ν ω * G ω = ∑ ω, (if ω k = z₀ then Wo ν k ω * G ω else 0) := by
    intro ν hν
    simp only [W_eq ν k, mul_assoc]
    rw [sum_group k (ν k)]
    simp only [hT ν]
    rw [← Finset.sum_mul, hν, one_mul]
  rw [key μ hμ, key μ' hμ']
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [Wo_congr hoff]

end Weights

end FlockSoundness.Discharge.ZkCoins
