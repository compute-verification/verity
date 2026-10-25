import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# The cover theorem: per-block bounds, summed over a division

A division into IUs gives one block per IU `u`, handed `k_u` bits and finishing `f_u` counted products. The rate form of
the NCI policy bounds each `k_u` by `α σ_u`, where `σ_u ≤ Γ` is the IU's size, so a per-block bound
`c f_u ≤ k_u² + c k_u` (`intensity`'s at `c = b`) adds up to `c Σ f_u ≤ (α² Γ + c α) Σ σ_u`: each block's
`k_u² ≤ α² σ_u² ≤ α² Γ σ_u` (`sum_area_le`).
-/

namespace Nci.Cover.SecurityProofs

/-- **One block's area bound at its own rate**: `c f ≤ k² + c k` with `0 ≤ k ≤ α σ` and `0 ≤ σ ≤ Γ` gives
`c f ≤ (α² Γ + c α) σ`. -/
theorem area_le_rate {c α Γ f k σ : ℝ} (hc : 0 ≤ c) (hα : 0 ≤ α) (hk0 : 0 ≤ k) (hσ0 : 0 ≤ σ)
    (hf : c * f ≤ k * k + c * k) (hk : k ≤ α * σ) (hσ : σ ≤ Γ) : c * f ≤ (α * α * Γ + c * α) * σ := by
  have h1 : k * k ≤ (α * σ) * (α * σ) := mul_self_le_mul_self hk0 hk
  have h2 : (α * σ) * (α * σ) ≤ α * α * Γ * σ := by
    have := mul_le_mul_of_nonneg_left hσ (mul_nonneg (mul_nonneg hα hα) hσ0)
    nlinarith
  have h3 : c * k ≤ c * (α * σ) := mul_le_mul_of_nonneg_left hk hc
  nlinarith

/-- **The area bound, summed over a division**: per-block bounds `c f_u ≤ k_u² + c k_u` with `0 ≤ k_u ≤ α σ_u` and
`0 ≤ σ_u ≤ Γ` add up to `c Σ f_u ≤ (α² Γ + c α) Σ σ_u`. -/
theorem sum_area_le {ι : Type*} (t : Finset ι) {c α Γ : ℝ} {f k σ : ι → ℝ} (hc : 0 ≤ c) (hα : 0 ≤ α)
    (hk0 : ∀ u ∈ t, 0 ≤ k u) (hσ0 : ∀ u ∈ t, 0 ≤ σ u) (hf : ∀ u ∈ t, c * f u ≤ k u * k u + c * k u)
    (hk : ∀ u ∈ t, k u ≤ α * σ u) (hσ : ∀ u ∈ t, σ u ≤ Γ) :
    c * ∑ u ∈ t, f u ≤ (α * α * Γ + c * α) * ∑ u ∈ t, σ u := by
  rw [Finset.mul_sum, Finset.mul_sum]
  exact Finset.sum_le_sum fun u hu =>
    area_le_rate hc hα (hk0 u hu) (hσ0 u hu) (hf u hu) (hk u hu) (hσ u hu)

end Nci.Cover.SecurityProofs
