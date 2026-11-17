module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.Linarith.Frontend
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific

/-! The cover theorem: per-block bounds, summed over a division. -/

@[expose] public section

namespace Slowdown.Cover.SecurityProofs

theorem area_le_rate {c α Γ f k σ : ℝ} (hc : 0 ≤ c) (hα : 0 ≤ α) (hk0 : 0 ≤ k) (hσ0 : 0 ≤ σ)
    (hf : c * f ≤ k * k + c * k) (hk : k ≤ α * σ) (hσ : σ ≤ Γ) : c * f ≤ (α * α * Γ + c * α) * σ := by
  have h1 : k * k ≤ (α * σ) * (α * σ) := mul_self_le_mul_self hk0 hk
  have h2 : (α * σ) * (α * σ) ≤ α * α * Γ * σ := by
    have := mul_le_mul_of_nonneg_left hσ (mul_nonneg (mul_nonneg hα hα) hσ0)
    nlinarith
  have h3 : c * k ≤ c * (α * σ) := mul_le_mul_of_nonneg_left hk hc
  nlinarith

theorem sum_area_le {ι : Type*} (t : Finset ι) {c α Γ : ℝ} {f k σ : ι → ℝ} (hc : 0 ≤ c) (hα : 0 ≤ α)
    (hk0 : ∀ u ∈ t, 0 ≤ k u) (hσ0 : ∀ u ∈ t, 0 ≤ σ u) (hf : ∀ u ∈ t, c * f u ≤ k u * k u + c * k u)
    (hk : ∀ u ∈ t, k u ≤ α * σ u) (hσ : ∀ u ∈ t, σ u ≤ Γ) :
    c * ∑ u ∈ t, f u ≤ (α * α * Γ + c * α) * ∑ u ∈ t, σ u := by
  rw [Finset.mul_sum, Finset.mul_sum]
  exact Finset.sum_le_sum fun u hu =>
    area_le_rate hc hα (hk0 u hu) (hσ0 u hu) (hf u hu) (hk u hu) (hσ u hu)

end Slowdown.Cover.SecurityProofs
