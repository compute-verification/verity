import Proofs.Accounting.Compose.SplitRateClosed

/-!
# The merged rate: proof

`Accounting.Guarantees.Compose.SplitRateMerged`, from `SplitRateClosed` at `p = p0 = q`: there the whole branch is the
larger (`a ≤ b`), and lower lucks only raise `SplitRateAt`'s right side (`splitRateAt_anti`).
-/

namespace Accounting.SecurityProofs.Compose.Rates

open Accounting.Compose

/-- Lower lucks keep `SplitRateAt` at the same `ν, ν₀`. -/
theorem splitRateAt_anti {K : ℕ} {ρ p p0 q q0 a b wmax Λ ν ν0 : ℝ} (hp : p ≤ q) (hp0 : p0 ≤ q0)
    (h : SplitRateAt K ρ q q0 a b wmax Λ ν ν0) : SplitRateAt K ρ p p0 a b wmax Λ ν ν0 := by
  obtain ⟨hν, hν0, h⟩ := h
  refine ⟨hν, hν0, fun WF W0 LF L0 h1 h2 h3 h4 h5 => (h WF W0 LF L0 h1 h2 h3 h4 h5).trans ?_⟩
  have hF := mul_nonneg hν (mul_nonneg (sub_nonneg.mpr hp) (by linarith : (0 : ℝ) ≤ WF))
  have h0 := mul_nonneg hν0 (mul_nonneg (sub_nonneg.mpr hp0) (by linarith : (0 : ℝ) ≤ W0))
  linarith

theorem splitRateMerged : Accounting.Guarantees.Compose.SplitRateMerged := by
  intro K ρ p p0 q a b wmax hK hρ hw hρw hq0 hpq hp0q hq1 ha hab
  have hb : 0 < b := lt_of_lt_of_le ha hab
  have hc : 0 ≤ 1 + (K : ℝ) * ρ * wmax := by
    have := mul_nonneg (mul_nonneg (Nat.cast_nonneg K) hρ.le) hw
    linarith
  have hd : 0 ≤ ρ * (K : ℝ) * (1 - q) := mul_nonneg (mul_nonneg hρ.le (Nat.cast_nonneg K)) (by linarith)
  have h := splitRateClosed K ρ q q a b wmax hK hρ hw hρw hq0 hq1 hq0 hq1 ha hb
  rw [max_eq_right (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hab hc) hd)] at h
  obtain ⟨ν, ν0, h⟩ := h
  exact ⟨ν, ν0, splitRateAt_anti hpq hp0q h⟩

end Accounting.SecurityProofs.Compose.Rates
