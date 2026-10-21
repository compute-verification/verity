import Proofs.Accounting.Compose.SplitRateExp

/-!
# The closed rate: proof

`Accounting.Guarantees.Compose.SplitRateClosed`, from `SplitRateExp`. A larger rate keeps `SplitRateAt` at the same
`ν, ν₀` (`splitRateAt_mono`), and each closed branch is at least the sharp one (`sharp_le_closed`): with
`c = Kρw_max` and `q = p + (1 − p)e^{−c}`, `−ln q ≥ 1 − q = (1 − p)(1 − e^{−c})` and `(1 + c)e^{−c} ≤ 1`, so
`(1 + c)·(−ln q) ≥ (1 − p)·c`. At `w_max = 0` every work is `0` and `ν = ν₀ = 0` do.
-/

namespace Accounting.SecurityProofs.Compose.Rates

open Accounting.Compose

/-- A larger rate keeps `SplitRateAt` at the same `ν, ν₀`. -/
theorem splitRateAt_mono {K : ℕ} {ρ p p0 a b wmax Λ Λ' ν ν0 : ℝ} (hρ : 0 ≤ ρ) (hρw : ρ * wmax ≤ 1) (ha : 0 ≤ a)
    (hb : 0 ≤ b) (hΛ : 0 < Λ) (hle : Λ ≤ Λ') (h : SplitRateAt K ρ p p0 a b wmax Λ ν ν0) :
    SplitRateAt K ρ p p0 a b wmax Λ' ν ν0 := by
  obtain ⟨hν, hν0, h⟩ := h
  refine ⟨hν, hν0, fun WF W0 LF L0 h1 h2 h3 h4 h5 => le_trans ?_ (h WF W0 LF L0 h1 h2 h3 h4 h5)⟩
  have hD : 0 ≤ 1 - ρ * ((WF - LF) + (W0 - L0)) := by
    have := mul_le_mul_of_nonneg_left (show (WF - LF) + (W0 - L0) ≤ wmax by linarith) hρ
    linarith
  refine mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr ?_) (pow_nonneg hD K)
  exact div_le_div_of_nonneg_left
    (add_nonneg (mul_nonneg ha (by linarith)) (mul_nonneg hb (by linarith))) hΛ hle

/-- The sharp branch is positive. -/
theorem sharp_pos {K ρ p A w : ℝ} (hc : 0 < K * ρ * w) (hp0 : 0 ≤ p) (hp1 : p < 1) (hA : 0 < A * w) :
    0 < A * w / -Real.log (p + (1 - p) * Real.exp (-(K * ρ * w))) :=
  div_pos hA (neg_pos.mpr (Real.log_neg (mix_pos hp0 hp1) (mix_lt_one hp1 hc)))

/-- Each closed branch is at least the sharp one: `A·w/−ln(p + (1 − p)e^{−Kρw}) ≤ A(1 + Kρw)/(ρK(1 − p))`. -/
theorem sharp_le_closed {K ρ p A w : ℝ} (hK : 0 < K) (hρ : 0 < ρ) (hw : 0 < w) (hp0 : 0 ≤ p) (hp1 : p < 1)
    (hA : 0 < A) :
    A * w / -Real.log (p + (1 - p) * Real.exp (-(K * ρ * w))) ≤ A * (1 + K * ρ * w) / (ρ * K * (1 - p)) := by
  have hc0 : 0 < K * ρ * w := by positivity
  have hq0 := mix_pos (c := K * ρ * w) hp0 hp1
  have hq1 := mix_lt_one hp1 hc0
  set c := K * ρ * w with hc
  set q := p + (1 - p) * Real.exp (-c) with hq
  have hL : 0 < -Real.log q := neg_pos.mpr (Real.log_neg hq0 hq1)
  have hL1 : 1 - q ≤ -Real.log q := by linarith [Real.log_le_sub_one_of_pos hq0]
  have hexp : (1 + c) * Real.exp (-c) ≤ 1 := by
    have h1 := Real.add_one_le_exp c
    have he : Real.exp (-c) * Real.exp c = 1 := by rw [← Real.exp_add]; simp
    nlinarith [Real.exp_pos (-c)]
  have key : c * (1 - p) ≤ (1 + c) * -Real.log q := by
    have hid : (1 + c) * (1 - q) = (1 - p) * ((1 + c) - (1 + c) * Real.exp (-c)) := by rw [hq]; ring
    have h1 := mul_le_mul_of_nonneg_left hL1 (by linarith : (0 : ℝ) ≤ 1 + c)
    have h2 := mul_le_mul_of_nonneg_left (by linarith : c ≤ (1 + c) - (1 + c) * Real.exp (-c))
      (sub_pos.mpr hp1).le
    linarith
  rw [div_le_div_iff₀ hL (by have := sub_pos.mpr hp1; positivity)]
  calc A * w * (ρ * K * (1 - p)) = A * (c * (1 - p)) := by rw [hc]; ring
    _ ≤ A * ((1 + c) * -Real.log q) := mul_le_mul_of_nonneg_left key hA.le
    _ = A * (1 + c) * -Real.log q := by ring

theorem splitRateClosed : Accounting.Guarantees.Compose.SplitRateClosed := by
  intro K ρ p p0 a b wmax hK hρ hw hρw hp0 hp1 hq0 hq1 ha hb
  rcases hw.eq_or_lt with h0 | hw'
  · subst h0
    refine ⟨0, 0, le_rfl, le_rfl, fun WF W0 LF L0 h1 h2 h3 h4 h5 => ?_⟩
    obtain rfl : WF = 0 := by linarith
    obtain rfl : W0 = 0 := by linarith
    obtain rfl : LF = 0 := by linarith
    obtain rfl : L0 = 0 := by linarith
    simp
  · have hK' : (0 : ℝ) < K := by exact_mod_cast hK
    have hc : 0 < (K : ℝ) * ρ * wmax := by positivity
    obtain ⟨hν, hν0, h⟩ := splitRateExp K ρ p p0 a b wmax _ hK hρ hw' hρw hp0 hp1 hq0 hq1 ha hb rfl
    exact ⟨_, _, splitRateAt_mono hρ.le hρw ha.le hb.le
      (lt_max_of_lt_left (sharp_pos hc hp0 hp1 (mul_pos ha hw')))
      (max_le_max (sharp_le_closed hK' hρ hw' hp0 hp1 ha) (sharp_le_closed hK' hρ hw' hq0 hq1 hb))
      ⟨hν, hν0, h⟩⟩

end Accounting.SecurityProofs.Compose.Rates
