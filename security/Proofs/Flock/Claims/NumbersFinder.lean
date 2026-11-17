module

public import Proofs.Flock.Claims.Numbers

/-! The by-position first-hit finder's terms, as numbers. -/

@[expose] public section

namespace Claims.Flock.Numbers

def firstHitCalls (T e Ns Ks Km c : ℚ) : ℚ :=
  T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c

def firstHitBound (T e Ns Ks Km c : ℚ) : ℚ :=
  4 * firstHitCalls T e Ns Ks Km c / 2 ^ 256

theorem firstHitBound_table1 :
    firstHitBound (2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 42) (2 ^ 42) (2 ^ 80) ≤ 1 / 2 ^ 100 := by
  norm_num [firstHitBound, firstHitCalls]

theorem firstHitBound_cast (T e Ns Ks Km c : ℚ) :
    ((firstHitBound T e Ns Ks Km c : ℚ) : ℝ) =
      4 * ((T : ℝ) * (1 + (e : ℝ) * (Ns : ℝ) + (Ks : ℝ) + (e : ℝ) * (Km : ℝ)) + (1 + (Ks : ℝ)) * (c : ℝ)) /
        2 ^ (256 : ℝ) := by
  rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  simp only [firstHitBound, firstHitCalls]
  push_cast
  all_goals ring

theorem real_le_firstHitBound {T e Ns Ks Km c : ℝ} {T' e' Ns' Ks' Km' c' : ℚ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e)
    (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ T') (he : e ≤ e') (hNs : Ns ≤ Ns')
    (hKs : Ks ≤ Ks') (hKm : Km ≤ Km') (hc : c ≤ c') :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤
      ((firstHitBound T' e' Ns' Ks' Km' c' : ℚ) : ℝ) := by
  have hT' : (0 : ℝ) ≤ T' := hT0.trans hT
  have he' : (0 : ℝ) ≤ e' := he0.trans he
  have hKs' : (0 : ℝ) ≤ Ks' := hKs0.trans hKs
  have h1 : e * Ns ≤ (e' : ℝ) * (Ns' : ℝ) := mul_le_mul he hNs hNs0 he'
  have h2 : e * Km ≤ (e' : ℝ) * (Km' : ℝ) := mul_le_mul he hKm hKm0 he'
  have h3 : 0 ≤ e * Ns := mul_nonneg he0 hNs0
  have h4 : 0 ≤ e * Km := mul_nonneg he0 hKm0
  have h5 : T * (1 + e * Ns + Ks + e * Km) ≤
      (T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) :=
    mul_le_mul hT (by linarith) (by linarith) hT'
  have h6 : (1 + Ks) * c ≤ (1 + (Ks' : ℝ)) * (c' : ℝ) := mul_le_mul (by linarith) hc hc0 (by linarith)
  have h7 : 4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) ≤
      4 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) + (1 + (Ks' : ℝ)) * (c' : ℝ)) := by
    linarith
  rw [firstHitBound_cast]
  exact div_le_div_of_nonneg_right h7 (by positivity)

theorem cast_two_pow_neg_100 : ((1 / 2 ^ 100 : ℚ) : ℝ) = 2 ^ (-(100 : ℝ)) := by
  rw [Real.rpow_neg (by norm_num), show (100 : ℝ) = ((100 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  push_cast
  all_goals ring

theorem real_le_two_rpow_neg_100 {T e Ns Ks Km c : ℝ} {T' e' Ns' Ks' Km' c' : ℚ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e)
    (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ T') (he : e ≤ e') (hNs : Ns ≤ Ns')
    (hKs : Ks ≤ Ks') (hKm : Km ≤ Km') (hc : c ≤ c') (h : firstHitBound T' e' Ns' Ks' Km' c' ≤ 1 / 2 ^ 100) :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  rw [← cast_two_pow_neg_100]
  exact (real_le_firstHitBound hT0 he0 hNs0 hKs0 hKm0 hc0 hT he hNs hKs hKm hc).trans (Rat.cast_le.2 h)

theorem real_table1_le {T e Ns Ks Km c : ℝ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks)
    (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ 2 ^ 80) (he : e ≤ 2592237061) (hNs : Ns ≤ 2 ^ 30) (hKs : Ks ≤ 2 ^ 42)
    (hKm : Km ≤ 2 ^ 42) (hc : c ≤ 2 ^ 80) :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  refine real_le_two_rpow_neg_100 (T' := 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 30) (Ks' := 2 ^ 42)
    (Km' := 2 ^ 42) (c' := 2 ^ 80) hT0 he0 hNs0 hKs0 hKm0 hc0 ?_ ?_ ?_ ?_ ?_ ?_ firstHitBound_table1
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hT
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hc

/-! The expression's values with every call charged as `513` compressions; no premise here counts compressions. -/

theorem firstHitBound_table1_513 :
    firstHitBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 33) (2 ^ 33) (513 * 2 ^ 80) ≤ 1 / 2 ^ 100 := by
  norm_num [firstHitBound, firstHitCalls]

theorem real_table1_513_le {T e Ns Ks Km c : ℝ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks)
    (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ 513 * 2 ^ 80) (he : e ≤ 2592237061) (hNs : Ns ≤ 2 ^ 30)
    (hKs : Ks ≤ 2 ^ 33) (hKm : Km ≤ 2 ^ 33) (hc : c ≤ 513 * 2 ^ 80) :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  refine real_le_two_rpow_neg_100 (T' := 513 * 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 30) (Ks' := 2 ^ 33)
    (Km' := 2 ^ 33) (c' := 513 * 2 ^ 80) hT0 he0 hNs0 hKs0 hKm0 hc0 ?_ ?_ ?_ ?_ ?_ ?_ firstHitBound_table1_513
  · exact hT.trans (le_of_eq (by norm_num))
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · exact hc.trans (le_of_eq (by norm_num))

set_option exponentiation.threshold 600 in
theorem sqrt_term_le_strict {k γ p q : ℝ} (hγ0 : 0 ≤ γ) (hp0 : 0 ≤ p) (hq0 : 0 ≤ q) (hk : k ≤ 4 + 4402 * 2 ^ 30)
    (hγ : γ ≤ 2 ^ (51 : ℝ)) (hq : q ≤ 2 * 2 ^ 80) (hp : p ≤ q ^ 2 / 2 ^ (513 : ℝ)) :
    Real.sqrt (k * γ * p) ≤ 2 ^ (-(128 : ℝ)) := by
  have h51 : (2 : ℝ) ^ (51 : ℝ) = 2 ^ 51 := by
    rw [show (51 : ℝ) = ((51 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h513 : (2 : ℝ) ^ (513 : ℝ) = 2 ^ 513 := by
    rw [show (513 : ℝ) = ((513 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h128 : (2 : ℝ) ^ (-(128 : ℝ)) = 1 / 2 ^ 128 := by
    rw [Real.rpow_neg (by norm_num), show (128 : ℝ) = ((128 : ℕ) : ℝ) by norm_num, Real.rpow_natCast, one_div]
  rw [h51] at hγ
  rw [h513] at hp
  have hq2 : q ^ 2 ≤ (2 * 2 ^ 80) ^ 2 := pow_le_pow_left₀ hq0 hq 2
  have hp' : p ≤ (2 * 2 ^ 80) ^ 2 / 2 ^ 513 := hp.trans (div_le_div_of_nonneg_right hq2 (by positivity))
  have hkγ : k * γ ≤ (4 + 4402 * 2 ^ 30) * 2 ^ 51 := mul_le_mul hk hγ hγ0 (by positivity)
  have h : k * γ * p ≤ (4 + 4402 * 2 ^ 30) * 2 ^ 51 * ((2 * 2 ^ 80) ^ 2 / 2 ^ 513) :=
    mul_le_mul hkγ hp' hp0 (by positivity)
  rw [h128, Real.sqrt_le_left (by positivity)]
  refine h.trans ?_
  norm_num

end Claims.Flock.Numbers
