import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Definitions.Nci.Guarantees.CoverPartialSum

/-!
# The partial-sum bounds: the rate, one block at a time

`rate_block` is the per-block inequality behind `PartialSumCoverBound`: a block whose parameters cost `X ≤ k`, whose
high and low tokens cost `Ah + Al ≤ k`, and whose `nH` high and `nL` low products obey `block_counts`' bounds, has
`r(μ) · (nH + nL) ≤ X + μ (Ah + Al) + (1 − μ) Y / 3`, with `r = psRate b f a aOut (2k) μ` and `Y` its crossings. The
right side is the block's share of the two read budgets, `μ` of weights and activations and `1 − μ` of weights and
crossings at a third each, which add up to `2k` per block over a cover.

Each face of `r` is its own inequality. With `A = Ah + Al` and `K = 2k ≥ X + A`, the pools' AM–GM is
`(1 + √μ)² X A ≤ K (X + μ A)` (`rate_star`, `(X − √μ A)² ≥ 0`). `c₁` pays for the high products out of `Ah / A` of the
budget's `X + μ A`, `c₂` for the low ones out of the rest and the crossings (AM–GM on the two, `rate_min`), and the
fan-out face for both out of `μ (X + A)`, at `aOut` per activation and `K / (4 b a)` per high read (`rate_fan`).
-/

namespace Nci.Cover.SecurityProofs

/-- The pools' AM–GM: `(1 + r)² X A ≤ K (X + r² A)` when `X + A ≤ K`. -/
theorem rate_star {X A K r : ℝ} (hX : 0 ≤ X) (hA : 0 ≤ A) (hXA : X + A ≤ K) :
    (1 + r) ^ 2 * X * A ≤ K * (X + r ^ 2 * A) := by
  have h2 : 0 ≤ X + r ^ 2 * A := by positivity
  nlinarith [sq_nonneg (X - r * A), mul_le_mul_of_nonneg_right hXA h2]

/-- The fan-out face: `μ f / Λ` products per read, with `Λ = max {aOut, K / (4 b a)}`. -/
theorem rate_fan {b f a aOut K μ X Ah Al Y nH nL : ℝ} (hb : 0 < b) (ha : 0 < a) (haOut : 0 ≤ aOut) (hμ0 : 0 ≤ μ)
    (hμ1 : μ ≤ 1) (hX : 0 ≤ X) (hAh : 0 ≤ Ah) (hAl : 0 ≤ Al) (hY : 0 ≤ Y) (hXAh : X + Ah ≤ K)
    (hH : b * f * a * nH ≤ X * Ah) (hF : f * nL ≤ aOut * Al) :
    μ * f / max aOut (K / (4 * b * a)) * (nH + nL) ≤ X + μ * (Ah + Al) + (1 - μ) * Y / 3 := by
  have h1μ : 0 ≤ 1 - μ := sub_nonneg.2 hμ1
  have hR0 : 0 ≤ X + μ * (Ah + Al) + (1 - μ) * Y / 3 := by positivity
  have hΛ1 : aOut ≤ max aOut (K / (4 * b * a)) := le_max_left _ _
  have hΛ2 : K / (4 * b * a) ≤ max aOut (K / (4 * b * a)) := le_max_right _ _
  generalize max aOut (K / (4 * b * a)) = Λ at hΛ1 hΛ2 ⊢
  rcases (haOut.trans hΛ1).eq_or_lt with hΛ0 | hΛpos
  · rw [← hΛ0, div_zero, zero_mul]
    exact hR0
  have hba : 0 < 4 * b * a := by positivity
  have hXAh0 : 0 ≤ X + Ah := add_nonneg hX hAh
  have h4 : 4 * b * a * (f * nH) ≤ K * (X + Ah) := by
    have h5 : 4 * (X * Ah) ≤ (X + Ah) * (X + Ah) := by nlinarith [sq_nonneg (X - Ah)]
    have h6 : (X + Ah) * (X + Ah) ≤ K * (X + Ah) := mul_le_mul_of_nonneg_right hXAh hXAh0
    calc 4 * b * a * (f * nH) = 4 * (b * f * a * nH) := by ring
      _ ≤ 4 * (X * Ah) := by linarith
      _ ≤ K * (X + Ah) := h5.trans h6
  have hfH : f * nH ≤ Λ * (X + Ah) := by
    calc f * nH ≤ K * (X + Ah) / (4 * b * a) := by
          rw [le_div_iff₀ hba]
          linarith
      _ = K / (4 * b * a) * (X + Ah) := by ring
      _ ≤ Λ * (X + Ah) := mul_le_mul_of_nonneg_right hΛ2 hXAh0
  have hfL : f * nL ≤ Λ * Al := hF.trans (mul_le_mul_of_nonneg_right hΛ1 hAl)
  have hsum : f * (nH + nL) ≤ Λ * (X + (Ah + Al)) := by
    calc f * (nH + nL) = f * nH + f * nL := by ring
      _ ≤ Λ * (X + Ah) + Λ * Al := add_le_add hfH hfL
      _ = Λ * (X + (Ah + Al)) := by ring
  have hμsum : μ * (f * (nH + nL)) ≤ μ * (Λ * (X + (Ah + Al))) := mul_le_mul_of_nonneg_left hsum hμ0
  have hμX : μ * X ≤ X := mul_le_of_le_one_left hX hμ1
  have hW : 0 ≤ (1 - μ) * Y / 3 := by positivity
  calc μ * f / Λ * (nH + nL) = μ * (f * (nH + nL)) / Λ := by ring
    _ ≤ μ * (Λ * (X + (Ah + Al))) / Λ := div_le_div_of_nonneg_right hμsum hΛpos.le
    _ = Λ * (μ * (X + (Ah + Al))) / Λ := by ring
    _ = μ * (X + (Ah + Al)) := mul_div_cancel_left₀ _ hΛpos.ne'
    _ = μ * X + μ * (Ah + Al) := by ring
    _ ≤ X + μ * (Ah + Al) + (1 - μ) * Y / 3 := by linarith

/-- The Loomis–Whitney face: any `m ≤ c₁` with `m² ≤ c₂²` pays for both kinds of products. -/
theorem rate_min {b f a K μ r X Ah Al Y nH nL m : ℝ} (hμ1 : μ ≤ 1) (hr0 : 0 ≤ r) (hrr : r ^ 2 = μ) (hX : 0 ≤ X)
    (hAh : 0 ≤ Ah) (hAl : 0 ≤ Al) (hY : 0 ≤ Y) (hnH : 0 ≤ nH) (hXA : X + (Ah + Al) ≤ K)
    (hH : b * f * a * nH ≤ X * Ah) (hL : b * f * nL ^ 2 ≤ X * Al * Y) (hm1 : m ≤ (1 + r) ^ 2 * (b * f * a / K))
    (hm2 : m ^ 2 ≤ 4 * (1 + r) ^ 2 * (1 - μ) * (b * f / (3 * K))) :
    m * (nH + nL) ≤ X + μ * (Ah + Al) + (1 - μ) * Y / 3 := by
  have h1μ : 0 ≤ 1 - μ := sub_nonneg.2 hμ1
  have hμ0 : 0 ≤ μ := hrr ▸ sq_nonneg r
  have hA : 0 ≤ Ah + Al := add_nonneg hAh hAl
  have hK0 : 0 ≤ K := (add_nonneg hX hA).trans hXA
  have hZ0 : 0 ≤ (1 + r) ^ 2 * X / K := div_nonneg (by positivity) hK0
  have hkey : (Ah + Al) * ((1 + r) ^ 2 * X / K) ≤ X + μ * (Ah + Al) := by
    rcases hK0.eq_or_lt with hK | hK
    · rw [← hK, div_zero, mul_zero]
      positivity
    · have hs := rate_star (r := r) hX hA hXA
      rw [hrr] at hs
      rw [mul_div_assoc', div_le_iff₀ hK]
      calc (Ah + Al) * ((1 + r) ^ 2 * X) = (1 + r) ^ 2 * X * (Ah + Al) := by ring
        _ ≤ K * (X + μ * (Ah + Al)) := hs
        _ = (X + μ * (Ah + Al)) * K := by ring
  have hHpart : m * nH ≤ Ah * ((1 + r) ^ 2 * X / K) := by
    calc m * nH ≤ (1 + r) ^ 2 * (b * f * a / K) * nH := mul_le_mul_of_nonneg_right hm1 hnH
      _ = (1 + r) ^ 2 / K * (b * f * a * nH) := by ring
      _ ≤ (1 + r) ^ 2 / K * (X * Ah) := mul_le_mul_of_nonneg_left hH (div_nonneg (sq_nonneg _) hK0)
      _ = Ah * ((1 + r) ^ 2 * X / K) := by ring
  have hW : 0 ≤ (1 - μ) * Y / 3 := by positivity
  have hLsq : (m * nL) ^ 2 ≤ 4 * (Al * ((1 + r) ^ 2 * X / K)) * ((1 - μ) * Y / 3) := by
    have hc : 0 ≤ 4 * (1 + r) ^ 2 * (1 - μ) / (3 * K) := div_nonneg (by positivity) (by positivity)
    calc (m * nL) ^ 2 = m ^ 2 * nL ^ 2 := by ring
      _ ≤ 4 * (1 + r) ^ 2 * (1 - μ) * (b * f / (3 * K)) * nL ^ 2 :=
          mul_le_mul_of_nonneg_right hm2 (sq_nonneg _)
      _ = 4 * (1 + r) ^ 2 * (1 - μ) / (3 * K) * (b * f * nL ^ 2) := by ring
      _ ≤ 4 * (1 + r) ^ 2 * (1 - μ) / (3 * K) * (X * Al * Y) := mul_le_mul_of_nonneg_left hL hc
      _ = 4 * (Al * ((1 + r) ^ 2 * X / K)) * ((1 - μ) * Y / 3) := by ring
  have hLpart : m * nL ≤ Al * ((1 + r) ^ 2 * X / K) + (1 - μ) * Y / 3 :=
    le_of_pow_le_pow_left₀ two_ne_zero (add_nonneg (mul_nonneg hAl hZ0) hW)
      (hLsq.trans (four_mul_le_sq_add _ _))
  calc m * (nH + nL) = m * nH + m * nL := by ring
    _ ≤ Ah * ((1 + r) ^ 2 * X / K) + (Al * ((1 + r) ^ 2 * X / K) + (1 - μ) * Y / 3) :=
        add_le_add hHpart hLpart
    _ = (Ah + Al) * ((1 + r) ^ 2 * X / K) + (1 - μ) * Y / 3 := by ring
    _ ≤ X + μ * (Ah + Al) + (1 - μ) * Y / 3 := by linarith

/-- **The rate, per block.** -/
theorem rate_block {b f a aOut k μ X Ah Al Y nH nL : ℝ} (hb : 0 < b) (ha : 0 < a) (hf : 0 ≤ f) (haOut : 0 ≤ aOut)
    (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 1) (hX : 0 ≤ X) (hAh : 0 ≤ Ah) (hAl : 0 ≤ Al) (hY : 0 ≤ Y) (hnH : 0 ≤ nH)
    (hnL : 0 ≤ nL) (hXk : X ≤ k) (hAk : Ah + Al ≤ k) (hH : b * f * a * nH ≤ X * Ah)
    (hL : b * f * nL ^ 2 ≤ X * Al * Y) (hF : f * nL ≤ aOut * Al) :
    Guarantees.psRate b f a aOut (2 * k) μ * (nH + nL) ≤ X + μ * (Ah + Al) + (1 - μ) * Y / 3 := by
  have hk : 0 ≤ k := hX.trans hXk
  have h1μ : 0 ≤ 1 - μ := sub_nonneg.2 hμ1
  have hs : 0 ≤ nH + nL := add_nonneg hnH hnL
  have hXA : X + (Ah + Al) ≤ 2 * k := by linarith
  unfold Guarantees.psRate
  rw [max_mul_of_nonneg _ _ hs]
  refine max_le ?_ (rate_fan hb ha haOut hμ0 hμ1 hX hAh hAl hY (by linarith) hH hF)
  have hr0 : 0 ≤ √μ := Real.sqrt_nonneg μ
  have hrr : √μ ^ 2 = μ := Real.sq_sqrt hμ0
  have hs2 : √(1 - μ) ^ 2 = 1 - μ := Real.sq_sqrt h1μ
  have hs3 : √(b * f / (3 * (2 * k))) ^ 2 = b * f / (3 * (2 * k)) := Real.sq_sqrt (by positivity)
  have hs20 := Real.sqrt_nonneg (1 - μ)
  have hs30 := Real.sqrt_nonneg (b * f / (3 * (2 * k)))
  generalize √μ = r at hr0 hrr ⊢
  generalize √(1 - μ) = s2 at hs2 hs20 ⊢
  generalize √(b * f / (3 * (2 * k))) = s3 at hs3 hs30 ⊢
  have hc1 : 0 ≤ (1 + r) ^ 2 * (b * f * a / (2 * k)) := by positivity
  have hc2 : 0 ≤ 2 * (1 + r) * s2 * s3 := by positivity
  refine rate_min hμ1 hr0 hrr hX hAh hAl hY hnH hXA hH hL (min_le_left _ _) ?_
  calc _ ≤ (2 * (1 + r) * s2 * s3) ^ 2 := pow_le_pow_left₀ (le_min hc1 hc2) (min_le_right _ _) 2
    _ = 4 * (1 + r) ^ 2 * s2 ^ 2 * s3 ^ 2 := by ring
    _ = 4 * (1 + r) ^ 2 * (1 - μ) * (b * f / (3 * (2 * k))) := by rw [hs2, hs3]

end Nci.Cover.SecurityProofs
