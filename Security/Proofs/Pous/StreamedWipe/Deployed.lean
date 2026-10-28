import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Nat.Choose.Bounds
import Proofs.Pous.StreamedWipe.Bound

/-!
# The streamed-label wipe at 2.6 TB and 74,000 challenges

`streamed_wipe` at `N = 40,625,000,000`, `b = 512`, `M = N · 512`, `k = 74,000` and `g = N − 15,610,000`:

- **sampling** (`sampling_le`): `(1 − D/N)^k ≤ e^(−k·D/N) ≤ 2^-41` once `k·D/N ≥ 41 · 0.6931471808 > 41 · ln 2`; here
  `k·D/N = 28.434`;
- **compression** (`compression_le`): `C(N, N−D+1) = C(N, D−1) ≤ N^(D−1) ≤ 2^(36·(D−1))` and `k ≤ 2^17`, so the term is
  at most `2^(17 + 36·(D−1) + (D−1)·b + E − q) ≤ 2^-41` once `q ≥ E + 17 + 548·(D−1) + 41 = E + 8,554,279,510`.

Both lemmas are stated for any `N`, `D` and `k`, so that no power of the deployed numbers is ever evaluated.
-/

namespace PousStreamedWipe

open Pous Pous.StreamedWipe
open scoped ENNReal

/-- **Sampling.** `((N − D)/N)^k ≤ 2^-t` once `k · D / N ≥ t · 0.6931471808`. -/
theorem sampling_le {N D k t : ℕ} (hDN : D ≤ N) (hN : 0 < N) (h : (t : ℝ) * 0.6931471808 ≤ k * D / N) :
    (((N - D : ℕ) : ℝ≥0∞) / N) ^ k ≤ (2⁻¹ : ℝ≥0∞) ^ t := by
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  set x : ℝ := ((N - D : ℕ) : ℝ) / N with hx
  have hx0 : 0 ≤ x := by positivity
  have hx1 : x = -((D : ℝ) / N) + 1 := by
    rw [hx, Nat.cast_sub hDN]
    field_simp
    ring
  have hexp : x ^ k ≤ Real.exp (-(t * Real.log 2)) := by
    calc x ^ k ≤ Real.exp (-((D : ℝ) / N)) ^ k := by
          gcongr
          rw [hx1]
          exact Real.add_one_le_exp _
      _ = Real.exp (k * -((D : ℝ) / N)) := (Real.exp_nat_mul _ _).symm
      _ ≤ Real.exp (-(t * Real.log 2)) := by
          rw [Real.exp_le_exp]
          have hl := mul_le_mul_of_nonneg_left Real.log_two_lt_d9.le (Nat.cast_nonneg (α := ℝ) t)
          have : (k : ℝ) * ((D : ℝ) / N) = k * D / N := by ring
          nlinarith
  have htwo : Real.exp (-(t * Real.log 2)) = (2⁻¹ : ℝ) ^ t := by
    rw [Real.exp_neg, Real.exp_nat_mul, Real.exp_log two_pos, inv_pow]
  have hreal : x ^ k ≤ (2⁻¹ : ℝ) ^ t := htwo ▸ hexp
  have hl : (((N - D : ℕ) : ℝ≥0∞) / N) ^ k = ENNReal.ofReal (x ^ k) := by
    rw [ENNReal.ofReal_pow hx0, hx, ENNReal.ofReal_div_of_pos hNr, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
  have hr : (2⁻¹ : ℝ≥0∞) ^ t = ENNReal.ofReal ((2⁻¹ : ℝ) ^ t) := by
    rw [ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_inv_of_pos two_pos, ENNReal.ofReal_ofNat]
  rw [hl, hr]
  exact ENNReal.ofReal_le_ofReal hreal

/-- **Compression.** With `M = N · b`, `N ≤ 2^36` and `k ≤ 2^17`, the compression term at `g = N − D` is at most
`2^-41` once `q ≥ E + 17 + (36 + b) · (D − 1) + 41`. -/
theorem compression_le {N D k b q M E : ℕ} (hD : 1 ≤ D) (hDN : D ≤ N) (hN : N ≤ 2 ^ 36) (hk : k ≤ 2 ^ 17)
    (hM : M = N * b) (hq : 17 + 36 * (D - 1) + (D - 1) * b + E + 41 ≤ q) :
    (k : ℝ≥0∞) * (N.choose (N - D + 1) : ℝ≥0∞) * 2 ^ (M + E) / 2 ^ (q + (N - D + 1) * b) ≤ (2⁻¹ : ℝ≥0∞) ^ 41 := by
  have hC : N.choose (N - D + 1) ≤ 2 ^ (36 * (D - 1)) := by
    rw [show N - D + 1 = N - (D - 1) by omega, Nat.choose_symm (by omega), pow_mul]
    exact (Nat.choose_le_pow N (D - 1)).trans (Nat.pow_le_pow_left hN _)
  have hsplit : M = (N - D + 1) * b + (D - 1) * b := by
    rw [hM, ← add_mul]
    congr 1
    omega
  have hnat : k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ 41 ≤ 2 ^ (q + (N - D + 1) * b) := by
    calc k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ 41
        ≤ 2 ^ 17 * 2 ^ (36 * (D - 1)) * 2 ^ (M + E) * 2 ^ 41 := by gcongr
      _ = 2 ^ (17 + 36 * (D - 1) + (M + E) + 41) := by simp only [← pow_add]
      _ ≤ 2 ^ (q + (N - D + 1) * b) := Nat.pow_le_pow_right (by norm_num) (by rw [hsplit]; omega)
  rw [← ENNReal.inv_pow]
  apply ENNReal.div_le_of_le_mul
  rw [mul_comm ((2 : ℝ≥0∞) ^ 41)⁻¹, ← div_eq_mul_inv,
    ENNReal.le_div_iff_mul_le (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top))]
  exact_mod_cast hnat

theorem half_pow_succ_add (t : ℕ) : (2⁻¹ : ℝ≥0∞) ^ (t + 1) + (2⁻¹ : ℝ≥0∞) ^ (t + 1) = (2⁻¹ : ℝ≥0∞) ^ t := by
  rw [← two_mul, pow_succ, mul_comm, mul_assoc, ENNReal.inv_mul_cancel two_ne_zero ENNReal.ofNat_ne_top, mul_one]

/-- **`StreamedWipeMeets2600GB`**: `streamed_wipe` at `g = N − 15,610,000`, with the error split `2^-41 + 2^-41`. -/
theorem streamed_wipe_meets_2600GB : Pous.Guarantees.StreamedWipeMeets2600GB := by
  intro q E κ prf P hq
  refine (streamed_wipe 40625000000 74000 512 q (40625000000 * 512) E κ (40625000000 - 15610000) prf P).trans
    (add_le_add ?_ le_rfl)
  unfold bound
  calc _ ≤ (2⁻¹ : ℝ≥0∞) ^ 41 + (2⁻¹ : ℝ≥0∞) ^ 41 :=
        add_le_add (sampling_le (by norm_num) (by norm_num) (by norm_num))
          (compression_le (by norm_num) (by norm_num) (by norm_num) (by norm_num) rfl (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ 40 := half_pow_succ_add 40

end PousStreamedWipe
