import Definitions.Pous.Guarantees.StreamedWipeFamily
import Proofs.Pous.StreamedWipe.Deployed

/-!
# The streamed-label wipe at 2152 reads, for every fill of a segment or more

`streamed_wipe` at `b = 512`, `M = n · 512`, `k = 2152` and `g = n − D` with `D = ⌈n/24⌉`, for every `n ≥ 2^19`:

- **sampling** (`sampling_le` at `t = 129`): `24 · D ≥ n`, so `2152 · D / n ≥ 2152/24 > 129 · 0.6931471808` and
  `((n − D)/n)^2152 ≤ 2^-129`;
- **compression** (`compression_le_two_pow`): `C(n, n − D + 1) ≤ 2^n` and `2152 ≤ 2^12`, so the term is at most
  `2^-129` once `q ≥ E + 12 + n + 512 · (D − 1) + 129`, which `⌈M/19⌉ + E` exceeds: `24 · (D − 1) < n`, and
  `512/24 + 1 < 512/19` bits a label leaves room for the `141`.

The count is the crude `2^n`, so no binomial of the deployed numbers is ever evaluated, and nothing bounds `n` above.
-/

namespace PousStreamedWipe

open Pous Pous.StreamedWipe
open scoped ENNReal

/-- **Compression with the count `2^N`.** With `M = N · b` and `k ≤ 2^12`, the compression term at `g = N − D` is at
most `2^-t` once `q ≥ E + 12 + N + (D − 1) · b + t`. -/
theorem compression_le_two_pow {N D k b q M E t : ℕ} (hD : 1 ≤ D) (hDN : D ≤ N) (hk : k ≤ 2 ^ 12)
    (hM : M = N * b) (hq : 12 + N + (D - 1) * b + E + t ≤ q) :
    (k : ℝ≥0∞) * (N.choose (N - D + 1) : ℝ≥0∞) * 2 ^ (M + E) / 2 ^ (q + (N - D + 1) * b) ≤ (2⁻¹ : ℝ≥0∞) ^ t := by
  have hsplit : M = (N - D + 1) * b + (D - 1) * b := by
    rw [hM, ← add_mul]
    congr 1
    omega
  have hnat : k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t ≤ 2 ^ (q + (N - D + 1) * b) := by
    calc k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t
        ≤ 2 ^ 12 * 2 ^ N * 2 ^ (M + E) * 2 ^ t := by gcongr; exact Nat.choose_le_two_pow N _
      _ = 2 ^ (12 + N + (M + E) + t) := by simp only [← pow_add]
      _ ≤ 2 ^ (q + (N - D + 1) * b) := Nat.pow_le_pow_right (by norm_num) (by rw [hsplit]; omega)
  rw [← ENNReal.inv_pow]
  apply ENNReal.div_le_of_le_mul
  rw [mul_comm ((2 : ℝ≥0∞) ^ t)⁻¹, ← div_eq_mul_inv,
    ENNReal.le_div_iff_mul_le (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top))]
  exact_mod_cast hnat

/-- **`StreamedWipeMeetsFamily`**: `streamed_wipe` at `g = n − ⌈n/24⌉`, with the error split `2^-129 + 2^-129`. -/
theorem streamed_wipe_meets_family : Pous.Guarantees.StreamedWipeMeetsFamily := by
  intro n hn q E κ prf P hq
  have hD1 : 1 ≤ (n + 23) / 24 := by omega
  have hDn : (n + 23) / 24 ≤ n := by omega
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hsamp : ((129 : ℕ) : ℝ) * 0.6931471808 ≤ ((2152 : ℕ) : ℝ) * (((n + 23) / 24 : ℕ) : ℝ) / n := by
    have h24 : (n : ℝ) ≤ 24 * (((n + 23) / 24 : ℕ) : ℝ) := by exact_mod_cast (show n ≤ 24 * ((n + 23) / 24) by omega)
    rw [le_div_iff₀ hn0]
    push_cast
    nlinarith [Nat.cast_nonneg (α := ℝ) ((n + 23) / 24)]
  refine (streamed_wipe n 2152 512 q (n * 512) E κ (n - (n + 23) / 24) prf P).trans (add_le_add ?_ le_rfl)
  unfold bound
  calc _ ≤ (2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 129 :=
        add_le_add (sampling_le hDn (by omega) hsamp)
          (compression_le_two_pow hD1 hDn (by norm_num) rfl (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ 128 := half_pow_succ_add 128

/-- **`StreamedWipeMeets96GiB`**: the family at `3072` segments. -/
theorem streamed_wipe_meets_96GiB : Pous.Guarantees.StreamedWipeMeets96GiB := fun q E κ prf P hq =>
  streamed_wipe_meets_family 1610612736 (by norm_num) q E κ prf P (by omega)

/-- **`StreamedWipeMeets141GB`**: the family at `141·10^9` bytes. -/
theorem streamed_wipe_meets_141GB : Pous.Guarantees.StreamedWipeMeets141GB := fun q E κ prf P hq =>
  streamed_wipe_meets_family 2203125000 (by norm_num) q E κ prf P (by omega)

end PousStreamedWipe
