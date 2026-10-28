import Definitions.Pous.Guarantees.StreamedWipeFamily
import Definitions.Pous.Guarantees.StreamedWipeReads
import Proofs.Pous.StreamedWipe.Deployed

/-!
# The streamed-label wipe at any number of reads, for every fill

`streamed_wipe` at `b = 512`, `M = n · 512 + R`, `k` reads and `g = n − D` with `D = ⌈n/r⌉`, for every `n ≥ 1` and
every `r ≥ 1` with `(r − 1)^k · 2^129 ≤ r^k` (`streamed_wipe_le_of_reads`):

- **sampling** (`sampling_le_ratio` at `t = 129`): `r · D ≥ n`, so `(n − D)/n ≤ (r − 1)/r` and
  `((n − D)/n)^k ≤ ((r − 1)/r)^k ≤ 2^-129`;
- **compression** (`compression_le_two_pow`): `C(n, n − D + 1) ≤ 2^n` and `k < 2^(⌊log₂ k⌋ + 1)`, so the term is at most
  `2^-129` once `q ≥ ⌊log₂ k⌋ + 1 + n + 512 · (D − 1) + R + E + 129`.

The count is the crude `2^n`, so no binomial of the deployed numbers is ever evaluated, and nothing bounds `n` above.
`StreamedWipeMeetsFamily` is the instance at `r = 24`, `k = 2152` and `R = 0`, where `⌈M/19⌉ + E` exceeds the threshold
(`24 · (D − 1) < n`, and `512/24 + 1 < 512/19` bits a label leaves room for the `141`); `StreamedWipeMeetsH200` is the
one at `r = 33`, `k = 2906` and an H200's whole memory, `R = 0`. The kernel decides each `(r − 1)^k · 2^129 ≤ r^k`.
-/

namespace PousStreamedWipe

open Pous Pous.StreamedWipe
open scoped ENNReal

/-- **Sampling at a ratio.** `((N − D)/N)^k ≤ 2^-t` once `N ≤ r · D` and `(r − 1)^k · 2^t ≤ r^k`, that is once
`(1 − 1/r)^k ≤ 2^-t`. -/
theorem sampling_le_ratio {N D k r t : ℕ} (hr : 0 < r) (hND : N ≤ r * D) (h : (r - 1) ^ k * 2 ^ t ≤ r ^ k) :
    (((N - D : ℕ) : ℝ≥0∞) / N) ^ k ≤ (2⁻¹ : ℝ≥0∞) ^ t := by
  have h1 : (N - D) * r ≤ (r - 1) * N := by
    rw [Nat.sub_mul, Nat.sub_mul, one_mul, mul_comm D r, mul_comm r N]
    omega
  have hnat : (N - D) ^ k * 2 ^ t ≤ N ^ k := by
    refine Nat.le_of_mul_le_mul_right ?_ (pow_pos hr k)
    calc (N - D) ^ k * 2 ^ t * r ^ k = ((N - D) * r) ^ k * 2 ^ t := by ring
      _ ≤ ((r - 1) * N) ^ k * 2 ^ t := by gcongr
      _ = (r - 1) ^ k * 2 ^ t * N ^ k := by ring
      _ ≤ r ^ k * N ^ k := by gcongr
      _ = N ^ k * r ^ k := by ring
  rw [div_eq_mul_inv, mul_pow, ← ENNReal.inv_pow, ← div_eq_mul_inv, ← ENNReal.inv_pow]
  apply ENNReal.div_le_of_le_mul
  rw [mul_comm ((2 : ℝ≥0∞) ^ t)⁻¹, ← div_eq_mul_inv,
    ENNReal.le_div_iff_mul_le (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top))]
  exact_mod_cast hnat

/-- **Compression with the count `2^N`.** With `M = N · b + R` and `k ≤ 2^j`, the compression term at `g = N − D` is
at most `2^-t` once `q ≥ j + N + (D − 1) · b + R + E + t`. -/
theorem compression_le_two_pow {N D k b q M E t j R : ℕ} (hD : 1 ≤ D) (hDN : D ≤ N) (hk : k ≤ 2 ^ j)
    (hM : M = N * b + R) (hq : j + N + (D - 1) * b + R + E + t ≤ q) :
    (k : ℝ≥0∞) * (N.choose (N - D + 1) : ℝ≥0∞) * 2 ^ (M + E) / 2 ^ (q + (N - D + 1) * b) ≤ (2⁻¹ : ℝ≥0∞) ^ t := by
  have hsplit : M = (N - D + 1) * b + (D - 1) * b + R := by
    rw [hM, ← add_mul, show N - D + 1 + (D - 1) = N by omega]
  have hnat : k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t ≤ 2 ^ (q + (N - D + 1) * b) := by
    calc k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t
        ≤ 2 ^ j * 2 ^ N * 2 ^ (M + E) * 2 ^ t := by gcongr; exact Nat.choose_le_two_pow N _
      _ = 2 ^ (j + N + (M + E) + t) := by simp only [← pow_add]
      _ ≤ 2 ^ (q + (N - D + 1) * b) := Nat.pow_le_pow_right (by norm_num) (by rw [hsplit]; omega)
  rw [← ENNReal.inv_pow]
  apply ENNReal.div_le_of_le_mul
  rw [mul_comm ((2 : ℝ≥0∞) ^ t)⁻¹, ← div_eq_mul_inv,
    ENNReal.le_div_iff_mul_le (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top))]
  exact_mod_cast hnat

/-- `streamed_wipe` at `g = n − ⌈n/r⌉` on a server of `M = n · 512 + R` bits, with the error split
`2^-129 + 2^-129`. -/
theorem streamed_wipe_le_of_reads {r k n R q M E κ : ℕ} (hr : 0 < r) (hk : (r - 1) ^ k * 2 ^ 129 ≤ r ^ k)
    (hn : 0 < n) (hM : M = n * 512 + R) (hq : 512 * ((n + r - 1) / r - 1) + n + Nat.log2 k + 130 + R + E ≤ q)
    (prf : Bits κ → Fin n → Bits 512) (P : Prover n k 512 q M E) :
    winReal prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisher P) := by
  have hmod := Nat.div_add_mod (n + r - 1) r
  have hlt := Nat.mod_lt (n + r - 1) hr
  have hD1 : 1 ≤ (n + r - 1) / r := (Nat.le_div_iff_mul_le hr).2 (by omega)
  have hrD : n ≤ r * ((n + r - 1) / r) := by omega
  have hDn : (n + r - 1) / r ≤ n := by
    by_contra h
    have h1 := Nat.mul_le_mul_left r (show n + 1 ≤ (n + r - 1) / r by omega)
    have h2 := Nat.le_mul_of_pos_left n hr
    rw [mul_add_one] at h1
    omega
  refine (streamed_wipe n k 512 q M E κ (n - (n + r - 1) / r) prf P).trans (add_le_add ?_ le_rfl)
  unfold bound
  calc _ ≤ (2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 129 :=
        add_le_add (sampling_le_ratio hr hrD hk)
          (compression_le_two_pow hD1 hDn Nat.lt_log2_self.le hM (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ 128 := half_pow_succ_add 128

/-- **`StreamedWipeMeetsReads`**: `streamed_wipe_le_of_reads` at `M = n · 512 + R`. -/
theorem streamed_wipe_meets_reads : Pous.Guarantees.StreamedWipeMeetsReads :=
  fun _ _ hr hk _ hn _ _ _ _ prf P hq => streamed_wipe_le_of_reads hr hk hn rfl hq prf P

/-- **`StreamedWipeMeetsFamily`**: `streamed_wipe_le_of_reads` at `r = 24`, `k = 2152` and `R = 0`. -/
theorem streamed_wipe_meets_family : Pous.Guarantees.StreamedWipeMeetsFamily := by
  intro n hn q E κ prf P hq
  have hk : Nat.log2 2152 < 12 := (Nat.log2_lt (by norm_num)).2 (by norm_num)
  exact streamed_wipe_le_of_reads (r := 24) (R := 0) (by norm_num) (by decide +kernel) (by omega)
    (Nat.add_zero _).symm (by omega) prf P

/-- **`StreamedWipeMeets96GiB`**: the family at `3072` segments. -/
theorem streamed_wipe_meets_96GiB : Pous.Guarantees.StreamedWipeMeets96GiB := fun q E κ prf P hq =>
  streamed_wipe_meets_family 1610612736 (by norm_num) q E κ prf P (by omega)

/-- **`StreamedWipeMeets141GB`**: the family at `141·10^9` bytes. -/
theorem streamed_wipe_meets_141GB : Pous.Guarantees.StreamedWipeMeets141GB := fun q E κ prf P hq =>
  streamed_wipe_meets_family 2203125000 (by norm_num) q E κ prf P (by omega)

/-- **`StreamedWipeMeetsH200`**: `StreamedWipeMeetsReads` at `r = 33`, `k = 2906`, an H200's 141 GiB (`151,397,597,184`
bytes) filled whole and `R = 0`. -/
theorem streamed_wipe_meets_H200 : Pous.Guarantees.StreamedWipeMeetsH200 := by
  intro q E κ prf P hq
  have hk : Nat.log2 2906 < 12 := (Nat.log2_lt (by norm_num)).2 (by norm_num)
  exact streamed_wipe_meets_reads 33 2906 (by norm_num) (by decide +kernel) 2365587456 (by norm_num) 0
    q E κ prf P (by omega)

end PousStreamedWipe
