import Definitions.Pous.Guarantees.StreamedWipeServer
import Proofs.Pous.StreamedWipe.Family

/-!
# The streamed-label wipe per server, with the count of the missing labels

`streamed_wipe` at `b = 512`, `M = n · 512 + R`, `k` reads and `g = n − D` with `D = ⌈n/r⌉`, as
`streamed_wipe_le_of_reads`, with the compression term's count `C(n, D − 1)` bounded by
`2^(m·(⌊log₂⌊3n/m⌋⌋ + 1))`, `m = D − 1` (`choose_le_two_pow_countBits`), instead of `2^n`:

- `C(n, m) · m! ≤ n^m` (`Nat.descFactorial_le_pow`);
- `m^m ≤ 3^m · m!` (`pow_self_le_three_pow_mul_factorial`, from `m^m/m! ≤ e^m` and `e < 3`);
- `3n < m · 2^(⌊log₂⌊3n/m⌋⌋ + 1)`, so `(3n)^m ≤ m^m · 2^(m·(⌊log₂⌊3n/m⌋⌋ + 1))`.

`StreamedWipeMeetsServer` is it at `M = M_s`, `StreamedWipeMeetsServer5GB` at the 5 GB point, and the four servers'
instances are the 5 GB form at their `n`, `R`, `r` and `k` (the H200's the measured deployment `StreamedWipe.h200`),
each `(r − 1)^k · 2^129 ≤ r^k` and each `B(n, r, k, R) ≤ 4·10^10` decided by the kernel. Each guarantee is its claim at
`StreamedWipe.Accepts`, whose `winRealBy` and `wipeDistinguisherBy` are `winReal` and `wipeDistinguisher` by
definition. The claims' witnesses are `Security/Proofs/Pous/StreamedWipe/Witnesses.lean`.
-/

namespace PousStreamedWipe

open Pous Pous.StreamedWipe
open scoped ENNReal

/-- `m^m ≤ 3^m · m!`, from `m^m / m! ≤ e^m` and `e < 3`. -/
theorem pow_self_le_three_pow_mul_factorial (m : ℕ) : m ^ m ≤ 3 ^ m * m.factorial := by
  have h := Real.pow_div_factorial_le_exp (m : ℝ) (Nat.cast_nonneg m) m
  have he : Real.exp m ≤ 3 ^ m := by
    rw [← Real.exp_one_pow]
    exact pow_le_pow_left₀ (Real.exp_pos 1).le Real.exp_one_lt_three.le m
  have hf : (0 : ℝ) < m.factorial := by exact_mod_cast m.factorial_pos
  rw [div_le_iff₀ hf] at h
  exact_mod_cast h.trans (mul_le_mul_of_nonneg_right he hf.le)

/-- **The count of the missing labels**: `C(n, m) ≤ 2^(m·(⌊log₂⌊3n/m⌋⌋ + 1))`. -/
theorem choose_le_two_pow_countBits (n m : ℕ) : n.choose m ≤ 2 ^ countBits n m := by
  unfold countBits
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · simp
  set L := Nat.log2 (3 * n / m)
  have h3 : 3 * n < m * 2 ^ (L + 1) :=
    (Nat.lt_mul_div_succ (3 * n) hm).trans_le (Nat.mul_le_mul_left m Nat.lt_log2_self)
  have hC : m.factorial * n.choose m ≤ n ^ m :=
    (Nat.descFactorial_eq_factorial_mul_choose n m) ▸ Nat.descFactorial_le_pow n m
  refine Nat.le_of_mul_le_mul_left ?_ (Nat.mul_pos (pow_pos (by norm_num : 0 < 3) m) m.factorial_pos)
  calc 3 ^ m * m.factorial * n.choose m ≤ 3 ^ m * n ^ m := by
        rw [mul_assoc]; exact Nat.mul_le_mul_left _ hC
    _ = (3 * n) ^ m := (mul_pow 3 n m).symm
    _ ≤ (m * 2 ^ (L + 1)) ^ m := Nat.pow_le_pow_left h3.le m
    _ = m ^ m * 2 ^ (m * (L + 1)) := by rw [mul_pow, ← pow_mul, mul_comm (L + 1)]
    _ ≤ 3 ^ m * m.factorial * 2 ^ (m * (L + 1)) :=
        Nat.mul_le_mul_right _ (pow_self_le_three_pow_mul_factorial m)

/-- **Compression with a count.** With `M = N · b + R`, `k ≤ 2^j` and `C(N, D − 1) ≤ 2^c`, the compression term at
`g = N − D` is at most `2^-t` once `q ≥ j + c + (D − 1) · b + R + E + t`. -/
theorem compression_le_count {N D k b q M E t j R c : ℕ} (hD : 1 ≤ D) (hDN : D ≤ N) (hk : k ≤ 2 ^ j)
    (hC : N.choose (D - 1) ≤ 2 ^ c) (hM : M = N * b + R) (hq : j + c + (D - 1) * b + R + E + t ≤ q) :
    (k : ℝ≥0∞) * (N.choose (N - D + 1) : ℝ≥0∞) * 2 ^ (M + E) / 2 ^ (q + (N - D + 1) * b) ≤ (2⁻¹ : ℝ≥0∞) ^ t := by
  have hsplit : M = (N - D + 1) * b + (D - 1) * b + R := by
    rw [hM, ← add_mul, show N - D + 1 + (D - 1) = N by omega]
  have hsymm : N.choose (N - D + 1) = N.choose (D - 1) := by
    rw [show N - D + 1 = N - (D - 1) by omega, Nat.choose_symm (by omega)]
  have hnat : k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t ≤ 2 ^ (q + (N - D + 1) * b) := by
    calc k * N.choose (N - D + 1) * 2 ^ (M + E) * 2 ^ t
        ≤ 2 ^ j * 2 ^ c * 2 ^ (M + E) * 2 ^ t := by rw [hsymm]; gcongr
      _ = 2 ^ (j + c + (M + E) + t) := by simp only [← pow_add]
      _ ≤ 2 ^ (q + (N - D + 1) * b) := Nat.pow_le_pow_right (by norm_num) (by rw [hsplit]; omega)
  rw [← ENNReal.inv_pow]
  apply ENNReal.div_le_of_le_mul
  rw [mul_comm ((2 : ℝ≥0∞) ^ t)⁻¹, ← div_eq_mul_inv,
    ENNReal.le_div_iff_mul_le (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top))]
  exact_mod_cast hnat

/-- `streamed_wipe` at `g = n − ⌈n/r⌉` on a server of `M = n · 512 + R` bits, with the count of the missing labels
and the error split `2^-129 + 2^-129`. -/
theorem streamed_wipe_le_of_reads_tight {r k n R q M E κ : ℕ} (hr : 0 < r) (hk : (r - 1) ^ k * 2 ^ 129 ≤ r ^ k)
    (hn : 0 < n) (hM : M = n * 512 + R) (hq : fillShare n r k + R + E ≤ q)
    (prf : Bits κ → Fin n → Bits 512) (P : Prover n k 512 q M E) :
    winReal prf P ≤ (2⁻¹ : ℝ≥0∞) ^ 128 + prfAdvantage prf (wipeDistinguisher P) := by
  unfold fillShare missing at hq
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
          (compression_le_count hD1 hDn Nat.lt_log2_self.le (choose_le_two_pow_countBits n _) hM (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ 128 := half_pow_succ_add 128

/-- **`StreamedWipeMeetsReadsTight`**: `streamed_wipe_le_of_reads_tight` at `M = n · 512 + R`. -/
theorem streamed_wipe_meets_reads_tight : Pous.Guarantees.StreamedWipeMeetsReadsTight :=
  fun _ _ hr hk _ hn _ _ _ _ prf P hq => streamed_wipe_le_of_reads_tight hr hk hn rfl hq prf P

/-- **`StreamedWipeMeetsServer`**: `streamed_wipe_le_of_reads_tight` at `M = M_s = 512 · n + R`. -/
theorem streamed_wipe_meets_server : Pous.Guarantees.StreamedWipeMeetsServer :=
  fun _ _ _ hMs hn _ _ hr hk _ _ _ prf P hq =>
    streamed_wipe_le_of_reads_tight hr hk hn (by rw [hMs, mul_comm]) hq prf P

/-- **`StreamedWipeMeetsServer5GB`**: `StreamedWipeMeetsServer` with `B(n, r, k, R) ≤ 4·10^10`. -/
theorem streamed_wipe_meets_server_5GB : Pous.Guarantees.StreamedWipeMeetsServer5GB :=
  fun Ms n R hMs hn r k hr hk hB q E κ prf P hq =>
    streamed_wipe_meets_server Ms n R hMs hn r k hr hk q E κ prf P
      (by unfold OperatingPoint at hB; omega)

/-- **`StreamedWipeMeetsServerH200`**: the 5 GB form at the deployed H200 as measured, `StreamedWipe.h200`:
`n = 2,319,974,400`, `M_s = 1,206,038,560,768`, `r = 62` and `k = 5500`. -/
theorem streamed_wipe_meets_server_H200 : Pous.Guarantees.StreamedWipeMeetsServerH200 := fun q E κ prf P hq =>
  streamed_wipe_meets_server_5GB h200.Ms h200.n h200.R (by decide +kernel) (by decide +kernel) h200.r h200.k
    (by decide +kernel) (by decide +kernel) (by unfold OperatingPoint; decide +kernel) q E κ prf P hq

/-- **`StreamedWipeMeetsServer8H200HBM`**: the 5 GB form at `n = 17,625,000,000`, `R = 31,200,000,000`, `r = 1050`
and `k = 93,843`. -/
theorem streamed_wipe_meets_server_8H200_HBM : Pous.Guarantees.StreamedWipeMeetsServer8H200HBM :=
  fun q E κ prf P hq =>
    streamed_wipe_meets_server_5GB _ 17625000000 31200000000 (by norm_num) (by norm_num) 1050 93843 (by norm_num)
      (by decide +kernel) (by unfold OperatingPoint; decide +kernel) q E κ prf P hq

/-- **`StreamedWipeMeetsServer8H200HBMRAM`**: the 5 GB form at `n = 48,875,000,000`, `R = 31,200,000,000`,
`r = 2922` and `k = 261,229`. -/
theorem streamed_wipe_meets_server_8H200_HBMRAM : Pous.Guarantees.StreamedWipeMeetsServer8H200HBMRAM :=
  fun q E κ prf P hq =>
    streamed_wipe_meets_server_5GB _ 48875000000 31200000000 (by norm_num) (by norm_num) 2922 261229 (by norm_num)
      (by decide +kernel) (by unfold OperatingPoint; decide +kernel) q E κ prf P hq

/-- **`StreamedWipeMeetsServer8H200`**: the 5 GB form at `n = 236,375,000,000`, `R = 31,200,000,000`, `r = 14,183`
and `k = 1,268,143`. -/
theorem streamed_wipe_meets_server_8H200 : Pous.Guarantees.StreamedWipeMeetsServer8H200 :=
  fun q E κ prf P hq =>
    streamed_wipe_meets_server_5GB _ 236375000000 31200000000 (by norm_num) (by norm_num) 14183 1268143
      (by norm_num) (by decide +kernel) (by unfold OperatingPoint; decide +kernel) q E κ prf P hq

end PousStreamedWipe
