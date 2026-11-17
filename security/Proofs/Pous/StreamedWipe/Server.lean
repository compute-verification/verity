module

public import Definitions.Pous.Guarantees.StreamedWipeServer
import Mathlib.Analysis.Complex.ExponentialBounds
import Proofs.Pous.StreamedWipe.Bound
import Proofs.Pous.StreamedWipe.Deployed
import Proofs.Pous.StreamedWipe.Family

/-! The streamed-label wipe per server, with the count of the missing labels. -/

@[expose] public section

namespace PousStreamedWipe

open Pous Pous.StreamedWipe
open scoped ENNReal

theorem pow_self_le_three_pow_mul_factorial (m : ℕ) : m ^ m ≤ 3 ^ m * m.factorial := by
  have h := Real.pow_div_factorial_le_exp (m : ℝ) (Nat.cast_nonneg m) m
  have he : Real.exp m ≤ 3 ^ m := by
    rw [← Real.exp_one_pow]
    exact pow_le_pow_left₀ (Real.exp_pos 1).le Real.exp_one_lt_three.le m
  have hf : (0 : ℝ) < m.factorial := by exact_mod_cast m.factorial_pos
  rw [div_le_iff₀ hf] at h
  exact_mod_cast h.trans (mul_le_mul_of_nonneg_right he hf.le)

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

theorem streamed_wipe_meets_reads_tight : Pous.Guarantees.StreamedWipeMeetsReadsTight :=
  fun _ _ hr hk _ hn _ _ _ _ prf P hq => streamed_wipe_le_of_reads_tight hr hk hn rfl hq prf P

theorem streamed_wipe_meets_server : Pous.Guarantees.StreamedWipeMeetsServer :=
  fun _ _ _ hMs hn _ _ hr hk _ _ _ prf P hq =>
    streamed_wipe_le_of_reads_tight hr hk hn (by rw [hMs, mul_comm]) hq prf P

theorem streamed_wipe_meets_server_5GB : Pous.Guarantees.StreamedWipeMeetsServer5GB :=
  fun Ms n R hMs hn r k hr hk hB q E κ prf P hq =>
    streamed_wipe_meets_server Ms n R hMs hn r k hr hk q E κ prf P
      (by unfold OperatingPoint at hB; omega)

theorem streamed_wipe_meets_server_H200 : Pous.Guarantees.StreamedWipeMeetsServerH200 := fun q E κ prf P hq =>
  streamed_wipe_meets_server_5GB h200.Ms h200.n h200.R (by decide +kernel) (by decide +kernel) h200.r h200.k
    (by decide +kernel) (by decide +kernel) (by unfold OperatingPoint; decide +kernel) q E κ prf P hq

end PousStreamedWipe
