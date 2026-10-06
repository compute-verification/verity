import Proofs.Core.TC.Proofs.Fold

/-!
# Normalisation: the relation's columns against `group_sum`'s shifts

Pure arithmetic for the Hopper adder (`W = 26`, `B = 32`, `K = 36`). `norm_nat` is `group_sum`'s normalisation to 26 bits,
subnormal shift and rescale, written as the relation computes it: `normB = |T| * 2^(32 - L)`, `magW = normB / 2^6`,
`mag24 = magW / 2^min(e_fin - e_out + 2, 36)`. The `*_of_*` lemmas read each column off the predicates that constrain it.
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

theorem bitLength_spec (n : Nat) (hn : n ≠ 0) : 2 ^ (bitLength n - 1) ≤ n ∧ n < 2 ^ bitLength n ∧ 1 ≤ bitLength n := by
  unfold bitLength
  simp only [hn, ite_false, Nat.add_sub_cancel]
  exact ⟨Nat.log2_self_le hn, Nat.lt_log2_self, by omega⟩

theorem bitLength_le (n k : Nat) (hn : n < 2 ^ k) : bitLength n ≤ k := by
  unfold bitLength
  split
  · omega
  · have := (Nat.log2_lt (by omega)).mpr hn; omega

theorem bitLength_eq (n L : Nat) (h1 : 1 ≤ L) (hlo : 2 ^ (L - 1) ≤ n) (hhi : n < 2 ^ L) : bitLength n = L := by
  have hn : n ≠ 0 := by have := Nat.two_pow_pos (L - 1); omega
  unfold bitLength
  simp only [hn, ite_false]
  have := (Nat.log2_eq_iff hn).mpr ⟨hlo, by rw [Nat.sub_add_cancel h1]; exact hhi⟩
  omega

/-- The 26-bit window: `|T| * 2^(32 - L) / 2^6` is `group_sum`'s shift of `|T|` to 26 bits, with its top bit set. -/
theorem window (N : Nat) (hN0 : N ≠ 0) (hN : N < 2 ^ 32) :
    N * 2 ^ (32 - bitLength N) / 2 ^ 6 =
        (if bitLength N > 26 then N >>> (bitLength N - 26) else N <<< (26 - bitLength N)) ∧
      2 ^ 25 ≤ N * 2 ^ (32 - bitLength N) / 2 ^ 6 ∧ N * 2 ^ (32 - bitLength N) / 2 ^ 6 < 2 ^ 26 := by
  obtain ⟨hlo, hhi, h1⟩ := bitLength_spec N hN0
  have hle := bitLength_le N 32 hN
  generalize bitLength N = L at *
  have hlo' : 2 ^ 31 ≤ N * 2 ^ (32 - L) := by
    have : 2 ^ 31 = 2 ^ (L - 1) * 2 ^ (32 - L) := by rw [← Nat.pow_add]; congr 1; omega
    rw [this]; exact Nat.mul_le_mul_right _ hlo
  have hhi' : N * 2 ^ (32 - L) < 2 ^ 32 := by
    have : 2 ^ 32 = 2 ^ L * 2 ^ (32 - L) := by rw [← Nat.pow_add]; congr 1; omega
    rw [this]; exact Nat.mul_lt_mul_of_pos_right hhi (Nat.two_pow_pos _)
  refine ⟨?_, ?_, ?_⟩
  · split
    · rw [Nat.shiftRight_eq_div_pow]
      have : 2 ^ 6 = 2 ^ (32 - L) * 2 ^ (L - 26) := by rw [← Nat.pow_add]; congr 1; omega
      rw [this, Nat.mul_comm N, Nat.mul_div_mul_left _ _ (Nat.two_pow_pos _)]
    · rw [Nat.shiftLeft_eq]
      have : 2 ^ (32 - L) = 2 ^ (26 - L) * 2 ^ 6 := by rw [← Nat.pow_add]; congr 1; omega
      rw [this, ← Nat.mul_assoc, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
  · exact (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).mpr (by omega)
  · exact (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr (by omega)

/-- The subnormal shift and the rescale: `(m >> k) >> 2`, or `m >> 2`, is `m / 2^min(e_fin - e_out + 2, 36)`. -/
theorem final_shift (m : Nat) (hm : m < 2 ^ 26) (e_out : Int) :
    (if e_out < -126 then m >>> (-126 - e_out).toNat else m) >>> 2 =
      m / 2 ^ (min (max e_out (-126) - e_out + 2) 36).toNat := by
  rw [Nat.shiftRight_eq_div_pow]
  split
  · rw [Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul, ← Nat.pow_add]
    by_cases hk : (-126 - e_out).toNat + 2 ≤ 36
    · congr 2; omega
    · have h1 : m < 2 ^ ((-126 - e_out).toNat + 2) :=
        Nat.lt_of_lt_of_le hm (Nat.pow_le_pow_right (by decide) (by omega))
      have h2 : m < 2 ^ (min (max e_out (-126) - e_out + 2) 36).toNat :=
        Nat.lt_of_lt_of_le hm (Nat.pow_le_pow_right (by decide) (by omega))
      rw [Nat.div_eq_of_lt h1, Nat.div_eq_of_lt h2]
  · congr 2; omega

/-! ## Reading columns off the predicates -/

theorem pow2_pos (k : Nat) : (0 : Int) < 2 ^ k := by
  have := Nat.two_pow_pos k
  exact_mod_cast this

/-- Division with remainder against `2^k`, the remainder bounded through `r * 2^(n - k) ∈ [0, 2^n)`. -/
theorem div_of_rem (s q r ri : Int) (k n : Nat) (hk : k ≤ n) (h1 : q * 2 ^ k = s - r)
    (h2 : r * 2 ^ (n - k) = ri) (h3 : 0 ≤ ri) (h4 : ri < 2 ^ n) : q = s / 2 ^ k ∧ 0 ≤ r ∧ r < 2 ^ k := by
  have hp := pow2_pos (n - k)
  have hp' := pow2_pos k
  have hn : (2 : Int) ^ n = 2 ^ k * 2 ^ (n - k) := by rw [← Int.pow_add]; congr 1; omega
  have hr0 : 0 ≤ r := by
    rw [← h2] at h3
    exact Int.le_of_mul_le_mul_right (by simpa using h3) hp
  have hr1 : r < 2 ^ k := by
    rw [← h2, hn] at h4
    exact Int.lt_of_mul_lt_mul_right h4 (Int.le_of_lt hp)
  refine ⟨?_, hr0, hr1⟩
  have := (Int.ediv_emod_unique hp').mpr ⟨(by rw [Int.mul_comm]; omega : r + 2 ^ k * q = s), hr0, hr1⟩
  exact this.1.symm

end Verity.TC.Proofs
