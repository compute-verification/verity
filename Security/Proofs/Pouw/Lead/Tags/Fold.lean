import Definitions.Core.Fp32
import Proofs.Core.Fp32.Round
import Mathlib.Data.Int.ModEq

/-!
# The epilogue's fold is exact while `|p| ≤ 2^22`

The lead's int32 accumulator holds `D ≡ p + ρ + c (mod 2^32)`: the row source `ρ`, the column source `c` (the code,
`κ·G_j` and the coin lanes' weighted sum), integers of any size, and `p`, the rest: the product `(A·Bᵀ)_ij` served
plain, and `(A·Bᵀ)_ij + n_ij` served noisy, `n` the noise. The window is on that `p`. The epilogue folds both
sources out with one IADD3, `D + (MAGIC_BITS - ρ) - c` mod `2^32`, and reads the word as FP32. While `|p| ≤ 2^22` that
word is `MAGIC_BITS + p` whatever the sources (`iadd3_fold`), the FP32 word of `MAGIC + p`, `MAGIC = 1.5 · 2^23`
(`decode_fold`). The FFMA `v · s - MAGIC · s` then gives stock's word, FMUL of `I2F(p)` by `s`, for every positive finite
column scale `s` whose `-MAGIC · s` is an FP32 word (`ffma_fold`), and the epilogue's BF16 output is stock's
(`epilogue_fold`). FMUL's `-MAGIC · s` is exact for a scale with a 22-bit significand (`negMagic_exact`), and `I2F(p)`
is exact (`exists_int_word`): `roundRn` takes a magnitude `M · 2^k` with `M < 2^24` to its own word
(`decode_roundRn_grid`).

Magnitudes are `Verity.Fp32.decode`'s, in units of `2^-149`: the FP32 word of an integer `p` decodes to the sign
`p < 0` and the magnitude `|p| · 2^149`.
-/

namespace Pouw.SecurityProofs.Lead.Tags

open Verity.Fp32 Verity.Fp32.Round Verity.Round

/-- `1.5 · 2^23`, the value of the magic word. -/
def MAGIC : ℕ := 12582912

/-- The FP32 word of `MAGIC`. -/
def MAGIC_BITS : ℕ := 0x4B400000

/-- The FP32 word of `-MAGIC`. -/
def NEG_MAGIC_BITS : ℕ := 0xCB400000

theorem abs_le_22 {p : ℤ} (hp : |p| ≤ 2 ^ 22) : -4194304 ≤ p ∧ p ≤ 4194304 := by
  rw [abs_le] at hp; norm_num at hp; exact hp

/-- **The IADD3 fold**: `D + (MAGIC_BITS - ρ) - c` mod `2^32` is `MAGIC_BITS + p` for every `D ≡ p + ρ + c` and
`|p| ≤ 2^22`, whatever the row and column sources are. -/
theorem iadd3_fold {p ρ c D : ℤ} (hp : |p| ≤ 2 ^ 22) (hD : D ≡ p + ρ + c [ZMOD 2 ^ 32]) :
    (D + ((MAGIC_BITS : ℤ) - ρ) - c) % 2 ^ 32 = MAGIC_BITS + p := by
  have hp' := abs_le_22 hp
  have h32 : (2 : ℤ) ^ 32 = 4294967296 := by norm_num
  rw [Int.ModEq, h32] at hD
  rw [h32]
  simp only [MAGIC_BITS]
  omega

/-- The fold's word decodes to `MAGIC + p`. -/
theorem decode_fold {p : ℤ} (hp : |p| ≤ 2 ^ 22) :
    decode (MAGIC_BITS + p).toNat = .fin false ((MAGIC + p).toNat * 2 ^ 149) := by
  have hp' := abs_le_22 hp
  have hq : (MAGIC + p).toNat ≤ 2 ^ 24 := by simp only [MAGIC]; norm_num; omega
  have hj : 149 = 0 ∨ 2 ^ 23 ≤ (MAGIC + p).toNat := Or.inr (by simp only [MAGIC]; norm_num; omega)
  have hw : 149 * 2 ^ 23 + (MAGIC + p).toNat < 0x7F800000 := by simp only [MAGIC]; norm_num; omega
  have e : (MAGIC_BITS + p).toNat = zeroWord false ||| (149 * 2 ^ 23 + (MAGIC + p).toNat) := by
    simp only [zeroWord, Bool.false_eq_true, ↓reduceIte, Nat.zero_or, MAGIC_BITS, MAGIC]; norm_num; omega
  rw [e, decode_word false hq hj hw]

/-- `roundRn`'s words are never NaNs. -/
theorem f32IsNan_roundRn (s : Bool) (n : ℕ) {d : ℕ} (hd : 0 < d) : f32IsNan (roundRn s n d) = false := by
  obtain ⟨j, q, -, -, -, heq⟩ := roundRn_eq s n hd
  rw [heq]
  generalize hw : (if j * 2 ^ 23 + q ≥ 0x7F800000 then 0x7F800000 else j * 2 ^ 23 + q) = w
  have hw' : w ≤ 0x7F800000 := by rw [← hw]; split_ifs <;> omega
  rcases hw'.lt_or_eq with hlt | rfl
  · have hm : (zeroWord s ||| w) &&& 0x7F800000 ≤ w := by
      calc (zeroWord s ||| w) &&& 0x7F800000 = ((zeroWord s ||| w) &&& (2 ^ 31 - 1)) &&& 0x7F800000 := by
            rw [Nat.and_assoc]; rfl
        _ = (zeroWord s ||| w) % 2 ^ 31 &&& 0x7F800000 := by rw [Nat.and_two_pow_sub_one_eq_mod]
        _ ≤ (zeroWord s ||| w) % 2 ^ 31 := Nat.and_le_left
        _ = w := by rw [zeroWord_or s (by omega)]; cases s <;> simp <;> omega
    unfold f32IsNan
    have : ((zeroWord s ||| w) &&& 0x7F800000 == 0x7F800000) = false := by
      rw [beq_eq_false_iff_ne]; omega
    rw [this, Bool.false_and]
  · cases s <;> rfl

theorem f32GpuNan_roundRn (s : Bool) (n : ℕ) {d : ℕ} (hd : 0 < d) : f32GpuNan (roundRn s n d) = roundRn s n d := by
  unfold f32GpuNan; rw [f32IsNan_roundRn s n hd]; rfl

/-- **The FFMA fold**: for `|p| ≤ 2^22`, a positive finite column scale `s` and a word `c` of exactly `-MAGIC · s`,
`FFMA(MAGIC_BITS + p, s, c)` is stock's `FMUL(x, s)` for the word `x` of `p`. -/
theorem ffma_fold {p : ℤ} (hp : |p| ≤ 2 ^ 22) {s c x S : ℕ} (hS : 0 < S) (hs : decode s = .fin false S)
    (hc : decode c = .fin true (MAGIC * S)) (hx : decode x = .fin (decide (p < 0)) (p.natAbs * 2 ^ 149)) :
    f32FmaV2 (MAGIC_BITS + p).toNat s c = f32MulV2 x s := by
  have hp' := abs_le_22 hp
  have hM : MAGIC = 12582912 := rfl
  have hd : 0 < 1 <<< 149 := by rw [Nat.shiftLeft_eq]; positivity
  unfold f32FmaV2 f32FmaBits f32MulV2
  rw [decode_fold hp, hs, hc, hx]
  simp only [Decoded.mul, Decoded.addRn, roundSum, Bool.xor_false, Bool.false_eq_true,
    ↓reduceIte]
  have hT : 0 < p.natAbs * 2 ^ 149 * S ∨ p = 0 := by
    rcases eq_or_ne p 0 with h | h
    · exact Or.inr h
    · exact Or.inl (Nat.mul_pos (Nat.mul_pos (Int.natAbs_pos.2 h) (by positivity)) hS)
  rcases lt_trichotomy p 0 with hn | h0 | hpos
  · have e : (MAGIC * S) <<< 149 = (MAGIC + p).toNat * 2 ^ 149 * S + p.natAbs * 2 ^ 149 * S := by
      rw [Nat.shiftLeft_eq, show MAGIC = (MAGIC + p).toNat + p.natAbs by omega]
      rw [show ((((MAGIC + p).toNat + p.natAbs : ℕ) : ℤ) + p).toNat = (MAGIC + p).toNat by omega]
      ring
    have hT' := hT.resolve_right hn.ne
    rw [e, ite_eq_right (by omega), ite_eq_left (by omega), Nat.add_sub_cancel_left, decide_eq_true hn,
      f32GpuNan_roundRn _ _ hd]
  · subst h0
    have e : (MAGIC + (0 : ℤ)).toNat * 2 ^ 149 * S = (MAGIC * S) <<< 149 := by
      rw [Nat.shiftLeft_eq, show (MAGIC + (0 : ℤ)).toNat = MAGIC by omega]; ring
    rw [e, ite_eq_right (lt_irrefl _), ite_eq_right (lt_irrefl _)]
    simp only [Int.natAbs_zero, Nat.zero_mul, lt_irrefl, decide_false, roundRn_zero false hd]
    rfl
  · have e : (MAGIC + p).toNat * 2 ^ 149 * S = (MAGIC * S) <<< 149 + p.natAbs * 2 ^ 149 * S := by
      rw [Nat.shiftLeft_eq, show (MAGIC + p).toNat = MAGIC + p.natAbs by omega]; ring
    have hT' := hT.resolve_right hpos.ne'
    rw [e, ite_eq_left (by omega), Nat.add_sub_cancel_left, decide_eq_false (not_lt.2 hpos.le),
      f32GpuNan_roundRn _ _ hd]

/-- **The epilogue is stock's**: for `D ≡ p + ρ + c` with `|p| ≤ 2^22`, the IADD3 fold, the FFMA by a positive column
scale `s` with `c'` exactly `-MAGIC · s`, the FMUL by the row scale and the BF16 conversion give stock's
`bf16(FMUL(FMUL(I2F(p), s), sr))`, whatever the sources `ρ` and `c` are. -/
theorem epilogue_fold {p ρ c D : ℤ} (hp : |p| ≤ 2 ^ 22) (hD : D ≡ p + ρ + c [ZMOD 2 ^ 32]) {s c' x S : ℕ}
    (hS : 0 < S) (hs : decode s = .fin false S) (hc : decode c' = .fin true (MAGIC * S))
    (hx : decode x = .fin (decide (p < 0)) (p.natAbs * 2 ^ 149)) (sr nan : ℕ) :
    f32ToBf16RnBits (f32MulV2 (f32FmaV2 ((D + ((MAGIC_BITS : ℤ) - ρ) - c) % 2 ^ 32).toNat s c') sr) nan =
      f32ToBf16RnBits (f32MulV2 (f32MulV2 x s) sr) nan := by
  rw [iadd3_fold hp hD, ffma_fold hp hS hs hc hx]

/-! ## Exact roundings -/

set_option exponentiation.threshold 300 in
/-- **A grid magnitude rounds to itself**: `M · 2^k` units with `M < 2^24`, below `2^277`, scaled by any `d > 0`, is
`roundRn`'s word of that magnitude. -/
theorem decode_roundRn_grid (s : Bool) {M k d : ℕ} (hd : 0 < d) (hM : M < 2 ^ 24) (hN : M * 2 ^ k < 2 ^ 277) :
    decode (roundRn s (M * 2 ^ k * d) d) = .fin s (M * 2 ^ k) := by
  obtain ⟨j, q, hj, hlt, hq, heq⟩ := roundRn_eq s (M * 2 ^ k * d) hd
  have hj' : j = 0 ∨ 2 ^ (j + 23) ≤ M * 2 ^ k := hj.imp id fun h => Nat.le_of_mul_le_mul_right h hd
  have hlt' : M * 2 ^ k < 2 ^ (j + 24) := Nat.lt_of_mul_lt_mul_right hlt
  have hjk : j ≤ k := by
    rcases hj' with h | h
    · omega
    · by_contra hc
      have h1 : M * 2 ^ k < 2 ^ 24 * 2 ^ k := Nat.mul_lt_mul_of_pos_right hM (by positivity)
      have h2 : 2 ^ 24 * 2 ^ k ≤ 2 ^ (j + 23) := by
        rw [← pow_add]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
      omega
  have hN' : M * 2 ^ k = M * 2 ^ (k - j) * 2 ^ j := by rw [mul_assoc, ← pow_add, Nat.sub_add_cancel hjk]
  generalize M * 2 ^ (k - j) = N' at hN'
  rw [hN'] at hq hj' hlt' heq ⊢
  have hx : ((N' * 2 ^ j * d : ℕ) : ℚ) / (d * 2 ^ j) = N' := by
    have : (0 : ℚ) < d * 2 ^ j := by positivity
    push_cast; field_simp
  rw [hx] at hq
  obtain rfl : q = N' := le_antisymm (hq.le (by linarith)) (hq.ge (by linarith))
  have hq24 : q < 2 ^ 24 := by
    rw [pow_add, Nat.mul_comm (2 ^ j)] at hlt'; exact Nat.lt_of_mul_lt_mul_right hlt'
  have hq23 : j = 0 ∨ 2 ^ 23 ≤ q := hj'.imp id fun h => by
    rw [pow_add, Nat.mul_comm (2 ^ j)] at h; exact Nat.le_of_mul_le_mul_right h (by positivity)
  have hj253 : j ≤ 253 := by
    rcases hj' with h | h
    · omega
    · by_contra hc
      have : 2 ^ 277 ≤ 2 ^ (j + 23) := Nat.pow_le_pow_right (by norm_num) (by omega)
      omega
  have h23 : (2 : ℕ) ^ 23 = 8388608 := by norm_num
  have h24 : (2 : ℕ) ^ 24 = 16777216 := by norm_num
  have hw : j * 2 ^ 23 + q < 0x7F800000 := by rw [h23]; rw [h24] at hq24; omega
  rw [heq, ite_eq_right (by omega), decode_word s hq24.le hq23 hw]

set_option exponentiation.threshold 300 in
/-- `I2F` is exact below `2^24`: the integer `p` has an FP32 word, of sign `p < 0` and magnitude `|p| · 2^149`. -/
theorem exists_int_word {p : ℤ} (hp : |p| < 2 ^ 24) :
    ∃ x, decode x = .fin (decide (p < 0)) (p.natAbs * 2 ^ 149) := by
  have ha : p.natAbs < 2 ^ 24 := by
    have : ((p.natAbs : ℕ) : ℤ) < 2 ^ 24 := by rw [Int.natCast_natAbs]; exact hp
    exact_mod_cast this
  refine ⟨roundRn (decide (p < 0)) (p.natAbs * 2 ^ 149 * 1) 1, decode_roundRn_grid _ one_pos ha ?_⟩
  calc p.natAbs * 2 ^ 149 < 2 ^ 24 * 2 ^ 149 := Nat.mul_lt_mul_of_pos_right ha (by positivity)
    _ ≤ 2 ^ 277 := by rw [← pow_add]; exact Nat.pow_le_pow_right (by norm_num) (by norm_num)

theorem decode_negMagic : decode NEG_MAGIC_BITS = .fin true (MAGIC * 2 ^ 149) := by
  have e : NEG_MAGIC_BITS = zeroWord true ||| (149 * 2 ^ 23 + MAGIC) := by decide
  rw [e, decode_word true (by simp only [MAGIC]; norm_num) (Or.inr (by simp only [MAGIC]; norm_num))
    (by simp only [MAGIC]; norm_num)]

/-- **The column constant is exact**: for a column scale of a 22-bit significand, `M · 2^k` units with `0 < M < 2^22`,
whose `MAGIC` multiple is below `2^277` units, FMUL's `-MAGIC · s` is the word of exactly `-MAGIC · s`. -/
theorem negMagic_exact {s M k : ℕ} (hs : decode s = .fin false (M * 2 ^ k)) (hM : M < 2 ^ 22)
    (hS : MAGIC * (M * 2 ^ k) < 2 ^ 277) : decode (f32MulV2 NEG_MAGIC_BITS s) = .fin true (MAGIC * (M * 2 ^ k)) := by
  unfold f32MulV2
  rw [decode_negMagic, hs]
  simp only [Decoded.mul, Bool.true_xor, Bool.not_false]
  have e1 : MAGIC * 2 ^ 149 * (M * 2 ^ k) = 3 * M * 2 ^ (22 + k) * (1 <<< 149) := by
    rw [Nat.shiftLeft_eq, pow_add]; simp only [MAGIC]; ring
  have e2 : MAGIC * (M * 2 ^ k) = 3 * M * 2 ^ (22 + k) := by rw [pow_add]; simp only [MAGIC]; ring
  rw [e1, e2]
  rw [e2] at hS
  exact decode_roundRn_grid true (by rw [Nat.shiftLeft_eq]; positivity) (by omega) hS

end Pouw.SecurityProofs.Lead.Tags
