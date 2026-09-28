import Definitions.Core.Fp32
import Proofs.Core.Round

/-!
# `roundRn` is the nearest FP32 word, ties to even

`Verity.Fp32.roundRn s n d` rounds the exact value `(-1)^s · n / d` units of `2^-149`. `roundRn_eq` opens it: the
exponent `j` of `n / d`'s binade less 23, clamped at the subnormals (`j = 0`, or `2^(j + 23) ≤ n / d`; always
`n / d < 2^(j + 24)`), the significand `q`, which is `n / (d · 2^j)` rounded to nearest even
(`Verity.Round.RoundsEven`), and the word `j · 2^23 + q` under the sign, `±inf` from `0x7F800000` up.

Read through core's `decode`, every finite word's magnitude is `M · 2^k` units with `M < 2^24` (`decode_grid`), so
`Verity.Round.nearest_grid` applies. Below `2^277 - 2^252` units (`2^128 - 2^103`, half an ulp past the largest finite
word) the word is finite, and its signed value is the nearest to `(-1)^s · n / d` of every finite word's, subnormals and
both zeros included; a finite word of another value equally near makes the word even, its last bit clear
(`roundRn_fin`). From `2^128 - 2^103` up it is `±inf` (`roundRn_inf`), and a zero is the zero of its sign
(`roundRn_zero`).
-/

namespace Verity.Fp32.Round

open Verity.Fp32 Verity.Round

/-! ## Words -/

theorem or_hidden {m : ℕ} (hm : m < 2 ^ 23) : m ||| 0x800000 = 2 ^ 23 + m := by
  have h := Nat.shiftLeft_add_eq_or_of_lt hm 1
  rw [Nat.one_shiftLeft] at h
  rw [Nat.or_comm]; exact h.symm

theorem zeroWord_or (s : Bool) {w : ℕ} (hw : w < 2 ^ 31) : zeroWord s ||| w = (if s then 2 ^ 31 else 0) + w := by
  cases s
  · show 0 ||| w = 0 + w
    rw [Nat.zero_or, Nat.zero_add]
  · show 2 ^ 31 ||| w = 2 ^ 31 + w
    have h := Nat.shiftLeft_add_eq_or_of_lt hw 1
    rw [Nat.one_shiftLeft] at h; exact h.symm

/-- `decode`'s fields of a word below `0x7F800000` under a sign. -/
theorem fields (s : Bool) {w : ℕ} (hw : w < 0x7F800000) :
    (zeroWord s ||| w) >>> 23 &&& 0xFF = w / 2 ^ 23 ∧ (zeroWord s ||| w) &&& 0x7FFFFF = w % 2 ^ 23 ∧
      ((zeroWord s ||| w) >>> 31 &&& 1 == 1) = s := by
  have h8 : (0xFF : ℕ) = 2 ^ 8 - 1 := rfl
  have h23 : (0x7FFFFF : ℕ) = 2 ^ 23 - 1 := rfl
  rw [h8, h23, Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod, Nat.and_one_is_mod,
    Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow, zeroWord_or s (by omega)]
  cases s
  · show (0 + w) / 2 ^ 23 % 2 ^ 8 = w / 2 ^ 23 ∧ (0 + w) % 2 ^ 23 = w % 2 ^ 23 ∧ ((0 + w) / 2 ^ 31 % 2 == 1) = false
    rw [Nat.zero_add, Nat.div_eq_of_lt (by omega : w < 2 ^ 31)]
    exact ⟨by omega, rfl, rfl⟩
  · show (2 ^ 31 + w) / 2 ^ 23 % 2 ^ 8 = w / 2 ^ 23 ∧ (2 ^ 31 + w) % 2 ^ 23 = w % 2 ^ 23 ∧
      ((2 ^ 31 + w) / 2 ^ 31 % 2 == 1) = true
    rw [show (2 ^ 31 + w) / 2 ^ 31 % 2 = 1 by omega]
    exact ⟨by omega, by omega, rfl⟩

theorem decode_or (s : Bool) {w : ℕ} (hw : w < 0x7F800000) :
    decode (zeroWord s ||| w) = if w / 2 ^ 23 = 0 then .fin s (w % 2 ^ 23)
      else .fin s ((w % 2 ^ 23 ||| 0x800000) <<< (w / 2 ^ 23 - 1)) := by
  obtain ⟨h1, h2, h3⟩ := fields s hw
  simp only [decode, h1, h2, h3]
  rw [ite_eq_right (show ¬ (w / 2 ^ 23 = 0xFF) by omega)]

/-- The word `j · 2^23 + q` under a sign, with `q ≤ 2^24` and `q ≥ 2^23` unless `j = 0`, decodes to the magnitude
`q · 2^j`: a subnormal for `q < 2^23`, and a carry `q = 2^24` into the next binade. -/
theorem decode_word (s : Bool) {j q : ℕ} (hq : q ≤ 2 ^ 24) (hj : j = 0 ∨ 2 ^ 23 ≤ q)
    (hw : j * 2 ^ 23 + q < 0x7F800000) : decode (zeroWord s ||| (j * 2 ^ 23 + q)) = .fin s (q * 2 ^ j) := by
  rw [decode_or s hw]
  rcases Nat.lt_or_ge q (2 ^ 23) with h1 | h1
  · obtain rfl : j = 0 := hj.resolve_right (by omega)
    rw [ite_eq_left (by omega)]
    congr 1; rw [pow_zero, mul_one]; omega
  · rw [ite_eq_right (by omega)]
    rcases Nat.lt_or_ge q (2 ^ 24) with h2 | h2
    · rw [show (j * 2 ^ 23 + q) % 2 ^ 23 = q - 2 ^ 23 by omega, show (j * 2 ^ 23 + q) / 2 ^ 23 - 1 = j by omega,
        or_hidden (by omega), Nat.shiftLeft_eq, show 2 ^ 23 + (q - 2 ^ 23) = q by omega]
    · obtain rfl : q = 2 ^ 24 := by omega
      rw [show (j * 2 ^ 23 + 2 ^ 24) % 2 ^ 23 = 0 by omega,
        show (j * 2 ^ 23 + 2 ^ 24) / 2 ^ 23 - 1 = j + 1 by omega, Nat.zero_or, Nat.shiftLeft_eq]
      congr 1; ring

/-- `decode`'s three outcomes, by the exponent and mantissa fields. -/
theorem decode_cases (u : ℕ) :
    ((u >>> 23) &&& 0xFF = 0xFF ∧ u &&& 0x7FFFFF = 0 ∧ decode u = .inf ((u >>> 31) &&& 1 == 1)) ∨
    ((u >>> 23) &&& 0xFF = 0xFF ∧ u &&& 0x7FFFFF ≠ 0 ∧ decode u = .nan) ∨
    ((u >>> 23) &&& 0xFF ≠ 0xFF ∧ decode u = .fin ((u >>> 31) &&& 1 == 1)
      (if (u >>> 23) &&& 0xFF = 0 then u &&& 0x7FFFFF
        else ((u &&& 0x7FFFFF) ||| 0x800000) <<< (((u >>> 23) &&& 0xFF) - 1))) := by
  by_cases h1 : (u >>> 23) &&& 0xFF = 0xFF
  · by_cases h2 : u &&& 0x7FFFFF = 0
    · refine Or.inl ⟨h1, h2, ?_⟩
      simp only [decode]
      rw [ite_eq_left h1, ite_eq_left h2]
    · refine Or.inr (Or.inl ⟨h1, h2, ?_⟩)
      simp only [decode]
      rw [ite_eq_left h1, ite_eq_right h2]
  · refine Or.inr (Or.inr ⟨h1, ?_⟩)
    simp only [decode]
    rw [ite_eq_right h1]
    by_cases h3 : (u >>> 23) &&& 0xFF = 0
    · rw [ite_eq_left h3, ite_eq_left h3]
    · rw [ite_eq_right h3, ite_eq_right h3]

theorem decode_fin_fields {u : ℕ} {s : Bool} {m : ℕ} (h : decode u = .fin s m) :
    (u >>> 23) &&& 0xFF ≠ 0xFF ∧ s = ((u >>> 31) &&& 1 == 1) ∧
      m = if (u >>> 23) &&& 0xFF = 0 then u &&& 0x7FFFFF
        else ((u &&& 0x7FFFFF) ||| 0x800000) <<< (((u >>> 23) &&& 0xFF) - 1) := by
  rcases decode_cases u with ⟨-, -, h3⟩ | ⟨-, -, h3⟩ | ⟨h1, h3⟩
  · rw [h3] at h; cases h
  · rw [h3] at h; cases h
  · rw [h3] at h
    obtain ⟨h4, h5⟩ := Decoded.fin.inj h
    exact ⟨h1, h4.symm, h5.symm⟩

/-- Every finite word's magnitude is a 24-bit significand times a power of two. -/
theorem decode_grid {u : ℕ} {s : Bool} {m : ℕ} (h : decode u = .fin s m) : ∃ M k, M < 2 ^ 24 ∧ m = M * 2 ^ k := by
  obtain ⟨-, -, rfl⟩ := decode_fin_fields h
  rw [show (0x7FFFFF : ℕ) = 2 ^ 23 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  by_cases h0 : (u >>> 23) &&& 0xFF = 0
  · rw [ite_eq_left h0]
    exact ⟨u % 2 ^ 23, 0, by omega, by rw [pow_zero, Nat.mul_one]⟩
  · rw [ite_eq_right h0, or_hidden (Nat.mod_lt _ (by norm_num))]
    exact ⟨_, _, by omega, Nat.shiftLeft_eq _ _⟩

/-! ## `roundRn` opened -/

theorem log2_bounds (n : ℕ) {d : ℕ} (hd : 0 < d) :
    ((n / d).log2 - 23 = 0 ∨ 2 ^ ((n / d).log2 - 23 + 23) * d ≤ n) ∧ n < 2 ^ ((n / d).log2 - 23 + 24) * d := by
  have hlt : n < (n / d + 1) * d := (Nat.div_lt_iff_lt_mul hd).1 (Nat.lt_succ_self _)
  have hle : n / d * d ≤ n := Nat.div_mul_le_self n d
  have h1 : n / d + 1 ≤ 2 ^ ((n / d).log2 + 1) := Nat.lt_log2_self
  have h2 : 2 ^ ((n / d).log2 + 1) ≤ 2 ^ ((n / d).log2 - 23 + 24) := Nat.pow_le_pow_right (by norm_num) (by omega)
  refine ⟨?_, lt_of_lt_of_le hlt (Nat.mul_le_mul_right d (h1.trans h2))⟩
  by_cases hL : (n / d).log2 < 23
  · left; omega
  · right
    have hN : n / d ≠ 0 := by intro h; rw [h, Nat.log2_zero] at hL; exact hL (by norm_num)
    rw [show (n / d).log2 - 23 + 23 = (n / d).log2 by omega]
    exact (Nat.mul_le_mul_right d (Nat.log2_self_le hN)).trans hle

/-- **`roundRn` opened**: the clamped binade exponent `j`, the significand `q` rounded to nearest even, and the word. -/
theorem roundRn_eq (s : Bool) (n : ℕ) {d : ℕ} (hd : 0 < d) :
    ∃ j q : ℕ, (j = 0 ∨ 2 ^ (j + 23) * d ≤ n) ∧ n < 2 ^ (j + 24) * d ∧
      RoundsEven ((n : ℚ) / (d * 2 ^ j)) q ∧
      roundRn s n d = zeroWord s ||| (if j * 2 ^ 23 + q ≥ 0x7F800000 then 0x7F800000 else j * 2 ^ 23 + q) := by
  obtain ⟨h1, h2⟩ := log2_bounds n hd
  have hD : 0 < d * 2 ^ ((n / d).log2 - 23) := Nat.mul_pos hd (pow_pos (by norm_num) _)
  have hq := roundsEven_div n (d * 2 ^ ((n / d).log2 - 23)) hD
  rw [show ((d * 2 ^ ((n / d).log2 - 23) : ℕ) : ℚ) = (d : ℚ) * 2 ^ ((n / d).log2 - 23) by push_cast; ring] at hq
  refine ⟨_, _, h1, h2, hq, ?_⟩
  simp only [roundRn, Nat.shiftLeft_eq]

/-! ## The nearest word -/

/-- A zero rounds to the zero of its sign. -/
theorem roundRn_zero (s : Bool) {d : ℕ} (hd : 0 < d) : roundRn s 0 d = zeroWord s := by
  obtain ⟨j, q, hj, -, hq, heq⟩ := roundRn_eq s 0 hd
  have hj0 : j = 0 := hj.resolve_right (by have := Nat.mul_pos (pow_pos (by norm_num : 0 < 2) (j + 23)) hd; omega)
  have hq0 : q = 0 := Nat.le_zero.1 (hq.le (by rw [Nat.cast_zero, zero_div]; norm_num))
  rw [heq, hj0, hq0, ite_eq_right (by norm_num), Nat.zero_mul, Nat.zero_add, Nat.or_zero]

set_option exponentiation.threshold 300 in
/-- **`roundRn` is the nearest finite word, ties to even**, below `2^128 - 2^103` (`2^277 - 2^252` units): the word
decodes to the sign `s` and a magnitude `m` whose signed value is at least as near to `(-1)^s · n / d` as every finite
word's, and when a finite word of another value is as near, the word is even. -/
theorem roundRn_fin (s : Bool) {n d : ℕ} (hn : 0 < n) (hd : 0 < d) (h : (n : ℚ) / d < 2 ^ 277 - 2 ^ 252) :
    ∃ m : ℕ, decode (roundRn s n d) = .fin s m ∧
      (∀ u s' m', decode u = .fin s' m' →
        |signed s m - signed s ((n : ℚ) / d)| ≤ |signed s' m' - signed s ((n : ℚ) / d)|) ∧
      (∀ u s' m', decode u = .fin s' m' → signed s' m' ≠ signed s m →
        |signed s' m' - signed s ((n : ℚ) / d)| = |signed s m - signed s ((n : ℚ) / d)| →
        roundRn s n d % 2 = 0) := by
  obtain ⟨j, q, hj, hlt, hq, heq⟩ := roundRn_eq s n hd
  have hd' : (0 : ℚ) < d := by exact_mod_cast hd
  have hx0 : (0 : ℚ) < (n : ℚ) / d := div_pos (by exact_mod_cast hn) hd'
  rw [← div_div] at hq
  have hy24 : (n : ℚ) / d / 2 ^ j < 2 ^ 24 := qdiv_lt (by positivity) (by
    rw [← pow_add, Nat.add_comm 24 j]; exact qdiv_lt hd' (by exact_mod_cast hlt))
  have hq24 : q ≤ 2 ^ 24 := hq.le (c := 2 ^ 24) (by push_cast; linarith)
  have hlow : j = 0 ∨ (2 : ℚ) ^ (j + 23) ≤ n / d := hj.imp id fun h => qle_div hd' (by exact_mod_cast h)
  have hq23 : j = 0 ∨ 2 ^ 23 ≤ q := hlow.imp id fun h => hq.ge (c := 2 ^ 23) (by
    push_cast
    have : (2 : ℚ) ^ 23 ≤ n / d / 2 ^ j := qle_div (by positivity) (by rw [← pow_add, Nat.add_comm 23 j]; exact h)
    linarith)
  have hw : j * 2 ^ 23 + q < 0x7F800000 := by
    rcases Nat.lt_or_ge j 253 with hj1 | hj1
    · omega
    · rcases Nat.lt_or_ge j 254 with hj2 | hj2
      · obtain rfl : j = 253 := by omega
        have : q < 2 ^ 24 := hq.lt (c := 2 ^ 24) (by
          push_cast
          exact qdiv_lt (by positivity) (by linarith))
        omega
      · exfalso
        have h1 : (2 : ℚ) ^ (j + 23) ≤ (n : ℚ) / d := hlow.resolve_left (by omega)
        have h2 : (2 : ℚ) ^ 277 ≤ 2 ^ (j + 23) := by exact_mod_cast Nat.pow_le_pow_right (by norm_num) (by omega)
        have h3 : (2 : ℚ) ^ 277 - 2 ^ 252 < 2 ^ 277 := by norm_num
        linarith
  have hword : roundRn s n d = zeroWord s ||| (j * 2 ^ 23 + q) := by rw [heq, ite_eq_right (by omega)]
  have hnear : ∀ M k : ℕ, M < 2 ^ 24 → |(q : ℚ) * 2 ^ j - n / d| ≤ |(M : ℚ) * 2 ^ k - n / d| ∧
      (2 * |(q : ℚ) - n / d / 2 ^ j| < 1 → (M : ℚ) * 2 ^ k ≠ q * 2 ^ j →
        |(q : ℚ) * 2 ^ j - n / d| < |(M : ℚ) * 2 ^ k - n / d|) := fun M k hM =>
    nearest_grid (p := 23) (k₀ := 0) hlow hq.near hM (Nat.zero_le k)
  refine ⟨q * 2 ^ j, by rw [hword, decode_word s hq24 hq23 hw], ?_, ?_⟩
  · intro u s' m' hu
    obtain ⟨M, k, hM, rfl⟩ := decode_grid hu
    push_cast
    exact signed_near s s' hx0.le (by positivity) (hnear M k hM).1
  · intro u s' m' hu hne heq2
    rw [hword, zeroWord_or s (by omega)]
    by_cases ht : 2 * |(q : ℚ) - n / d / 2 ^ j| = 1
    · have := hq.tie ht
      cases s
      · show (0 + (j * 2 ^ 23 + q)) % 2 = 0
        omega
      · show (2 ^ 31 + (j * 2 ^ 23 + q)) % 2 = 0
        omega
    · exfalso
      have ht' : 2 * |(q : ℚ) - n / d / 2 ^ j| < 1 := lt_of_le_of_ne hq.near ht
      obtain ⟨M, k, hM, rfl⟩ := decode_grid hu
      push_cast at hne heq2
      exact signed_tie s s' hx0 (by positivity) (by positivity) ((hnear M k hM).2 ht') hne heq2

set_option exponentiation.threshold 300 in
/-- **`roundRn` overflows to `±inf` from `2^128 - 2^103`** (`2^277 - 2^252` units), half an ulp past the largest finite
word, the tie there included. -/
theorem roundRn_inf (s : Bool) {n d : ℕ} (hd : 0 < d) (h : 2 ^ 277 - 2 ^ 252 ≤ (n : ℚ) / d) :
    roundRn s n d = infWord s := by
  obtain ⟨j, q, hj, hlt, hq, heq⟩ := roundRn_eq s n hd
  have hd' : (0 : ℚ) < d := by exact_mod_cast hd
  rw [← div_div] at hq
  have hy : (n : ℚ) / d < 2 ^ (j + 24) := qdiv_lt hd' (by exact_mod_cast hlt)
  have hj253 : 253 ≤ j := by
    by_contra hj'
    have : (2 : ℚ) ^ (j + 24) ≤ 2 ^ 276 := by exact_mod_cast Nat.pow_le_pow_right (by norm_num) (by omega)
    have : (2 : ℚ) ^ 276 < 2 ^ 277 - 2 ^ 252 := by norm_num
    linarith
  have hw : j * 2 ^ 23 + q ≥ 0x7F800000 := by
    rcases Nat.lt_or_ge j 254 with hj1 | hj1
    · obtain rfl : j = 253 := by omega
      have : 2 ^ 24 ≤ q := hq.ge_even (c := 2 ^ 24) (by
          push_cast
          exact qle_div (by positivity) (by linarith)) (by norm_num)
      omega
    · have h1 : (2 : ℚ) ^ (j + 23) ≤ n / d := qle_div hd' (by exact_mod_cast hj.resolve_left (by omega))
      have : 2 ^ 23 ≤ q := hq.ge (c := 2 ^ 23) (by
          push_cast
          have : (2 : ℚ) ^ 23 ≤ n / d / 2 ^ j :=
            qle_div (by positivity) (by rw [← pow_add, Nat.add_comm 23 j]; exact h1)
          linarith)
      omega
  rw [heq, ite_eq_left hw]; rfl

end Verity.Fp32.Round
