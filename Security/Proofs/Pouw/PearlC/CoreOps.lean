import Proofs.Pouw.PearlC.RowStats
import Proofs.Pouw.PearlC.FreezePPure
import Proofs.Core.Fp32.Round
import Proofs.Core.Prims.Round

/-!
# Core's primitives at `pc8`'s and `pc4`'s rows (proofs)

`Pc8OpsAgree` and `Pc4OpsAgree` name what the row units need of their primitives. At core's instance `coreOps` they
follow from core's definitions (`Verity.Fp32`, `Verity.Scalar`, `Verity.Prims`) and nothing else (`pc8OpsAgree_coreOps`,
`pc4OpsAgree_coreOps`).

From the definitions alone: `F32Neg` flips and `F32Fabs` clears the sign bit, `SelectI32` reads its condition's low bit,
the BF16 narrowing of a finite word is `rnBf16?`'s rounding, the E4M3 widening of every code is its value's word (by
evaluation), and the FMA of `−0` by `−0` is that of `+0` by `+0` (both products decode to `+0`). `decode` reads a
finite word as its sign and its magnitude in units of `2⁻¹⁴⁹` (`decode_fin`), so a zero product of opposite signs is
`−0`, a finite word over `+inf` is the zero of its sign, and a product or quotient with a non-finite first operand is a
NaN or an infinity. `order` is the value order of finite words (`order_cmp`), which gives both `F32Fmaxf` rows; and the
`E2M1_UP` boundaries are the words of `e2m1Mids`, so `F32ToE2m1Sat`'s count of the boundaries a magnitude passes is
`e2m1Mag`'s count of the midpoints its value passes (`count_eq`).

The rest is core's rounding and E4M3 cast against the protocol's: `roundRn` of a positive magnitude (`n / d` units of
`2⁻¹⁴⁹`) is `f32Word` of its value (`roundRn_f32Word`), and `F32ToE4m3Sat` of a word is `e4m3Word`'s code
(`cast_e4m3Word`). Core's package gives each as its integer round's grid and nearest-even rounding
(`Verity.Fp32.Round.roundRn_eq`, `Verity.Prims.Round.f32ToE4m3Sat_fin`), which here read as `binade` and `roundEven` of
the value (`roundEven_eq`). With the first, the multiply, divide and FMA rows follow, the FMA's `roundSum` and its
signed zeros included (`roundSum_eq`).
-/

namespace Pouw.PearlC.CoreOps

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4Dev Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit Pouw.PearlC.PeelExact

/-! ## Bit fields -/

theorem and_ff (u : ℕ) : u &&& 0xFF = u % 256 := Nat.and_two_pow_sub_one_eq_mod u 8
theorem and_ffff (u : ℕ) : u &&& 0xFFFF = u % 65536 := Nat.and_two_pow_sub_one_eq_mod u 16
theorem and_7fffff (u : ℕ) : u &&& 0x7FFFFF = u % 2 ^ 23 := Nat.and_two_pow_sub_one_eq_mod u 23
theorem and_7fffffff (u : ℕ) : u &&& 0x7FFFFFFF = u % 2 ^ 31 := Nat.and_two_pow_sub_one_eq_mod u 31
theorem and_ffffffff (u : ℕ) : u &&& 0xFFFFFFFF = u % 2 ^ 32 := Nat.and_two_pow_sub_one_eq_mod u 32

theorem and_field (u c k : ℕ) (hc : c % 2 ^ k = 0) : u &&& c = (u >>> k &&& c >>> k) * 2 ^ k := by
  have h1 : (u &&& c) % 2 ^ k = 0 := by rw [Nat.and_mod_two_pow, hc, Nat.and_zero]
  have h3 := Nat.div_add_mod (u &&& c) (2 ^ k)
  rw [h1, Nat.add_zero, ← Nat.shiftRight_eq_div_pow, Nat.shiftRight_and_distrib] at h3
  rw [Nat.mul_comm]; exact h3.symm

theorem and_exp (u : ℕ) : u &&& 0x7F800000 = (u >>> 23) % 256 * 2 ^ 23 := by
  rw [and_field u 0x7F800000 23 (by decide), show (0x7F800000 : ℕ) >>> 23 = 0xFF by decide, and_ff]

theorem and_bit31 (u : ℕ) : u &&& 0x80000000 = (u >>> 31) % 2 * 2 ^ 31 := by
  rw [and_field u 0x80000000 31 (by decide), show (0x80000000 : ℕ) >>> 31 = 1 by decide, Nat.and_one_is_mod]

theorem or_hidden {m : ℕ} (hm : m < 2 ^ 23) : m ||| 0x800000 = 2 ^ 23 + m := by
  have h := Nat.shiftLeft_add_eq_or_of_lt hm 1
  rw [Nat.one_shiftLeft] at h
  rw [Nat.or_comm]; exact h.symm

theorem or8 {P : Prop} [Decidable P] {C : ℕ} (hC : C < 8) :
    (if P then 8 else 0) ||| C = (if P then 8 else 0) + C := by
  split_ifs
  · have h := Nat.shiftLeft_add_eq_or_of_lt (show C < 2 ^ 3 by omega) 1
    rw [Nat.one_shiftLeft] at h; exact h.symm
  · rw [Nat.zero_or, Nat.zero_add]

theorem lt31 {a : ℕ} (ha : a < 2 ^ 32) (hs : signBit a = false) : a < 2 ^ 31 := by
  unfold signBit at hs; rw [beq_eq_false_iff_ne] at hs; omega

theorem mag_lt {u : ℕ} (he : (u >>> 23) % 256 ≠ 255) : u % 2 ^ 31 < 0x7F800000 := by omega

theorem mag_inf {u : ℕ} (he : (u >>> 23) % 256 = 255) (hm : u % 2 ^ 23 = 0) : u % 2 ^ 31 = 0x7F800000 := by omega

theorem mag_le {u : ℕ} (h : (u >>> 23) % 256 = 255 → u % 2 ^ 23 = 0) : u % 2 ^ 31 ≤ 0x7F800000 := by omega

/-! ## Words as values -/

theorem wordQ_zero : wordQ 0 = some 0 := by decide +kernel

theorem wordQ_inf : wordQ 0x7F800000 = none := by decide +kernel

theorem wordQ_nanPtx : wordQ (Verity.Fp32.F32_NAN_PTX % 2 ^ 32) = none := by decide +kernel

theorem wordQ_infWord (s : Bool) : wordQ (Verity.Fp32.infWord s % 2 ^ 32) = none := by
  cases s <;> decide +kernel

theorem f32Word_zero : f32Word 0 = 0 := by simp [f32Word]

theorem ne_negZero {w : ℕ} {v : ℚ} (h : wordQ w = some v) (hv : v ≠ 0) : w ≠ 0x80000000 := by
  rintro rfl; rw [PeelExactProofs.wordQ_negZero] at h; exact hv (Option.some.inj h).symm

theorem sgn_ite (w : ℕ) :
    (if (w >>> 31) % 2 = 1 then (-1 : ℚ) else 1) = (if signBit w = true then -1 else 1) := by
  unfold signBit; rcases Nat.mod_two_eq_zero_or_one (w >>> 31) with h | h <;> rw [h] <;> simp

theorem sgn_xor (a b : Bool) :
    (if (a ^^ b) = true then (-1 : ℚ) else 1) = (if a = true then -1 else 1) * (if b = true then -1 else 1) := by
  cases a <;> cases b <;> norm_num

theorem sgn_not (s : Bool) : (if (!s) = true then (-1 : ℚ) else 1) = -(if s = true then -1 else 1) := by
  cases s <;> norm_num

theorem val_eq_zero (s : Bool) (M : ℕ) : (if s = true then (-1 : ℚ) else 1) * ((M : ℚ) / 2 ^ 149) = 0 ↔ M = 0 := by
  cases s <;> simp

theorem val_ne_zero (s : Bool) {n d : ℕ} (hn : 0 < n) (hd : 0 < d) :
    (if s = true then (-1 : ℚ) else 1) * ((n : ℚ) / d / 2 ^ 149) ≠ 0 := by
  have hn' : (n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
  have hd' : (d : ℚ) ≠ 0 := by exact_mod_cast hd.ne'
  cases s <;> simp [hn', hd']

theorem prod_eq_zero {sa sb : Bool} {ma mb : ℕ} :
    (if sa = true then (-1 : ℚ) else 1) * ((ma : ℚ) / 2 ^ 149) *
      ((if sb = true then (-1 : ℚ) else 1) * ((mb : ℚ) / 2 ^ 149)) = 0 ↔ ma * mb = 0 := by
  rw [mul_eq_zero, val_eq_zero, val_eq_zero, Nat.mul_eq_zero]

theorem xor_of_ne {p q : Bool} (h : p ≠ q) : (p ^^ q) = true := by
  revert h; cases p <;> cases q <;> decide

theorem val_fields (E m : ℕ) :
    ((((if E = 0 then 0 else 2 ^ 23) + m : ℕ) : ℚ) * (2 : ℚ) ^ (((max E 1 : ℕ) : ℤ) - 150)) =
      ((((if E = 0 then 0 else 2 ^ 23) + m) * 2 ^ (max E 1 - 1) : ℕ) : ℚ) / 2 ^ 149 := by
  have h1 : (((max E 1 : ℕ) : ℤ) - 150) = ((max E 1 - 1 : ℕ) : ℤ) - ((149 : ℕ) : ℤ) := by omega
  rw [h1, zpow_sub₀ two_ne_zero, zpow_natCast, zpow_natCast]; push_cast; ring

/-- A finite word's value is its magnitude's, signed. -/
theorem wordQ_abs {u : ℕ} {x : ℚ} (hx : wordQ u = some x) :
    wordQ (u % 2 ^ 31) = some (if (u >>> 31) % 2 = 1 then -x else x) := by
  have he := wordQ_exp hx
  have h1 : ((u % 2 ^ 31) >>> 23) % 256 = (u >>> 23) % 256 := by
    simp only [Nat.shiftRight_eq_div_pow]; omega
  have h2 : u % 2 ^ 31 % 2 ^ 23 = u % 2 ^ 23 := Nat.mod_mod_of_dvd u (pow_dvd_pow 2 (by norm_num))
  have h3 : ((u % 2 ^ 31) >>> 31) % 2 = 0 := by simp only [Nat.shiftRight_eq_div_pow]; omega
  have he' : ((u % 2 ^ 31) >>> 23) % 256 ≠ 255 := by rw [h1]; exact he
  rw [PeelExactProofs.wordQ_fields he] at hx
  rw [PeelExactProofs.wordQ_fields he', h1, h2, h3]
  obtain rfl := Option.some.inj hx
  congr 1; rw [ite_eq_right (by norm_num : ¬ (0 : ℕ) = 1), one_mul]; split_ifs <;> ring

theorem mag_facts {u : ℕ} {x : ℚ} (hx : wordQ u = some x) :
    ∃ v : ℚ, wordQ (u % 2 ^ 31) = some v ∧ 0 ≤ v ∧ (v = 0 ↔ u % 2 ^ 31 = 0) ∧
      x = if (u >>> 31) % 2 = 1 then -v else v := by
  have hv := wordQ_abs hx
  have hm : u % 2 ^ 31 < 2 ^ 31 := Nat.mod_lt _ (by norm_num)
  refine ⟨_, hv, wordQ_nonneg hm hv, ⟨fun h0 => ?_, fun h0 => ?_⟩, ?_⟩
  · rw [h0] at hv
    exact FreezePPure.word_eq_of_wordQ (by omega) (by omega) (by norm_num) (by norm_num) hv wordQ_zero
  · rw [h0, wordQ_zero] at hv; exact (Option.some.inj hv).symm
  · split_ifs <;> simp

theorem wordQ_le_iff {a b : ℕ} {x y : ℚ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) (hx : wordQ a = some x)
    (hy : wordQ b = some y) : x ≤ y ↔ a ≤ b := by
  constructor
  · intro hxy
    by_contra hab
    have hyx := wordQ_mono (show b ≤ a by omega) ha hy hx
    have hxy' : x = y := le_antisymm hxy hyx
    subst hxy'
    have := FreezePPure.word_eq_of_wordQ (by omega) (by omega) (by omega) (by omega) hx hy
    omega
  · intro hab; exact wordQ_mono hab hb hx hy

theorem wordQ_lt_iff {a b : ℕ} {x y : ℚ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) (hx : wordQ a = some x)
    (hy : wordQ b = some y) : x < y ↔ a < b := by
  rw [← not_le, ← not_le, wordQ_le_iff hb ha hy hx]

theorem wordQ_eq_iff {a b : ℕ} {x y : ℚ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) (hx : wordQ a = some x)
    (hy : wordQ b = some y) : x = y ↔ a = b := by
  constructor
  · intro hxy; subst hxy
    exact FreezePPure.word_eq_of_wordQ (by omega) (by omega) (by omega) (by omega) hx hy
  · intro hab; subst hab; rw [hx] at hy; exact Option.some.inj hy

/-! ## `decode` -/

theorem decode_sign (u : ℕ) : ((u >>> 31) &&& 1 == 1) = signBit u := by
  unfold signBit; rw [Nat.and_one_is_mod]

theorem decode_fields (u : ℕ) : Verity.Fp32.decode u =
    if (u >>> 23) % 256 = 255 then
      (if u % 2 ^ 23 = 0 then Verity.Fp32.Decoded.inf (signBit u) else Verity.Fp32.Decoded.nan)
    else if (u >>> 23) % 256 = 0 then Verity.Fp32.Decoded.fin (signBit u) (u % 2 ^ 23)
    else Verity.Fp32.Decoded.fin (signBit u) ((u % 2 ^ 23 ||| 0x800000) <<< ((u >>> 23) % 256 - 1)) := by
  unfold Verity.Fp32.decode; rw [and_ff, and_7fffff, decode_sign]

/-- `decode` reads a finite word as its sign and its value in units of `2⁻¹⁴⁹`. -/
theorem decode_fin {u : ℕ} {x : ℚ} (hx : wordQ u = some x) :
    ∃ M : ℕ, Verity.Fp32.decode u = .fin (signBit u) M ∧
      x = (if signBit u = true then (-1 : ℚ) else 1) * ((M : ℚ) / 2 ^ 149) := by
  have he := wordQ_exp hx
  rw [PeelExactProofs.wordQ_fields he] at hx
  obtain rfl := Option.some.inj hx
  have hm : u % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  refine ⟨((if (u >>> 23) % 256 = 0 then 0 else 2 ^ 23) + u % 2 ^ 23) * 2 ^ (max ((u >>> 23) % 256) 1 - 1), ?_, ?_⟩
  · rw [decode_fields, ite_eq_right he]
    by_cases h0 : (u >>> 23) % 256 = 0
    · rw [ite_eq_left h0, ite_eq_left h0, h0]; simp
    · rw [ite_eq_right h0, ite_eq_right h0, Nat.max_eq_left (show 1 ≤ (u >>> 23) % 256 by omega), or_hidden hm,
        Nat.shiftLeft_eq]
  · rw [sgn_ite, val_fields]

theorem decode_nonfin {u : ℕ} (he : (u >>> 23) % 256 = 255) :
    Verity.Fp32.decode u = .nan ∨ ∃ s, Verity.Fp32.decode u = .inf s := by
  rw [decode_fields, ite_eq_left he]
  by_cases hm : u % 2 ^ 23 = 0
  · right; exact ⟨_, ite_eq_left hm⟩
  · left; exact ite_eq_right hm

/-! ## NaN tests -/

theorem isNan_false {u : ℕ} (h : (u >>> 23) % 256 = 255 → u % 2 ^ 23 = 0) : Verity.Scalar.f32IsNan u = false := by
  unfold Verity.Scalar.f32IsNan; rw [and_exp, and_7fffff]
  by_cases he : (u >>> 23) % 256 = 255
  · rw [h he, bne_self_eq_false, Bool.and_false]
  · have hb : ((u >>> 23) % 256 * 2 ^ 23 == 0x7F800000) = false := beq_eq_false_iff_ne.2 (by omega)
    rw [hb, Bool.false_and]

theorem isNan_true {u : ℕ} (he : (u >>> 23) % 256 = 255) (hm : u % 2 ^ 23 ≠ 0) : Verity.Scalar.f32IsNan u = true := by
  unfold Verity.Scalar.f32IsNan; rw [and_exp, and_7fffff, he, bne_iff_ne.2 hm]; decide

theorem isNan_fin {u : ℕ} {x : ℚ} (hx : wordQ u = some x) : Verity.Scalar.f32IsNan u = false :=
  isNan_false fun he => absurd he (wordQ_exp hx)

theorem gpuNan_eq {w : ℕ} (h : Verity.Scalar.f32IsNan w = false) : Verity.Fp32.f32GpuNan w = w := by
  have h' : Verity.Fp32.f32IsNan w = false := h
  unfold Verity.Fp32.f32GpuNan; rw [h']; exact ite_eq_right (by decide)

/-! ## Core's rounding and E4M3 cast are the protocol's -/

/-- Core's rounding to nearest even is `roundEven`. -/
theorem roundEven_eq {x : ℚ} {q : ℕ} (h : Verity.Round.RoundsEven x q) : roundEven x = q := by
  by_cases ht : 2 * |(q : ℚ) - x| = 1
  · have he := h.tie ht
    rcases (abs_eq (by norm_num : (0 : ℚ) ≤ 1 / 2)).1 (by linarith : |(q : ℚ) - x| = 1 / 2) with h1 | h1
    · have hx : x = (((q : ℤ) - 1 : ℤ) : ℚ) + 1 / 2 := by push_cast; linarith
      rw [hx, Freeze.roundEven_half_odd (by omega)]
      omega
    · have hx : x = ((q : ℤ) : ℚ) + 1 / 2 := by push_cast; linarith
      rw [hx, Freeze.roundEven_half_even (by omega)]
  · apply Proofs.roundEven_near
    have h2 : 2 * |(q : ℚ) - x| < 1 := lt_of_le_of_ne h.near ht
    rw [Int.cast_natCast, abs_sub_comm]
    linarith

theorem binade_div_pow {x : ℚ} (hx : 0 < x) (k : ℕ) : binade (x / 2 ^ k) = binade x - k := by
  obtain ⟨h1, h2⟩ := Proofs.binade_spec hx
  have hk : (0 : ℚ) < 2 ^ k := by positivity
  apply Proofs.binade_unique (div_pos hx hk)
  · rw [le_div_iff₀ hk, ← zpow_natCast, ← zpow_add₀ two_ne_zero, sub_add_cancel]; exact h1
  · rw [div_lt_iff₀ hk, ← zpow_natCast, ← zpow_add₀ two_ne_zero, show binade x - k + 1 + k = binade x + 1 by ring]
    exact h2

/-- **`roundRn` is `f32Word`**: the word core rounds `(-1)^s · n / d` units of `2⁻¹⁴⁹` to, `n, d > 0`, is `f32Word` of
that value (`Verity.Fp32.Round.roundRn_eq`: the same binade exponent, clamped at the subnormals, and the same
significand rounded to nearest even). -/
theorem roundRn_f32Word (s : Bool) {n d : ℕ} (hn : 0 < n) (hd : 0 < d) :
    Verity.Fp32.roundRn s n d = f32Word ((if s = true then -1 else 1) * ((n : ℚ) / d / 2 ^ 149)) := by
  obtain ⟨j, q, hj, hlt, hq, heq⟩ := Verity.Fp32.Round.roundRn_eq s n hd
  have hd' : (0 : ℚ) < d := by exact_mod_cast hd
  have hnd : (0 : ℚ) < n / d := div_pos (by exact_mod_cast hn) hd'
  have ha : (0 : ℚ) < n / d / 2 ^ 149 := div_pos hnd (by norm_num)
  obtain ⟨hb1, hb2⟩ := Proofs.binade_spec hnd
  have hup : (n : ℚ) / d < 2 ^ ((j + 24 : ℕ) : ℤ) := by
    rw [zpow_natCast, div_lt_iff₀ hd']; exact_mod_cast hlt
  have hbu : binade ((n : ℚ) / d) < ((j + 24 : ℕ) : ℤ) := by
    by_contra h
    have := zpow_le_zpow_right₀ (one_le_two : (1 : ℚ) ≤ 2) (not_lt.1 h)
    linarith
  have hbl : j = 0 ∨ ((j + 23 : ℕ) : ℤ) ≤ binade ((n : ℚ) / d) := by
    refine hj.imp id fun hj => ?_
    have hlo : (2 : ℚ) ^ ((j + 23 : ℕ) : ℤ) ≤ n / d := by
      rw [zpow_natCast, le_div_iff₀ hd']; exact_mod_cast hj
    by_contra h
    have := zpow_le_zpow_right₀ (one_le_two : (1 : ℚ) ≤ 2)
      (show binade ((n : ℚ) / d) + 1 ≤ ((j + 23 : ℕ) : ℤ) by omega)
    linarith
  have hE : max (binade ((n : ℚ) / d / 2 ^ 149)) (-126) = (j : ℤ) - 126 := by
    rw [binade_div_pow hnd 149]; omega
  have hP : (0 : ℚ) < d * 2 ^ j := mul_pos hd' (by positivity)
  have hdiv : (n : ℚ) / d / 2 ^ 149 / 2 ^ ((j : ℤ) - 126 - 23) = n / (d * 2 ^ j) := by
    rw [show (j : ℤ) - 126 - 23 = (j : ℤ) - (149 : ℕ) by omega, zpow_sub₀ two_ne_zero, zpow_natCast, zpow_natCast]
    field_simp
  have hq24 : q ≤ 2 ^ 24 := hq.le (by
    rw [div_lt_iff₀ hP]
    have : ((n : ℕ) : ℚ) < ((2 ^ (j + 24) * d : ℕ) : ℚ) := by exact_mod_cast hlt
    push_cast at this ⊢; rw [pow_add] at this; linarith)
  have hq23 : j = 0 ∨ 2 ^ 23 ≤ q := hj.imp id fun hj => hq.ge (by
    rw [lt_div_iff₀ hP]
    have : ((2 ^ (j + 23) * d : ℕ) : ℚ) ≤ ((n : ℕ) : ℚ) := by exact_mod_cast hj
    push_cast at this ⊢; rw [pow_add] at this; linarith)
  have hr : (roundEven ((n : ℚ) / d / 2 ^ 149 / 2 ^ ((j : ℤ) - 126 - 23))).toNat = q := by
    rw [hdiv, roundEven_eq hq, Int.toNat_natCast]
  have habs : |(if s = true then (-1 : ℚ) else 1) * ((n : ℚ) / d / 2 ^ 149)| = (n : ℚ) / d / 2 ^ 149 := by
    cases s
    · rw [ite_eq_right Bool.false_ne_true, one_mul, abs_of_pos ha]
    · rw [ite_eq_left rfl, neg_one_mul, abs_neg, abs_of_pos ha]
  have hneg : ((if s = true then (-1 : ℚ) else 1) * ((n : ℚ) / d / 2 ^ 149) < 0) = (s = true) := by
    apply propext
    cases s
    · rw [ite_eq_right Bool.false_ne_true, one_mul]
      exact ⟨fun h => absurd h (not_lt.2 ha.le), fun h => absurd h Bool.false_ne_true⟩
    · rw [ite_eq_left rfl, neg_one_mul, neg_lt_zero]
      exact ⟨fun _ => rfl, fun _ => ha⟩
  rw [heq, Verity.Fp32.Round.zeroWord_or s (by split_ifs <;> omega)]
  simp only [f32Word, habs, hneg, ha.ne', ↓reduceIte, hE, hr]
  split_ifs <;> omega

/-- `e4m3Mag` of `M > 0` units of `2⁻¹⁴⁹` below 464: `8 · (max(⌊log₂ M⌋, 143) - 143) + q`, `q` the magnitude on the
grid `2^(max(⌊log₂ M⌋, 143) - 3)` units rounded to nearest even. -/
theorem e4m3Mag_units {M q : ℕ} (hM : 0 < M) (hlt : M < 464 * 2 ^ 149)
    (hq : Verity.Round.RoundsEven ((M : ℚ) / 2 ^ (max M.log2 143 - 3)) q) :
    e4m3Mag ((M : ℚ) / 2 ^ 149) = 8 * (max M.log2 143 - 143) + q := by
  have ht : (M : ℚ) / 2 ^ 149 = (M : ℚ) * 2 ^ (-149 : ℤ) := by rw [zpow_neg, zpow_ofNat, div_eq_mul_inv]
  have h0 : (M : ℚ) / 2 ^ 149 ≠ 0 := (div_pos (by exact_mod_cast hM) (by norm_num)).ne'
  have h464 : ¬ (464 : ℚ) ≤ (M : ℚ) / 2 ^ 149 := by
    rw [not_le, div_lt_iff₀ (by norm_num)]; exact_mod_cast hlt
  have hb : binade ((M : ℚ) / 2 ^ 149) = (M.log2 : ℤ) - 149 := by
    rw [ht, H1T.Proofs.binade_nat_mul hM.ne']; omega
  have hdiv : (M : ℚ) / 2 ^ 149 / 2 ^ (max ((M.log2 : ℤ) - 149) (-6) - 3) = (M : ℚ) / 2 ^ (max M.log2 143 - 3) := by
    rw [show max ((M.log2 : ℤ) - 149) (-6) - 3 = ((max M.log2 143 - 3 : ℕ) : ℤ) - (149 : ℕ) by omega,
      zpow_sub₀ two_ne_zero, zpow_natCast, zpow_natCast]
    field_simp
  unfold e4m3Mag
  rw [ite_eq_right h0, ite_eq_right h464]
  dsimp only
  rw [hb, hdiv, roundEven_eq hq]
  omega

theorem sign_ite (u : ℕ) : (if signBit u = true then 0x80 else 0) = (if (u >>> 31) % 2 = 1 then 128 else 0) := by
  unfold signBit; rcases Nat.mod_two_eq_zero_or_one (u >>> 31) with h | h <;> rw [h] <;> decide

/-- **`F32ToE4m3Sat` is `e4m3Word`**: core's cast of a 32-bit word is the protocol's code (`Verity.Prims.Round`: NaN to
`0x7F`, `±inf` and every magnitude from 464 to `±0x7E`, and below it the code of the magnitude rounded to nearest even
on the E4M3 grid). -/
theorem cast_e4m3Word {u : ℕ} (hu : u < 2 ^ 32) : Verity.Prims.f32ToE4m3Sat u = (e4m3Word u).val := by
  have hS' : (if signBit u = true then 0x80 else 0) = 0 ∨ (if signBit u = true then 0x80 else 0) = 0x80 := by
    cases signBit u; exacts [.inl rfl, .inr rfl]
  have hT := sign_ite u
  cases hw : wordQ u with
  | none =>
    have he := wordQ_none_exp hw
    have hE : (e4m3Word u).val = if u % 2 ^ 23 = 0 then ((if (u >>> 31) % 2 = 1 then 128 else 0) + 126) % 256
        else 127 := by
      unfold e4m3Word; rw [hw]; dsimp only; split_ifs <;> rfl
    rw [hE]
    rcases decode_nonfin he with hn | ⟨s, hs⟩
    · have hm : u % 2 ^ 23 ≠ 0 := fun hm => by
        rw [decode_fields, ite_eq_left he, ite_eq_left hm] at hn; cases hn
      rw [Verity.Prims.Round.f32ToE4m3Sat_nan hu hn, ite_eq_right hm]
      rfl
    · have hm : u % 2 ^ 23 = 0 := by
        by_contra hm; rw [decode_fields, ite_eq_left he, ite_eq_right hm] at hs; cases hs
      have hs' : s = signBit u := by
        rw [decode_fields, ite_eq_left he, ite_eq_left hm] at hs; exact (Verity.Fp32.Decoded.inf.inj hs).symm
      subst hs'
      rw [Verity.Prims.Round.f32ToE4m3Sat_inf hu hs, Verity.Prims.Round.or_small hS' (by norm_num), ite_eq_left hm]
      rcases hS' with h | h <;> omega
  | some x =>
    obtain ⟨M, hd, rfl⟩ := decode_fin hw
    have habs : |(if signBit u = true then (-1 : ℚ) else 1) * ((M : ℚ) / 2 ^ 149)| = (M : ℚ) / 2 ^ 149 := by
      cases signBit u
      · rw [ite_eq_right Bool.false_ne_true, one_mul, abs_of_nonneg (by positivity)]
      · rw [ite_eq_left rfl, neg_one_mul, abs_neg, abs_of_nonneg (by positivity)]
    have hE : (e4m3Word u).val = ((if (u >>> 31) % 2 = 1 then 128 else 0) + e4m3Mag ((M : ℚ) / 2 ^ 149)) % 256 := by
      unfold e4m3Word; rw [hw]; dsimp only; rw [habs]
    rw [hE]
    rcases Nat.eq_zero_or_pos M with rfl | hM
    · rw [Verity.Prims.Round.cast_zero hu hd, Nat.cast_zero, zero_div, show e4m3Mag 0 = 0 by simp [e4m3Mag]]
      rcases hS' with h | h <;> omega
    · by_cases hsat : M < 464 * 2 ^ 149
      · obtain ⟨q, hq, -, -, hv, he⟩ := Verity.Prims.Round.f32ToE4m3Sat_fin hu hd hsat
        rw [he, e4m3Mag_units hM hsat hq]
        rcases hS' with h | h <;> omega
      · have he := Verity.Prims.Round.f32ToE4m3Sat_sat hu hd (by omega)
        have h464 : (464 : ℚ) ≤ (M : ℚ) / 2 ^ 149 := by
          rw [le_div_iff₀ (by norm_num)]; exact_mod_cast (show 464 * 2 ^ 149 ≤ M by omega)
        have h126 : e4m3Mag ((M : ℚ) / 2 ^ 149) = 126 := by
          unfold e4m3Mag
          rw [ite_eq_right (div_pos (by exact_mod_cast hM) (by norm_num)).ne', ite_eq_left h464]
        rw [he, Verity.Prims.Round.or_small hS' (by norm_num), h126]
        rcases hS' with h | h <;> omega

theorem rnWord_of_ne {v : ℚ} {b : Bool} (hv : v ≠ 0) : rnWord v b = f32Word v := by
  unfold rnWord; exact ite_eq_right hv

theorem rnWord_congr {v v' : ℚ} {b b' : Bool} (hv : v = v') (hb : v = 0 → b = b') : rnWord v b = rnWord v' b' := by
  rw [← hv]
  by_cases h0 : v = 0
  · rw [hb h0]
  · rw [rnWord_of_ne h0, rnWord_of_ne h0]

theorem rnWord_f32 {v : ℚ} {b : Bool} (hb : v = 0 → b = false) : rnWord v b = f32Word v := by
  by_cases h0 : v = 0
  · rw [hb h0, h0]; simp [rnWord, f32Word]
  · exact rnWord_of_ne h0

theorem rnWord_cancel (s : Bool) (n d : ℕ) (b : Bool) (hb : b = false) :
    rnWord ((if s = true then -1 else 1) * ((n : ℚ) / d / 2 ^ 149) +
      (if (!s) = true then -1 else 1) * ((n : ℚ) / d / 2 ^ 149)) b = 0 := by
  have hv : (if s = true then (-1 : ℚ) else 1) * ((n : ℚ) / d / 2 ^ 149) +
      (if (!s) = true then -1 else 1) * ((n : ℚ) / d / 2 ^ 149) = 0 := by rw [sgn_not]; ring
  rw [hb]; unfold rnWord; rw [ite_eq_left hv]; rfl

theorem rnWord_round (s : Bool) (n : ℕ) {d : ℕ} (hd : 0 < d) :
    Verity.Fp32.roundRn s n d = rnWord ((if s = true then -1 else 1) * ((n : ℚ) / d / 2 ^ 149)) s := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [Verity.Fp32.Round.roundRn_zero s hd]; cases s <;> simp [rnWord, Verity.Fp32.zeroWord]
  · rw [roundRn_f32Word s hn hd, rnWord_of_ne (val_ne_zero s hn hd)]

/-- `roundSum` is one rounding of the exact sum, an exact zero being `−0` only when both addends are zeros of negative
sign. -/
theorem roundSum_eq (s₁ s₂ : Bool) (n₁ n₂ : ℕ) {d : ℕ} (hd : 0 < d) :
    Verity.Fp32.roundSum s₁ n₁ s₂ n₂ d = rnWord ((if s₁ = true then -1 else 1) * ((n₁ : ℚ) / d / 2 ^ 149) +
      (if s₂ = true then -1 else 1) * ((n₂ : ℚ) / d / 2 ^ 149)) (decide (n₁ = 0) && decide (n₂ = 0) && s₁ && s₂) := by
  unfold Verity.Fp32.roundSum
  by_cases hs : s₁ = s₂
  · rw [ite_eq_left hs, ← hs, rnWord_round _ _ hd]
    apply rnWord_congr
    · push_cast; ring
    · intro h0
      rcases Nat.eq_zero_or_pos (n₁ + n₂) with h12 | h12
      · have h1 : n₁ = 0 := by omega
        have h2 : n₂ = 0 := by omega
        subst h1; subst h2; cases s₁ <;> decide
      · exact absurd h0 (val_ne_zero s₁ h12 hd)
  · rw [ite_eq_right hs]
    have hs2 : s₂ = !s₁ := by revert hs; cases s₁ <;> cases s₂ <;> decide
    subst hs2
    by_cases h12 : n₁ > n₂
    · rw [ite_eq_left h12, rnWord_round _ _ hd]
      apply rnWord_congr
      · rw [Nat.cast_sub h12.le, sgn_not]; ring
      · intro h0; exact absurd h0 (val_ne_zero s₁ (by omega) hd)
    · rw [ite_eq_right h12]
      by_cases h21 : n₂ > n₁
      · rw [ite_eq_left h21, rnWord_round _ _ hd]
        apply rnWord_congr
        · rw [Nat.cast_sub h21.le, sgn_not]; ring
        · intro h0; exact absurd h0 (val_ne_zero _ (by omega) hd)
      · rw [ite_eq_right h21]
        have he : n₁ = n₂ := by omega
        subst he; symm; apply rnWord_cancel; cases s₁ <;> simp

/-! ## Core's arithmetic on decoded words -/

theorem mulV2_eq {a b : ℕ} {sa sb : Bool} {ma mb : ℕ} (ha : Verity.Fp32.decode a = .fin sa ma)
    (hb : Verity.Fp32.decode b = .fin sb mb) :
    Verity.Fp32.f32MulV2 a b = Verity.Fp32.roundRn (sa ^^ sb) (ma * mb) (1 <<< 149) := by
  unfold Verity.Fp32.f32MulV2; rw [ha, hb]; rfl

theorem divV2_eq {a b : ℕ} {sa sb : Bool} {ma mb : ℕ} (ha : Verity.Fp32.decode a = .fin sa ma)
    (hb : Verity.Fp32.decode b = .fin sb mb) (hmb : mb ≠ 0) :
    Verity.Fp32.f32DivV2 a b = Verity.Fp32.roundRn (sa ^^ sb) (ma <<< 149) mb := by
  unfold Verity.Fp32.f32DivV2; rw [ha, hb]; exact ite_eq_right hmb

theorem divInf_eq {a : ℕ} {sa : Bool} {ma : ℕ} (ha : Verity.Fp32.decode a = .fin sa ma) :
    Verity.Fp32.f32DivV2 a 0x7F800000 = Verity.Fp32.zeroWord sa := by
  have hb : Verity.Fp32.decode 0x7F800000 = .inf false := rfl
  unfold Verity.Fp32.f32DivV2; rw [ha, hb]; exact congrArg Verity.Fp32.zeroWord (Bool.xor_false sa)

theorem fmaBits_eq {a b c : ℕ} {sa sb sc : Bool} {ma mb mc : ℕ} (ha : Verity.Fp32.decode a = .fin sa ma)
    (hb : Verity.Fp32.decode b = .fin sb mb) (hc : Verity.Fp32.decode c = .fin sc mc) :
    Verity.Fp32.f32FmaBits a b c = Verity.Fp32.roundSum (sa ^^ sb) (ma * mb) sc (mc <<< 149) (1 <<< 149) := by
  unfold Verity.Fp32.f32FmaBits; rw [ha, hb, hc]; rfl

theorem fmaBits_congr {a b a' b' c : ℕ}
    (h : (Verity.Fp32.decode a).mul (Verity.Fp32.decode b) = (Verity.Fp32.decode a').mul (Verity.Fp32.decode b')) :
    Verity.Fp32.f32FmaBits a b c = Verity.Fp32.f32FmaBits a' b' c := by
  unfold Verity.Fp32.f32FmaBits; rw [h]

theorem mul_inf_fin {s sb : Bool} {mb : ℕ} (hm : mb ≠ 0) :
    (Verity.Fp32.Decoded.inf s).mul (.fin sb mb) = .inf (s ^^ sb) := by
  show (if mb = 0 then Verity.Fp32.Decoded.nan else Verity.Fp32.Decoded.inf (s ^^ sb)) = _
  exact ite_eq_right hm

theorem gpu_mod_eq {v v' : ℚ} {b b' : Bool} (hv : v = v') (hb : v' = 0 → b = b')
    (hn : (rnWord v' b' >>> 23) % 256 = 255 → rnWord v' b' % 2 ^ 23 = 0) :
    Verity.Fp32.f32GpuNan (rnWord v b) % 2 ^ 32 = rnWord v' b' := by
  rw [rnWord_congr hv (fun h0 => hb (hv.symm.trans h0)), gpuNan_eq (isNan_false hn),
    Nat.mod_eq_of_lt (WholeCircuit.rnWord_lt _ _)]

theorem flag_eq {p q p' q' : Prop} [Decidable p] [Decidable q] [Decidable p'] [Decidable q'] (h1 : p ↔ p')
    (h2 : q ↔ q') (s₁ s₂ s₃ : Bool) :
    (decide p && decide q && (s₁ ^^ s₂) && s₃) = (decide p' && decide q' && (s₁ != s₂) && s₃) := by
  rw [decide_eq_decide.2 h1, decide_eq_decide.2 h2]

theorem div_val (sa sb : Bool) (ma mb : ℕ) :
    (if (sa ^^ sb) = true then (-1 : ℚ) else 1) * (((ma <<< 149 : ℕ) : ℚ) / mb / 2 ^ 149) =
      (if sa = true then (-1 : ℚ) else 1) * ((ma : ℚ) / 2 ^ 149) /
        ((if sb = true then (-1 : ℚ) else 1) * ((mb : ℚ) / 2 ^ 149)) := by
  have hB : ∀ t : ℚ, t / (if sb = true then (-1 : ℚ) else 1) = t * (if sb = true then (-1 : ℚ) else 1) := by
    intro t; split_ifs <;> ring
  rw [Nat.shiftLeft_eq, sgn_xor, div_mul_eq_div_div, hB]; push_cast; ring

/-! ## `Pc8OpsAgree` at `coreOps` -/

theorem coreOps_fabs : ∀ w, coreOps.f32Fabs w = w % 2 ^ 31 := fun w => and_7fffffff w

theorem coreOps_neg : ∀ w < 2 ^ 32, coreOps.f32Neg w % 2 ^ 32 = negWord w := by
  intro w hw
  have hlt : w ^^^ 0x80000000 < 2 ^ 32 := Nat.xor_lt_two_pow hw (by norm_num)
  show ((w ^^^ 0x80000000) &&& 0xFFFFFFFF) % 2 ^ 32 = w ^^^ 0x80000000
  rw [and_ffffffff, Nat.mod_mod, Nat.mod_eq_of_lt hlt]

theorem coreOps_sel : ∀ c a b, coreOps.selectI32 c a b = if c % 2 = 1 then a else b := by
  intro c a b
  show (if c &&& 1 ≠ 0 then a else b) = _
  rw [Nat.and_one_is_mod]
  rcases Nat.mod_two_eq_zero_or_one c with h | h <;> simp [h]

theorem coreOps_widenNeg0 : coreOps.e4m3ToF32 0x80 % 2 ^ 32 = 0x80000000 := by decide +kernel

theorem coreOps_widen :
    ∀ c : Code, c.isNaN = false → c.val ≠ 0x80 → coreOps.e4m3ToF32 c.val % 2 ^ 32 = f32Word (codeVal c) := by
  decide +kernel

theorem cls_ne {w : ℕ} (he : (w >>> 23) % 256 ≠ 255) : Verity.TC.Spec.fp32Class w ≠ .nan := by
  simp only [Verity.TC.Spec.fp32Class]; rw [and_ff, ite_eq_right he]; split_ifs <;> decide

theorem ite_none_some {c : Prop} [Decidable c] {o x : ℕ} (h : (if c then none else some o) = some x) : o = x := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · rw [ite_eq_right hc] at h; exact Option.some.inj h

theorem coreOps_bf16 : ∀ w x : ℕ, w < 2 ^ 32 → rnBf16? w = some x → coreOps.f2fpBf16 w % 2 ^ 16 = x := by
  intro w x _ h
  by_cases he : (w >>> 23) % 256 = 255
  · have hf : finite? w = none := by unfold finite?; exact ite_eq_left he
    rw [rnBf16?, hf, Option.bind_none] at h; cases h
  · have hf : finite? w = some w := by unfold finite?; exact ite_eq_right he
    rw [rnBf16?, hf, Option.bind_some] at h
    dsimp only at h
    rw [← ite_none_some h]
    show Verity.TC.Spec.cvtRnBf16F32 w % 2 ^ 16 = _
    unfold Verity.TC.Spec.cvtRnBf16F32
    rw [ite_eq_right (cls_ne he), and_ffff, Nat.and_one_is_mod]
    exact Nat.mod_eq_of_lt (Nat.mod_lt _ (by norm_num))

theorem coreOps_fmaSq0 : ∀ c < 2 ^ 32, coreOps.f32FmaV2 0x80000000 0x80000000 c = coreOps.f32FmaV2 0 0 c := by
  intro c _
  show Verity.Fp32.f32GpuNan (Verity.Fp32.f32FmaBits 0x80000000 0x80000000 c) =
    Verity.Fp32.f32GpuNan (Verity.Fp32.f32FmaBits 0 0 c)
  rw [fmaBits_congr (a := 0x80000000) (b := 0x80000000) (a' := 0) (b' := 0) (c := c) rfl]

theorem coreOps_mul : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x →
    wordQ b = some y → (x * y = 0 → signBit a = signBit b) → coreOps.f32MulV2 a b % 2 ^ 32 = f32Word (x * y) := by
  intro a b x y _ _ hx hy hz
  obtain ⟨ma, hda, rfl⟩ := decode_fin hx
  obtain ⟨mb, hdb, rfl⟩ := decode_fin hy
  show Verity.Fp32.f32MulV2 a b % 2 ^ 32 = _
  rw [mulV2_eq hda hdb, Nat.one_shiftLeft, rnWord_round _ _ (Nat.two_pow_pos 149), rnWord_f32,
    Nat.mod_eq_of_lt (f32Word_lt _)]
  · congr 1; rw [sgn_xor]; push_cast; ring
  · intro h0
    have hm : ma * mb = 0 := by
      by_contra hne; exact val_ne_zero _ (Nat.pos_of_ne_zero hne) (by norm_num) h0
    rw [hz (prod_eq_zero.2 hm), Bool.xor_self]

theorem coreOps_mulZero : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y →
    x * y = 0 → signBit a ≠ signBit b → coreOps.f32MulV2 a b % 2 ^ 32 = 0x80000000 := by
  intro a b x y _ _ hx hy hz hs
  obtain ⟨ma, hda, rfl⟩ := decode_fin hx
  obtain ⟨mb, hdb, rfl⟩ := decode_fin hy
  show Verity.Fp32.f32MulV2 a b % 2 ^ 32 = _
  rw [mulV2_eq hda hdb, prod_eq_zero.1 hz, Nat.one_shiftLeft, Verity.Fp32.Round.roundRn_zero _ (Nat.two_pow_pos 149),
    xor_of_ne hs]; rfl

theorem coreOps_div : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x →
    wordQ b = some y → y ≠ 0 → (x = 0 → signBit a = signBit b) → coreOps.f32DivV2 a b % 2 ^ 32 = f32Word (x / y) := by
  intro a b x y _ _ hx hy hy0 hz
  obtain ⟨ma, hda, rfl⟩ := decode_fin hx
  obtain ⟨mb, hdb, rfl⟩ := decode_fin hy
  have hmb : mb ≠ 0 := fun h0 => hy0 ((val_eq_zero _ _).2 h0)
  show Verity.Fp32.f32DivV2 a b % 2 ^ 32 = _
  rw [divV2_eq hda hdb hmb, rnWord_round _ _ (Nat.pos_of_ne_zero hmb), rnWord_f32, Nat.mod_eq_of_lt (f32Word_lt _)]
  · congr 1; exact div_val _ _ _ _
  · intro h0
    have hma : ma = 0 := by
      by_contra hne
      exact val_ne_zero _ (by rw [Nat.shiftLeft_eq]; exact Nat.mul_pos (Nat.pos_of_ne_zero hne) (by norm_num))
        (Nat.pos_of_ne_zero hmb) h0
    rw [hz ((val_eq_zero _ _).2 hma), Bool.xor_self]

theorem coreOps_divInf : ∀ a < 2 ^ 32, (wordQ a).isSome →
    coreOps.f32DivV2 a 0x7F800000 % 2 ^ 32 = if signBit a then 0x80000000 else 0 := by
  intro a _ ha
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 ha
  obtain ⟨ma, hda, -⟩ := decode_fin hx
  show Verity.Fp32.f32DivV2 a 0x7F800000 % 2 ^ 32 = _
  rw [divInf_eq hda]; cases signBit a <;> decide

theorem coreOps_mulNonfin : ∀ a b : ℕ, a < 2 ^ 32 → wordQ a = none → (wordQ b).isSome →
    wordQ (coreOps.f32MulV2 a b % 2 ^ 32) = none := by
  intro a b _ ha hb
  obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hb
  obtain ⟨mb, hdb, -⟩ := decode_fin hy
  show wordQ (Verity.Fp32.f32MulV2 a b % 2 ^ 32) = none
  unfold Verity.Fp32.f32MulV2; rw [hdb]
  rcases decode_nonfin (wordQ_none_exp ha) with hda | ⟨s, hda⟩
  · rw [hda]; exact wordQ_nanPtx
  · rw [hda]
    by_cases hm : mb = 0
    · subst hm; exact wordQ_nanPtx
    · rw [mul_inf_fin hm]; exact wordQ_infWord _

theorem coreOps_divNonfin : ∀ a b : ℕ, a < 2 ^ 32 → wordQ a = none → (wordQ b).isSome →
    wordQ (coreOps.f32DivV2 a b % 2 ^ 32) = none := by
  intro a b _ ha hb
  obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 hb
  obtain ⟨mb, hdb, -⟩ := decode_fin hy
  show wordQ (Verity.Fp32.f32DivV2 a b % 2 ^ 32) = none
  unfold Verity.Fp32.f32DivV2; rw [hdb]
  rcases decode_nonfin (wordQ_none_exp ha) with hda | ⟨s, hda⟩
  · rw [hda]; exact wordQ_nanPtx
  · rw [hda]; exact wordQ_infWord _

theorem coreOps_fma : ∀ a b c w : ℕ, a < 2 ^ 32 → b < 2 ^ 32 → c < 2 ^ 32 →
    fmaWord? a b c = some w → coreOps.f32FmaV2 a b c % 2 ^ 32 = w := by
  intro a b c w _ _ _ hw
  obtain ⟨x, y, z, hx, hy, hz⟩ := PeelExactProofs.fmaWord_some hw
  unfold fmaWord? at hw; rw [hx, hy, hz] at hw
  dsimp only at hw
  obtain ⟨rfl, v, hv⟩ := PeelExactProofs.finite?_eq_some hw
  obtain ⟨ma, hda, rfl⟩ := decode_fin hx
  obtain ⟨mb, hdb, rfl⟩ := decode_fin hy
  obtain ⟨mc, hdc, rfl⟩ := decode_fin hz
  show Verity.Fp32.f32GpuNan (Verity.Fp32.f32FmaBits a b c) % 2 ^ 32 = _
  rw [fmaBits_eq hda hdb hdc, Nat.one_shiftLeft, roundSum_eq _ _ _ _ (Nat.two_pow_pos 149)]
  apply gpu_mod_eq
  · rw [sgn_xor, Nat.shiftLeft_eq]; push_cast; ring
  · have hsh : mc <<< 149 = 0 ↔ mc = 0 := by rw [Nat.shiftLeft_eq]; omega
    exact fun _ => flag_eq prod_eq_zero.symm (hsh.trans (val_eq_zero _ _).symm) _ _ _
  · exact fun he => absurd he (wordQ_exp hv)

theorem coreOps_fmaOvf : ∀ a b c : ℕ, ∀ x y z : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → c < 2 ^ 32 →
    wordQ a = some x → wordQ b = some y → wordQ c = some z → wordQ (f32Word (x * y + z)) = none →
    coreOps.f32FmaV2 a b c % 2 ^ 32 = f32Word (x * y + z) := by
  intro a b c x y z _ _ _ hx hy hz hn
  have hv0 : x * y + z ≠ 0 := by
    intro h0; rw [h0, f32Word_zero, wordQ_zero] at hn; cases hn
  obtain ⟨ma, hda, rfl⟩ := decode_fin hx
  obtain ⟨mb, hdb, rfl⟩ := decode_fin hy
  obtain ⟨mc, hdc, rfl⟩ := decode_fin hz
  show Verity.Fp32.f32GpuNan (Verity.Fp32.f32FmaBits a b c) % 2 ^ 32 = _
  rw [fmaBits_eq hda hdb hdc, Nat.one_shiftLeft, roundSum_eq _ _ _ _ (Nat.two_pow_pos 149),
    ← rnWord_of_ne (b := false) hv0]
  apply gpu_mod_eq
  · rw [sgn_xor, Nat.shiftLeft_eq]; push_cast; ring
  · exact fun h0 => absurd h0 hv0
  · rw [rnWord_of_ne hv0]; exact f32Word_notNaN _

theorem cast_mod (u : ℕ) : Verity.Prims.f32ToE4m3Sat u = Verity.Prims.f32ToE4m3Sat (u % 2 ^ 32) := by
  unfold Verity.Prims.f32ToE4m3Sat; rw [and_ffffffff, and_ffffffff, Nat.mod_mod]

theorem e4m3Word_congr {u v : ℕ} (h1 : (u >>> 31) % 2 = (v >>> 31) % 2) (h2 : wordQ u = wordQ v)
    (h3 : u % 2 ^ 23 = v % 2 ^ 23) : e4m3Word u = e4m3Word v := by
  simp only [e4m3Word, h1, h2, h3]

theorem e4m3Word_mod (u : ℕ) : e4m3Word (u % 2 ^ 32) = e4m3Word u :=
  e4m3Word_congr (WholeCircuit.shr31_mod32 u) (WholeCircuit.wordQ_mod32 u)
    (Nat.mod_mod_of_dvd u (pow_dvd_pow 2 (by norm_num)))

theorem coreOps_cast : ∀ u, coreOps.f32ToE4m3Sat u = (e4m3Word u).val := by
  intro u
  show Verity.Prims.f32ToE4m3Sat u = _
  rw [cast_mod, cast_e4m3Word (Nat.mod_lt _ (by norm_num)), e4m3Word_mod]

/-! ## `order` and `F32Fmaxf` -/

theorem order_eq {u : ℕ} (hu : u < 2 ^ 32) : Verity.Scalar.order u 32 =
    if (u >>> 31) % 2 = 1 then -((u % 2 ^ 31 : ℕ) : ℤ) else ((u % 2 ^ 31 : ℕ) : ℤ) := by
  show (if (u &&& 0xFFFFFFFF) &&& 0x80000000 ≠ 0 then -(((u &&& 0xFFFFFFFF) &&& 0x7FFFFFFF : ℕ) : ℤ)
    else (((u &&& 0xFFFFFFFF) &&& 0x7FFFFFFF : ℕ) : ℤ)) = _
  rw [and_ffffffff, Nat.mod_eq_of_lt hu, and_bit31, and_7fffffff]
  split_ifs <;> omega

theorem cmp_signs {A B : ℕ} {va vb : ℚ} (h0a : 0 ≤ va) (h0b : 0 ≤ vb) (hza : va = 0 ↔ A = 0)
    (hzb : vb = 0 ↔ B = 0) (hlt : va < vb ↔ A < B) (hgt : vb < va ↔ B < A) (heq : va = vb ↔ A = B)
    (p q : Prop) [Decidable p] [Decidable q] :
    ((if p then -(A : ℤ) else A) < (if q then -(B : ℤ) else B) ↔
        (if p then -va else va) < (if q then -vb else vb)) ∧
      ((if p then -(A : ℤ) else A) = (if q then -(B : ℤ) else B) ↔
        (if p then -va else va) = (if q then -vb else vb)) := by
  by_cases hp : p <;> by_cases hq : q <;> simp only [hp, hq, ↓reduceIte]
  · exact ⟨by rw [neg_lt_neg_iff, neg_lt_neg_iff, hgt]; omega, by rw [neg_inj, neg_inj, heq]; omega⟩
  · refine ⟨⟨fun h => ?_, fun h => ?_⟩, ⟨fun h => ?_, fun h => ?_⟩⟩
    · by_contra hc
      have h1 : va = 0 := by linarith
      have h2 : vb = 0 := by linarith
      have := hza.1 h1; have := hzb.1 h2; omega
    · by_contra hc
      have h1 : A = 0 := by omega
      have h2 : B = 0 := by omega
      have := hza.2 h1; have := hzb.2 h2; linarith
    · have h1 : A = 0 := by omega
      have h2 : B = 0 := by omega
      rw [hza.2 h1, hzb.2 h2, neg_zero]
    · have h1 : va = 0 := by linarith
      have h2 : vb = 0 := by linarith
      have := hza.1 h1; have := hzb.1 h2; omega
  · refine ⟨⟨fun h => ?_, fun h => ?_⟩, ⟨fun h => ?_, fun h => ?_⟩⟩
    · exfalso; omega
    · exfalso; linarith
    · have h1 : A = 0 := by omega
      have h2 : B = 0 := by omega
      rw [hza.2 h1, hzb.2 h2, neg_zero]
    · have h1 : va = 0 := by linarith
      have h2 : vb = 0 := by linarith
      have := hza.1 h1; have := hzb.1 h2; omega
  · exact ⟨by rw [hlt]; omega, by rw [heq]; omega⟩

/-- `order` is the value order of finite words. -/
theorem order_cmp {a b : ℕ} {x y : ℚ} (ha : a < 2 ^ 32) (hb : b < 2 ^ 32) (hx : wordQ a = some x)
    (hy : wordQ b = some y) :
    (Verity.Scalar.order a 32 < Verity.Scalar.order b 32 ↔ x < y) ∧
      (Verity.Scalar.order a 32 = Verity.Scalar.order b 32 ↔ x = y) := by
  obtain ⟨va, hva, h0a, hza, rfl⟩ := mag_facts hx
  obtain ⟨vb, hvb, h0b, hzb, rfl⟩ := mag_facts hy
  have hma : a % 2 ^ 31 < 2 ^ 31 := Nat.mod_lt _ (by norm_num)
  have hmb : b % 2 ^ 31 < 2 ^ 31 := Nat.mod_lt _ (by norm_num)
  rw [order_eq ha, order_eq hb]
  exact cmp_signs h0a h0b hza hzb (wordQ_lt_iff hma hmb hva hvb) (wordQ_lt_iff hmb hma hvb hva)
    (wordQ_eq_iff hma hmb hva hvb) _ _

theorem fmaxf_eq {a b : ℕ} (ha : Verity.Scalar.f32IsNan a = false) (hb : Verity.Scalar.f32IsNan b = false) :
    Verity.Scalar.f32Fmaxf a b =
      if Verity.Scalar.order a 32 = Verity.Scalar.order b 32 then
        (if a &&& 0x7FFFFFFF = 0 ∧ b &&& 0x7FFFFFFF = 0 then (if a = 0 ∨ b = 0 then 0 else 0x80000000) else a)
      else if Verity.Scalar.order a 32 > Verity.Scalar.order b 32 then a else b := by
  unfold Verity.Scalar.f32Fmaxf; rw [ha, hb]; rfl

theorem coreOps_fmaxfAny : ∀ a b : ℕ, ∀ y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ b = some y → y ≠ 0 →
    coreOps.f32Fmaxf a b % 2 ^ 32 = if a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x then a else b := by
  intro a b y ha hb hy hy0
  have hnb := isNan_fin hy
  have hB : -(0x7F800000 : ℤ) < Verity.Scalar.order b 32 ∧ Verity.Scalar.order b 32 < 0x7F800000 := by
    have := mag_lt (wordQ_exp hy)
    rw [order_eq hb]; split_ifs <;> omega
  obtain ⟨vb, -, -, hzb, hyv⟩ := mag_facts hy
  have hB0 : b % 2 ^ 31 ≠ 0 := fun h0 => hy0 (by rw [hyv, hzb.2 h0]; simp)
  show Verity.Scalar.f32Fmaxf a b % 2 ^ 32 = _
  rcases Option.eq_none_or_eq_some (wordQ a) with hq | ⟨x, hq⟩
  · have hea := wordQ_none_exp hq
    by_cases hm : a % 2 ^ 23 = 0
    · have hinf := mag_inf hea hm
      have hka := order_eq ha
      rw [fmaxf_eq (isNan_false fun _ => hm) hnb, and_7fffffff, and_7fffffff]
      by_cases hs : (a >>> 31) % 2 = 1
      · rw [ite_eq_left hs, hinf] at hka
        have hR : ¬ (a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x) := by
          rintro (h | ⟨x, h, -⟩)
          · omega
          · rw [hq] at h; cases h
        rw [ite_eq_right hR]; split_ifs <;> omega
      · rw [ite_eq_right hs, hinf] at hka
        have ha' : a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x := Or.inl (by omega)
        rw [ite_eq_left ha']; split_ifs <;> omega
    · have hL : Verity.Scalar.f32Fmaxf a b = b := by
        unfold Verity.Scalar.f32Fmaxf; rw [isNan_true hea hm, hnb]; rfl
      have hR : ¬ (a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x) := by
        rintro (h | ⟨x, h, -⟩)
        · omega
        · rw [hq] at h; cases h
      rw [hL, ite_eq_right hR, Nat.mod_eq_of_lt hb]
  · obtain ⟨hlt, heq⟩ := order_cmp ha hb hq hy
    obtain ⟨hgt, -⟩ := order_cmp hb ha hy hq
    rw [fmaxf_eq (isNan_fin hq) hnb, and_7fffffff, and_7fffffff]
    rcases lt_trichotomy y x with hyx | hyx | hyx
    · have hR : a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x := Or.inr ⟨x, hq, hyx⟩
      rw [ite_eq_left hR]
      have h1 := hgt.2 hyx
      split_ifs <;> omega
    · have h1 := heq.2 hyx.symm
      rw [← hyx] at hq
      have hab : a = b := FreezePPure.word_eq_of_wordQ ha (ne_negZero hq hy0) hb (ne_negZero hy hy0) hq hy
      have hR : ¬ (a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x) := by
        rintro (h | ⟨x', h, hlt'⟩)
        · rw [h, wordQ_inf] at hq; cases hq
        · rw [hq] at h; obtain rfl := Option.some.inj h; exact lt_irrefl _ hlt'
      rw [ite_eq_right hR]
      split_ifs <;> omega
    · have hR : ¬ (a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x) := by
        rintro (h | ⟨x', h, hlt'⟩)
        · rw [h, wordQ_inf] at hq; cases hq
        · rw [hq] at h; obtain rfl := Option.some.inj h; exact lt_asymm hyx hlt'
      have h1 := hlt.2 hyx
      rw [ite_eq_right hR]
      split_ifs <;> omega

/-! ## `F32ToE2m1Sat` -/

theorem takeWhile_length {α : Type} (p : α → Bool) :
    ∀ l : List α, l.Pairwise (fun a b => p b = true → p a = true) → (l.takeWhile p).length = l.countP p
  | [], _ => rfl
  | a :: l, h => by
    rw [List.pairwise_cons] at h
    by_cases ha : p a = true
    · rw [List.takeWhile_cons_of_pos ha, List.countP_cons_of_pos ha, List.length_cons, takeWhile_length p l h.2]
    · rw [List.takeWhile_cons_of_neg ha, List.countP_cons_of_neg ha, List.length_nil]
      symm; rw [List.countP_eq_zero]
      intro b hb hpb; exact ha (h.1 b hb hpb)

/-- `E2M1_UP`'s boundaries, the words of `e2m1Mids`. -/
def e2m1Bnds : List ℕ := [0x3E800000, 0x3F400000, 0x3FA00000, 0x3FE00000, 0x40200000, 0x40600000, 0x40A00000]

theorem e2m1Up_eq :
    Verity.Prims.E2M1_UP = (List.range 7).map (fun i => (e2m1Bnds.getD i 0, (i + 1) % 2 == 0)) := by
  decide +kernel

theorem e2m1Bnds_wordQ :
    ∀ i < 7, e2m1Bnds.getD i 0 < 2 ^ 31 ∧ wordQ (e2m1Bnds.getD i 0) = some (e2m1Mids.getD i 0) := by
  decide +kernel

theorem e2m1Mids_lt : ∀ i < 7, ∀ j < 7, i < j → e2m1Mids.getD i 0 < e2m1Mids.getD j 0 := by decide +kernel

theorem e2m1Mag_480 : e2m1Mag 480 = 7 := by decide +kernel

theorem e2m1_pairwise (v : ℚ) : (List.range 7).Pairwise (fun i j =>
    (decide (e2m1Mids.getD j 0 < v) || (decide (e2m1Mids.getD j 0 = v) && (j + 1) % 2 == 0)) = true →
      (decide (e2m1Mids.getD i 0 < v) || (decide (e2m1Mids.getD i 0 = v) && (i + 1) % 2 == 0)) = true) := by
  refine List.Pairwise.imp_of_mem ?_ List.pairwise_lt_range
  intro i j hi hj hij hP
  rw [List.mem_range] at hi hj
  have hm := e2m1Mids_lt i hi j hj hij
  simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true, beq_iff_eq] at hP ⊢
  left
  rcases hP with h | ⟨h, -⟩
  · exact lt_trans hm h
  · rw [← h]; exact hm

/-- `F32ToE2m1Sat`'s count of the boundaries a magnitude passes is `e2m1Mag`'s count of the midpoints its value
passes. -/
theorem count_eq {m : ℕ} {v : ℚ} (hm : m < 2 ^ 31) (hv : wordQ m = some v) {q : ℕ × Bool → Bool}
    (hq : ∀ b up, q (b, up) = (decide (m > b) || (up && m == b))) :
    Verity.Prims.E2M1_UP.countP q = e2m1Mag v := by
  rw [e2m1Up_eq, List.countP_map, e2m1Mag, takeWhile_length _ _ (e2m1_pairwise v)]
  apply List.countP_congr
  intro i hi
  rw [List.mem_range] at hi
  obtain ⟨hb, hbv⟩ := e2m1Bnds_wordQ i hi
  have hlt := wordQ_lt_iff hb hm hbv hv
  have heq := wordQ_eq_iff hb hm hbv hv
  simp only [Function.comp_apply, hq, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true, beq_iff_eq, hlt, heq]
  omega

theorem count_inf {q : ℕ × Bool → Bool}
    (hq : ∀ b up, q (b, up) = (decide (0x7F800000 > b) || (up && 0x7F800000 == b))) :
    Verity.Prims.E2M1_UP.countP q = 7 := by
  have : q = fun p => decide (0x7F800000 > p.1) || (p.2 && 0x7F800000 == p.1) := funext fun ⟨b, up⟩ => hq b up
  subst this; decide +kernel

theorem coreOps_e2m1 : ∀ w < 2 ^ 32, ((w >>> 23) % 256 = 255 → w % 2 ^ 23 = 0) →
    coreOps.f32ToE2m1Sat w % 2 ^ 4 = (e2m1Word w).val := by
  intro w hw hnn
  have hmag := mag_le hnn
  have hcnt : ∀ q, Verity.Prims.E2M1_UP.countP q < 8 := fun q => lt_of_le_of_lt List.countP_le_length (by decide)
  show Verity.Prims.f32ToE2m1Sat w % 16 =
    ((if (w >>> 31) % 2 = 1 then 8 else 0) + e2m1Mag |PearlC.wordVal w|) % 16
  simp only [Verity.Prims.f32ToE2m1Sat]
  rw [and_ffffffff, Nat.mod_eq_of_lt hw, and_7fffffff, ite_eq_right (show ¬ (w % 2 ^ 31 > 0x7F800000) by omega)]
  have hs : (if w >>> 31 ≠ 0 then 8 else 0 : ℕ) = (if (w >>> 31) % 2 = 1 then 8 else 0) := by
    split_ifs <;> omega
  rw [hs, or8 (hcnt _)]
  rcases Option.eq_none_or_eq_some (wordQ w) with hq | ⟨x, hq⟩
  · have he := wordQ_none_exp hq
    have hm := mag_inf he (hnn he)
    have hval : |PearlC.wordVal w| = 480 := by
      simp only [PearlC.wordVal, hq]; split_ifs <;> norm_num
    rw [hval, e2m1Mag_480, hm, count_inf]
    intro b up; rfl
  · obtain ⟨v, hv, h0v, -, hxv⟩ := mag_facts hq
    have hval : |PearlC.wordVal w| = v := by
      rw [wordVal_of_wordQ hq, hxv]; split_ifs
      · rw [abs_neg, abs_of_nonneg h0v]
      · exact abs_of_nonneg h0v
    have hm31 : w % 2 ^ 31 < 2 ^ 31 := Nat.mod_lt _ (by norm_num)
    rw [hval, count_eq hm31 hv]
    intro b up; rfl

theorem coreOps_fmaxf : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y →
    signBit a = false → signBit b = false → coreOps.f32Fmaxf a b % 2 ^ 32 = if x < y then b else a := by
  intro a b x y ha hb hx hy hsa _
  obtain ⟨hlt, heq⟩ := order_cmp ha hb hx hy
  obtain ⟨hgt, -⟩ := order_cmp hb ha hy hx
  show Verity.Scalar.f32Fmaxf a b % 2 ^ 32 = _
  rw [fmaxf_eq (isNan_fin hx) (isNan_fin hy), and_7fffffff, and_7fffffff]
  rcases lt_trichotomy x y with hxy | hxy | hxy
  · rw [ite_eq_left hxy]; have h1 := hlt.2 hxy; split_ifs <;> omega
  · have ha31 := lt31 ha hsa
    rw [ite_eq_right (not_lt.2 hxy.ge), ite_eq_left (heq.2 hxy)]
    split_ifs <;> omega
  · rw [ite_eq_right (not_lt.2 hxy.le)]; have h1 := hgt.2 hxy; split_ifs <;> omega

/-! ## The agreements -/

/-- **Core's instance agrees with `pc8`'s rows.** -/
theorem pc8OpsAgree_coreOps : WholeCircuit.Pc8OpsAgree coreOps :=
  { cast := coreOps_cast, widen := coreOps_widen, mul := coreOps_mul, mulZero := coreOps_mulZero,
    fma := coreOps_fma, neg := coreOps_neg, bf16 := coreOps_bf16, widenNeg0 := coreOps_widenNeg0,
    fabs := coreOps_fabs, fmaSq0 := coreOps_fmaSq0, div := coreOps_div, divInf := coreOps_divInf,
    mulNonfin := coreOps_mulNonfin, fmaxfAny := coreOps_fmaxfAny, sel := coreOps_sel, fmaOvf := coreOps_fmaOvf,
    divNonfin := coreOps_divNonfin }

/-- **Core's instance agrees with `pc4`'s rows.** -/
theorem pc4OpsAgree_coreOps : RowCircuit.Pc4OpsAgree coreOps :=
  { pc8OpsAgree_coreOps with
    fmaxf := coreOps_fmaxf, e2m1 := coreOps_e2m1, bf16w := coreOps_bf16w, le := coreOps_le }

end Pouw.PearlC.CoreOps
