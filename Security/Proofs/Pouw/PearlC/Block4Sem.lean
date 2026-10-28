import Proofs.Pouw.PearlC.Row4Words
import Proofs.Pouw.PearlC.TileCheck8Words

/-!
# `pc4`'s blocks in DeviceFp4's terms (proofs)

`Row4Words` proves a `pc4` block's scale byte and codes as word formulas on `RowOps`: the scale byte
`e4m3(max(amax·f32(1/6), 2⁻⁹))` of the block's running `fmaxf` of `fabs` (`scale4W`, `blkMax4`) and each code
`e2m1(d·(1/scale))` (`code4W`). `DeviceFp4.nvf4Block` reads them on values: `max(e4m3(f32(amax·sixth)), 0x01)` of the
largest magnitude and `e2m1Word(f32(D_t·nvf4Recip s))`. Here they agree, under one named hypothesis on `ops`:

* `Pc4OpsAgree ops`: `Pc8OpsAgree`'s agreements (among them `F32Fabs` clears the sign bit and `F32Div` is `f32Word`
  of the quotient by a nonzero divisor), and `F32Fmaxf` of two finite words with the sign bit clear is the one of
  larger value (`a` at equal values), `F32ToE2m1Sat` of a word that is not NaN is `e2m1Word`'s code, `Bf16ToF32`
  of a BF16 word is the word shifted left by 16, and `I32Le` of two words below `2³¹` is their order.

The floors differ in order: the table floors the FP32 value at `2⁻⁹` and then casts, the protocol casts and then
floors the code at `0x01`. On a non-negative value the two agree (`e4m3_floor`), since `e4m3` is at least `0x01` from
`2⁻⁹` up and at most `0x01` below it. On a block of step words (finite, not `−0`) the running max is the fold's
(`blkMax4_word`), the scale byte is `nvf4Scale`'s (`scale4W_sem`) and the codes are `nvf4Block`'s (`code4W_sem`,
`block4_sem`).

A formed word is `nvf4NoiseElem`'s (`form4_sem`): the accumulator is `f32Word(α·x)` and the operands and scale bytes
are `noiseOperand`'s and `noiseScales`'. Every word a firing writes is a step word (`step_stepWord`,
`toF32RZ_stepWord`), so `formRow`'s noise atoms are (`noiseElem_stepWord`). So a row unit's codes and scale bytes are
`formRow`'s at `pc4Sem`'s `α` and `β` (`formRow_table`, `row4A_form`, `row4B_form`), under `Row4OK` (the row's BF16
words widen to step words of its values, `α > 0` and `β` are step words). The unit's row words are `pc4Sem`'s BF16
words or `−0` (`0x8000`) where those are `+0` (`ZeroSign16`), which no output reads: `α` and `β` read a word through
`|·|` and `fma(x, x, ·)` of its widening (`alpha4_zeroSign16`, `beta4_zeroSign16`), and the forming through the
accumulator `α·x`, `±0` either way, which the firing reads only as a value (`nvf4Word_negZero`, `form4_zeroSign16`).

On those codes and scale bytes, RowA's `Ẽ` and `P_A` outputs are `peelFp4`'s `P_A` row, `paRow?` of the line's
`eTilde` at `e4m3(β)` and the chain words against `F_B`'s columns (`row4A_paWords`), and RowB's `P_B` and `−Ẽ`
outputs are its `P_B` row, `pbRow?` of the line's `eTilde`, the chain words against `F_A`'s columns and `G`
(`row4B_pbWords`), wherever the peel's words exist. `PouwETilde_v1` is `eTilde` at a scale byte that is not NaN
(`eTilde_prim`), and `e4m3` of a value never is (`e4m3_notNaN`).
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev
  Pouw.PearlC.PeelExact

/-! ## E4M3 at the floor -/

/-- From `2⁻⁹` up, `e4m3` writes a nonzero magnitude. -/
theorem e4m3Mag_pos {t : ℚ} (ht : 1 / 512 ≤ t) : 1 ≤ e4m3Mag t := by
  have ht0 : 0 < t := lt_of_lt_of_le (by norm_num) ht
  unfold e4m3Mag
  rw [ite_eq_right ht0.ne']
  by_cases h464 : 464 ≤ t
  · rw [ite_eq_left h464]; norm_num
  rw [ite_eq_right h464]
  dsimp only
  obtain ⟨hb1, -⟩ := binade_spec ht0
  have hu : (0 : ℚ) < 2 ^ (max (binade t) (-6) - 3) := two_zpow_pos _
  have hle : (2 : ℚ) ^ (max (binade t) (-6) - 3) ≤ t := by
    rcases le_or_gt (-6) (binade t) with h | h
    · rw [max_eq_left h]
      exact le_trans (zpow_le_zpow_right₀ one_le_two (by omega)) hb1
    · rw [max_eq_right h.le]; norm_num; linarith
  have h1 : (1 : ℚ) ≤ t / 2 ^ (max (binade t) (-6) - 3) := by rw [le_div_iff₀ hu, one_mul]; exact hle
  have hf : (1 : ℤ) ≤ ⌊t / 2 ^ (max (binade t) (-6) - 3)⌋ := Int.le_floor.2 (by exact_mod_cast h1)
  have := floor_le_roundEven (t / 2 ^ (max (binade t) (-6) - 3))
  have h6 : -6 ≤ max (binade t) (-6) := le_max_right _ _
  omega

/-- Below `2⁻⁹`, `e4m3` writes at most the magnitude `1`. -/
theorem e4m3Mag_le_one {t : ℚ} (h0 : 0 ≤ t) (ht : t < 1 / 512) : e4m3Mag t ≤ 1 := by
  rcases h0.lt_or_eq with h0 | rfl
  swap
  · simp [e4m3Mag]
  unfold e4m3Mag
  rw [ite_eq_right h0.ne', ite_eq_right (by intro h; linarith)]
  dsimp only
  obtain ⟨hb1, -⟩ := binade_spec h0
  have hb : binade t < -6 := by
    have : (2 : ℚ) ^ binade t < 2 ^ (-9 : ℤ) := lt_of_le_of_lt hb1 (by norm_num; linarith)
    have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 this
    omega
  rw [max_eq_right hb.le]
  have h1 : t / 2 ^ ((-6 : ℤ) - 3) < ((1 : ℤ) : ℚ) + 1 / 2 := by
    rw [div_lt_iff₀ (by norm_num)]; norm_num; linarith
  have := roundEven_le_of_lt h1
  omega

theorem e4m3Mag_floor : e4m3Mag (1 / 512) = 1 := by
  have hb : binade (1 / 512 : ℚ) = -9 := binade_unique (by norm_num) (by norm_num) (by norm_num)
  unfold e4m3Mag
  rw [ite_eq_right (by norm_num), ite_eq_right (by norm_num)]
  dsimp only
  rw [hb, show max (-9 : ℤ) (-6) = -6 by norm_num,
    show (1 / 512 : ℚ) / 2 ^ ((-6 : ℤ) - 3) = ((1 : ℤ) : ℚ) by norm_num, Pouw.Fp8Atom.Proofs.roundEven_intCast]
  norm_num

theorem e4m3_val {t : ℚ} (ht : 0 ≤ t) : (e4m3 t).val = e4m3Mag t := by
  simp only [e4m3, abs_of_nonneg ht, not_lt.2 ht, ↓reduceIte, Nat.zero_add]
  exact Nat.mod_eq_of_lt (by have := e4m3Mag_le ht; omega)

/-- **The two floors agree on a non-negative value**: flooring the code at `0x01` (DeviceFp4's `nvf4Scale`) is
flooring the value at `2⁻⁹` before the cast (the `pc4` table's). -/
theorem e4m3_floor {v : ℚ} (hv : 0 ≤ v) : max (e4m3 v) 1 = e4m3 (max v (1 / 512)) := by
  rcases lt_or_ge v (1 / 512) with h | h
  · rw [max_eq_right h.le]
    have h1 : e4m3 (1 / 512) = 1 := Fin.ext (by rw [e4m3_val (by norm_num), e4m3Mag_floor]; rfl)
    rw [h1]
    apply max_eq_right
    rw [Fin.le_iff_val_le_val, e4m3_val hv]
    exact e4m3Mag_le_one hv h
  · rw [max_eq_left h]
    apply max_eq_left
    rw [Fin.le_iff_val_le_val, e4m3_val hv]
    exact e4m3Mag_pos h

/-! ## FP32 words -/

theorem wordQ_exp {w : ℕ} {x : ℚ} (h : wordQ w = some x) : (w >>> 23) % 256 ≠ 255 := by
  intro he
  unfold wordQ fp32Value at h
  simp only [he, ↓reduceIte] at h
  split_ifs at h <;> simp [Value.finite?] at h

theorem wordQ_wordVal {w : ℕ} (he : (w >>> 23) % 256 ≠ 255) : wordQ w = some (wordVal w) := by
  obtain ⟨y, hy⟩ : ∃ y, wordQ w = some y := ⟨_, PeelExactProofs.wordQ_fields he⟩
  simp [wordVal, hy]

theorem signBit_of_lt {w : ℕ} (h : w < 2 ^ 31) : signBit w = false := by
  unfold signBit
  rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt h]; rfl

theorem wordQ_nonneg {w : ℕ} {x : ℚ} (hw : w < 2 ^ 31) (h : wordQ w = some x) : 0 ≤ x := by
  have he := wordQ_exp h
  rw [PeelExactProofs.wordQ_fields he] at h
  have hs : (w >>> 31) % 2 = 0 := by rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt hw]
  obtain rfl := Option.some.inj h
  rw [hs]
  simp only [Nat.zero_ne_one, ↓reduceIte, one_mul]
  positivity

theorem wordVal_nonneg {w : ℕ} (hw : w < 2 ^ 31) : 0 ≤ wordVal w := by
  unfold wordVal
  rcases h : wordQ w with _ | x
  · have hs : (w >>> 31) % 2 = 0 := by rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt hw]
    simp [hs]
  · exact wordQ_nonneg hw h

/-- A finite word's value is below `2^128`. -/
theorem wordQ_abs_lt {w : ℕ} {x : ℚ} (h : wordQ w = some x) : |x| < 2 ^ 128 := by
  have he := wordQ_exp h
  rw [PeelExactProofs.wordQ_fields he] at h
  obtain rfl := Option.some.inj h
  have hM : ((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) < 2 ^ 24 := by split_ifs <;> omega
  have hmax : max ((w >>> 23) % 256) 1 ≤ 254 := max_le (by omega) (by norm_num)
  have hk : ((max ((w >>> 23) % 256) 1 : ℕ) : ℤ) - 150 ≤ 104 := by omega
  have hσ : |(if (w >>> 31) % 2 = 1 then (-1 : ℚ) else 1)| = 1 := by split_ifs <;> simp
  generalize (if (w >>> 31) % 2 = 1 then (-1 : ℚ) else 1) = σ at hσ ⊢
  generalize ((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) = M at hM ⊢
  generalize ((max ((w >>> 23) % 256) 1 : ℕ) : ℤ) - 150 = k at hk ⊢
  rw [abs_mul, abs_mul, hσ, one_mul, Nat.abs_cast, abs_of_pos (two_zpow_pos _)]
  calc (M : ℚ) * 2 ^ k ≤ M * 2 ^ (104 : ℤ) :=
        mul_le_mul_of_nonneg_left (zpow_le_zpow_right₀ one_le_two hk) (by positivity)
    _ < 2 ^ 24 * 2 ^ (104 : ℤ) := mul_lt_mul_of_pos_right (by exact_mod_cast hM) (two_zpow_pos _)
    _ = 2 ^ 128 := by norm_num

/-- **`fabs`**: the sign bit cleared, a finite word's value's magnitude. -/
theorem fabs_word {w : ℕ} {x : ℚ} (h : wordQ w = some x) :
    w % 2 ^ 31 < 2 ^ 31 ∧ wordQ (w % 2 ^ 31) = some |x| := by
  have he := wordQ_exp h
  have h1 : ((w % 2 ^ 31) >>> 23) % 256 = (w >>> 23) % 256 := by
    simp only [Nat.shiftRight_eq_div_pow]; omega
  have h2 : ((w % 2 ^ 31) >>> 31) % 2 = 0 := by simp only [Nat.shiftRight_eq_div_pow]; omega
  have h3 : (w % 2 ^ 31) % 2 ^ 23 = w % 2 ^ 23 := by omega
  refine ⟨Nat.mod_lt _ (by norm_num), ?_⟩
  rw [PeelExactProofs.wordQ_fields (h1 ▸ he), h1, h2, h3]
  rw [PeelExactProofs.wordQ_fields he] at h
  obtain rfl := Option.some.inj h
  have hp : (0 : ℚ) ≤ (((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) : ℚ) *
      (2 : ℚ) ^ (((max ((w >>> 23) % 256) 1 : ℕ) : ℤ) - 150) := by positivity
  rw [abs_mul, abs_of_nonneg hp]
  by_cases hs : (w >>> 31) % 2 = 1 <;> simp [hs]

/-- `f32Word` of a non-negative value has the sign bit clear. -/
theorem f32Word_lt31 {y : ℚ} (hy : 0 ≤ y) : f32Word y < 2 ^ 31 := by
  unfold f32Word
  dsimp only
  rw [ite_eq_right (not_lt.2 hy)]
  by_cases h0 : |y| = 0
  · rw [ite_eq_left h0]; norm_num
  rw [ite_eq_right h0]
  have hN := f32Word_N (lt_of_le_of_ne (abs_nonneg y) (Ne.symm h0))
  generalize (roundEven (|y| / 2 ^ (max (binade |y|) (-126) - 23))).toNat = N₀ at hN ⊢
  generalize max (binade |y|) (-126) = E₀
  split_ifs <;> omega

/-- `f32Word` of a non-negative value below `2^127` is finite. -/
theorem f32Word_exp {y : ℚ} (hy : 0 ≤ y) (h : y < 2 ^ 127) : (f32Word y >>> 23) % 256 ≠ 255 := by
  unfold f32Word
  dsimp only
  rw [ite_eq_right (not_lt.2 hy)]
  by_cases h0 : |y| = 0
  · rw [ite_eq_left h0]; decide
  rw [ite_eq_right h0]
  have ha : 0 < |y| := lt_of_le_of_ne (abs_nonneg y) (Ne.symm h0)
  have hN := f32Word_N ha
  have hb : binade |y| ≤ 126 := by
    obtain ⟨hb1, -⟩ := binade_spec ha
    have h' : |y| < (2 : ℚ) ^ (127 : ℤ) := by
      rw [abs_of_nonneg hy, show ((127 : ℤ)) = ((127 : ℕ) : ℤ) by norm_num, zpow_natCast]; exact h
    have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 (lt_of_le_of_lt hb1 h')
    omega
  have hE : max (binade |y|) (-126) ≤ 126 := max_le hb (by norm_num)
  generalize (roundEven (|y| / 2 ^ (max (binade |y|) (-126) - 23))).toNat = N₀ at hN ⊢
  generalize max (binade |y|) (-126) = E₀ at hE ⊢
  simp only [Nat.shiftRight_eq_div_pow]
  split_ifs <;> omega

theorem sixth_wordQ : wordQ 0x3E2AAAAB = some sixth := wordQ_wordVal (by decide)

theorem sixth_lt : sixth < 1 / 2 := by decide +kernel

theorem floor_wordQ : wordQ 0x3B000000 = some (1 / 512) := by decide +kernel

theorem one_wordQ : wordQ 0x3F800000 = some 1 := by decide +kernel

/-- Every scale byte from `0x01` to `0x7E` has a positive value whose FP32 reciprocal is finite, positive and has the
sign bit clear. -/
theorem recip_facts : ∀ c : Code, 1 ≤ c.val → c.val ≤ 126 →
    0 < codeVal c ∧ f32Word (1 / codeVal c) < 2 ^ 31 ∧ (f32Word (1 / codeVal c) >>> 23) % 256 ≠ 255 ∧
      0 < wordVal (f32Word (1 / codeVal c)) := by
  decide +kernel

/-! ## The agreement -/

/-- **Core's primitives agree with the values `pc4`'s blocks read**, beyond `Pc8OpsAgree`'s (which has `F32Fabs`
clearing the sign bit and `F32Div` of finite words by a nonzero divisor as `f32Word` of the quotient): `F32Fmaxf` of two
finite words with the sign bit clear is the one of larger value, `a` at equal values (PTX `max.f32`); `F32ToE2m1Sat` of
a word that is not NaN is `e2m1Word`'s code (it saturates `±inf` to `±6`, as `e2m1Word` does reading them at `±480`);
`Bf16ToF32` widens by a 16-bit shift; and `I32Le` (signed 32-bit `a ≤ b`) of two words with bit 31 clear is their
order, which the running max `s` reads. At core's instance `coreOps` the last two hold by their definitions
(`coreOps_bf16w`, `coreOps_le`), and all of them from core's definitions (`CoreOps.pc4OpsAgree_coreOps`). -/
structure Pc4OpsAgree (ops : RowOps) : Prop extends WholeCircuit.Pc8OpsAgree ops where
  fmaxf : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y →
    signBit a = false → signBit b = false → ops.f32Fmaxf a b % 2 ^ 32 = if x < y then b else a
  e2m1 : ∀ w < 2 ^ 32, ((w >>> 23) % 256 = 255 → w % 2 ^ 23 = 0) → ops.f32ToE2m1Sat w % 2 ^ 4 = (e2m1Word w).val
  bf16w : ∀ w < 2 ^ 16, ops.bf16ToF32 w = w <<< 16
  le : ∀ a b : ℕ, a < 2 ^ 31 → b < 2 ^ 31 → ops.i32Le a b % 2 = if a ≤ b then 1 else 0

theorem and_bit31_eq_zero {a : ℕ} (ha : a < 2 ^ 31) : a &&& 0x80000000 = 0 := by
  show a &&& 2 ^ 31 = 0
  have hlt : a &&& 2 ^ 31 < 2 ^ 31 := by rw [Nat.and_comm]; exact Nat.and_lt_two_pow _ ha
  have hm := @Nat.and_mod_two_pow a (2 ^ 31) 31
  rw [Nat.mod_eq_of_lt hlt, Nat.mod_self, Nat.and_zero] at hm
  exact hm

/-- Core's `I32Le_v1` of two words below `2³¹` is their order: bit 31 of each is clear, so each reads as itself. -/
theorem coreOps_le : ∀ a b : ℕ, a < 2 ^ 31 → b < 2 ^ 31 → coreOps.i32Le a b % 2 = if a ≤ b then 1 else 0 := by
  intro a b ha hb
  by_cases h : a ≤ b <;>
    simp [coreOps, Verity.Scalar.i32Le, and_bit31_eq_zero ha, and_bit31_eq_zero hb, h]

/-- Core's `Bf16ToF32_v1` of a BF16 word is the word shifted left by 16. -/
theorem coreOps_bf16w : ∀ w < 2 ^ 16, coreOps.bf16ToF32 w = w <<< 16 := by
  intro w hw
  show (w &&& (2 ^ 16 - 1)) <<< 16 = w <<< 16
  rw [Nat.and_two_pow_sub_one_of_lt_two_pow hw]

/-! ## A block -/

section UnitKind

variable {ops : RowOps}

theorem fmaxf_word (hA : Pc4OpsAgree ops) {a b : ℕ} {x y : ℚ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31)
    (hx : wordQ a = some x) (hy : wordQ b = some y) :
    Prim.eval ops .fmaxf [a, b] < 2 ^ 31 ∧ wordQ (Prim.eval ops .fmaxf [a, b]) = some (max x y) := by
  have h := hA.fmaxf a b x y (by omega) (by omega) hx hy (signBit_of_lt ha) (signBit_of_lt hb)
  have he : Prim.eval ops .fmaxf [a, b] = ops.f32Fmaxf a b % 2 ^ 32 := rfl
  rw [he, h]
  split_ifs with hxy
  · exact ⟨hb, by rw [hy, max_eq_right hxy.le]⟩
  · exact ⟨ha, by rw [hx, max_eq_left (not_lt.1 hxy)]⟩

/-- **The block's running max** of finite words is the fold of their magnitudes from `0`, a word with the sign bit
clear. -/
theorem blkMax4_word (hA : Pc4OpsAgree ops) {d : ℕ → ℕ}
    (hd : ∀ s < 16, d s < 2 ^ 32 ∧ (d s >>> 23) % 256 ≠ 255) :
    ∀ s < 15, blkMax4 ops d s < 2 ^ 31 ∧
      wordQ (blkMax4 ops d s) = some (((List.range (s + 2)).map fun i => |wordVal (d i)|).foldl max 0) := by
  have hf : ∀ s < 16, ops.f32Fabs (d s) < 2 ^ 31 ∧ wordQ (ops.f32Fabs (d s)) = some |wordVal (d s)| := by
    intro s hs
    rw [hA.fabs]
    exact fabs_word (wordQ_wordVal (hd s hs).2)
  intro s hs
  induction s with
  | zero =>
    obtain ⟨h0, q0⟩ := hf 0 (by omega)
    obtain ⟨h1, q1⟩ := hf 1 (by omega)
    obtain ⟨r1, r2⟩ := fmaxf_word hA h0 h1 q0 q1
    have hb : blkMax4 ops d 0 = Prim.eval ops .fmaxf [ops.f32Fabs (d 0), ops.f32Fabs (d 1)] := rfl
    rw [hb]
    refine ⟨r1, ?_⟩
    rw [r2]
    simp [List.range_succ]
  | succ s ih =>
    obtain ⟨h0, q0⟩ := ih (by omega)
    obtain ⟨h1, q1⟩ := hf (s + 2) (by omega)
    obtain ⟨r1, r2⟩ := fmaxf_word hA h0 h1 q0 q1
    have hb : blkMax4 ops d (s + 1) = Prim.eval ops .fmaxf [blkMax4 ops d s, ops.f32Fabs (d (s + 2))] := rfl
    rw [hb]
    refine ⟨r1, ?_⟩
    rw [r2, show s + 1 + 2 = (s + 2) + 1 by omega, List.range_succ (n := s + 2), List.map_append,
      List.foldl_append]
    rfl

/-- **The scale byte** of a running max with the sign bit clear is `nvf4Scale` of its value: the table's floor at
`2⁻⁹` before the cast is the protocol's at `0x01` after it (`e4m3_floor`). -/
theorem scale4W_sem (hA : Pc4OpsAgree ops) {w : ℕ} {a : ℚ} (hw : w < 2 ^ 31) (hq : wordQ w = some a) :
    scale4W ops w = (nvf4Scale a).val := by
  have ha : 0 ≤ a := wordQ_nonneg hw hq
  have hs0 : 0 ≤ sixth := wordQ_nonneg (by norm_num) sixth_wordQ
  have hm : Prim.eval ops .mul [w, 0x3E2AAAAB] = f32Word (a * sixth) :=
    hA.mul w 0x3E2AAAAB a sixth (by omega) (by norm_num) hq sixth_wordQ
      (fun _ => by rw [signBit_of_lt hw, signBit_of_lt (by norm_num)])
  have hy : 0 ≤ a * sixth := mul_nonneg ha hs0
  have hab := wordQ_abs_lt hq
  rw [abs_of_nonneg ha] at hab
  have hylt : a * sixth < 2 ^ 127 := by nlinarith [sixth_lt]
  have hm31 : f32Word (a * sixth) < 2 ^ 31 := f32Word_lt31 hy
  have hmq := wordQ_wordVal (f32Word_exp hy hylt)
  have hmv : 0 ≤ wordVal (f32Word (a * sixth)) := wordQ_nonneg hm31 hmq
  obtain ⟨hf1, hf2⟩ := fmaxf_word hA hm31 (by norm_num : (0x3B000000 : ℕ) < 2 ^ 31) hmq floor_wordQ
  have hfe := wordQ_exp hf2
  have hfv : wordVal (Prim.eval ops .fmaxf [f32Word (a * sixth), 0x3B000000]) =
      max (wordVal (f32Word (a * sixth))) (1 / 512) := by simp [wordVal, hf2]
  have hcast : e4m3 (wordVal (Prim.eval ops .fmaxf [f32Word (a * sixth), 0x3B000000])) =
      e4m3Word (Prim.eval ops .fmaxf [f32Word (a * sixth), 0x3B000000]) :=
    e4m3_wordVal (fun h => absurd h hfe) (fun _ => by rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt hf1])
  unfold scale4W nvf4Scale
  rw [hm]
  show ops.f32ToE4m3Sat (Prim.eval ops .fmaxf [f32Word (a * sixth), 0x3B000000]) % 2 ^ 8 = _
  rw [hA.cast, ← hcast, hfv, ← e4m3_floor hmv]
  exact Nat.mod_eq_of_lt (Fin.isLt _)

theorem nvf4Scale_bounds {a : ℚ} (ha : 0 ≤ a) : 1 ≤ (nvf4Scale a).val ∧ (nvf4Scale a).val ≤ 126 := by
  have hs0 : 0 ≤ sixth := wordQ_nonneg (by norm_num) sixth_wordQ
  have hv := wordVal_nonneg (f32Word_lt31 (mul_nonneg ha hs0))
  have h1 := e4m3_val hv
  have h2 := e4m3Mag_le hv
  unfold nvf4Scale
  rcases le_total (e4m3 (wordVal (f32Word (a * sixth)))) 1 with h | h
  · rw [max_eq_right h]; exact ⟨le_refl _, by decide⟩
  · rw [max_eq_left h]; exact ⟨Fin.le_iff_val_le_val.1 h, by omega⟩

/-- **A code** of a step word at a scale byte from `0x01` to `0x7E` is `nvf4Block`'s: `e2m1Word` of the word's value
times `nvf4Recip`. -/
theorem code4W_sem (hA : Pc4OpsAgree ops) {d : ℕ} (hd : AccumProofs.StepWord d) {c : Code} (hc1 : 1 ≤ c.val)
    (hc2 : c.val ≤ 126) : code4W ops d c.val = (e2m1Word (f32Word (wordVal d * nvf4Recip c))).val := by
  obtain ⟨hv, hr31, hre, hrv⟩ := recip_facts c hc1 hc2
  have hnan : c.isNaN = false := by simp [Code.isNaN]; omega
  have hw : Prim.eval ops .e4m3ToF32 [c.val] = f32Word (codeVal c) := hA.widen c hnan (by omega)
  have hwq : wordQ (f32Word (codeVal c)) = some (codeVal c) := WholeCircuit.wordQ_f32Word_code c hnan
  have hr : Prim.eval ops .div [0x3F800000, f32Word (codeVal c)] = f32Word (1 / codeVal c) :=
    hA.div 0x3F800000 (f32Word (codeVal c)) 1 (codeVal c) (by norm_num) (f32Word_lt _) one_wordQ hwq hv.ne'
      (fun h => absurd h one_ne_zero)
  have hrq := wordQ_wordVal hre
  have hdq := wordQ_wordVal (AccumProofs.finite?_exp hd.2.2)
  have hp : Prim.eval ops .mul [d, f32Word (1 / codeVal c)] = f32Word (wordVal d * nvf4Recip c) := by
    refine hA.mul d (f32Word (1 / codeVal c)) (wordVal d) (nvf4Recip c) hd.1 (f32Word_lt _) hdq hrq (fun h0 => ?_)
    rw [signBit_of_lt hr31]
    rcases mul_eq_zero.1 h0 with h0 | h0
    · rw [AccumProofs.wordQ_zero_word hd (by rw [hdq, h0])]; rfl
    · exact absurd h0 hrv.ne'
  unfold code4W
  rw [hw, hr, hp]
  show ops.f32ToE2m1Sat (f32Word (wordVal d * nvf4Recip c)) % 2 ^ 4 = _
  exact hA.e2m1 _ (f32Word_lt _) (f32Word_notNaN _)

/-- **A block is `nvf4Block`'s** on its 16 words where they are step words: the table's scale byte is its scale
byte and each code its code. -/
theorem block4_sem (hA : Pc4OpsAgree ops) {d : ℕ → ℕ} (hd : ∀ s < 16, AccumProofs.StepWord (d s)) :
    scale4W ops (blkMax4 ops d 14) = (nvf4Block ((List.range 16).map d)).2.val ∧
      ∀ s < 16, code4W ops (d s) (scale4W ops (blkMax4 ops d 14)) =
        ((nvf4Block ((List.range 16).map d)).1.getD s 0).val := by
  have hd' : ∀ s < 16, d s < 2 ^ 32 ∧ (d s >>> 23) % 256 ≠ 255 :=
    fun s hs => ⟨(hd s hs).1, AccumProofs.finite?_exp (hd s hs).2.2⟩
  obtain ⟨h31, hq⟩ := blkMax4_word hA hd' 14 (by omega)
  have hs := scale4W_sem hA h31 hq
  have hfold : ((List.range 16).map d).map (fun w => |wordVal w|) =
      (List.range (14 + 2)).map fun i => |wordVal (d i)| := by rw [List.map_map]; rfl
  have ha : 0 ≤ ((List.range (14 + 2)).map fun i => |wordVal (d i)|).foldl max 0 :=
    wordQ_nonneg h31 hq
  obtain ⟨hb1, hb2⟩ := nvf4Scale_bounds ha
  refine ⟨by rw [hs]; simp only [nvf4Block, hfold], fun s hsl => ?_⟩
  rw [hs, code4W_sem hA (hd s hsl) hb1 hb2]
  simp [nvf4Block, hfold, List.getD_eq_getElem?_getD, hsl, (hd s hsl).2.1]

end UnitKind

/-! ## A formed word -/

/-- A formed word's operands after its accumulator: `e`'s 32 codes and 32 zeros, `f`'s and 32 zeros, then `e`'s
byte `aE` and `f`'s `24`, four times each. -/
def formArgs4 (aE : ℕ) (e f : ℕ → ℕ) : List ℕ :=
  (List.range 32).map e ++ List.replicate 32 0 ++ (List.range 32).map f ++ List.replicate 32 0 ++
    List.replicate 4 aE ++ List.replicate 4 24

theorem formArgs4_at (aE : ℕ) (e f : ℕ → ℕ) (n : ℕ) (hn : n < 136) :
    (formArgs4 aE e f).getD n 0 = if n < 32 then e n else if n < 64 then 0 else if n < 96 then f (n - 64)
      else if n < 128 then 0 else if n < 132 then aE else 24 := by
  simp only [formArgs4, List.getD_eq_getElem?_getD, List.getElem?_append, List.length_append, List.length_map,
    List.length_range, List.length_replicate, List.getElem?_map, List.getElem?_replicate]
  split_ifs <;> first | rfl | omega | (rw [List.getElem?_range (by omega)]; rfl)

theorem formArgs4_getD (aE : ℕ) (e f : ℕ → ℕ) :
    (∀ i < 64, (formArgs4 aE e f).getD i 0 = (if i < 32 then e i else 0) ∧
      (formArgs4 aE e f).getD (64 + i) 0 = (if i < 32 then f i else 0)) ∧
      ∀ i < 4, (formArgs4 aE e f).getD (128 + i) 0 = aE ∧ (formArgs4 aE e f).getD (132 + i) 0 = 24 := by
  refine ⟨fun i hi => ⟨?_, ?_⟩, fun i hi => ⟨?_, ?_⟩⟩
  · rw [formArgs4_at aE e f i (by omega)]
    by_cases h32 : i < 32
    · rw [ite_eq_left h32, ite_eq_left h32]
    · rw [ite_eq_right h32, ite_eq_right h32, ite_eq_left hi]
  · rw [formArgs4_at aE e f (64 + i) (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]
    by_cases h32 : i < 32
    · rw [ite_eq_left (by omega), ite_eq_left h32, show 64 + i - 64 = i by omega]
    · rw [ite_eq_right (by omega), ite_eq_right h32, ite_eq_left (by omega)]
  · rw [formArgs4_at aE e f (128 + i) (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega)]
  · rw [formArgs4_at aE e f (132 + i) (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]

theorem noise_list {E : ℕ → ℕ} {e : Fin 32 → Nib} (hE : ∀ l : Fin 32, E l = (e l).val) :
    (List.ofFn fun l : Fin 64 => nib (if (l : ℕ) < 32 then E l else 0)) = noiseOperand sm120Nvf4 e := by
  apply List.ext_getElem (by simp [noiseOperand, show sm120Nvf4.k = 64 from rfl])
  intro i h1 h2
  simp only [List.length_ofFn] at h1
  simp only [noiseOperand, List.getElem_ofFn, List.getElem_append, List.length_ofFn, List.getElem_replicate]
  by_cases h32 : i < 32
  · simp only [h32, ↓reduceIte, ↓reduceDIte]
    rw [hE ⟨i, h32⟩]; exact Fin.ext (Nat.mod_eq_of_lt (e _).isLt)
  · simp only [h32, ↓reduceIte, ↓reduceDIte]; rfl

section Form

variable {ops : RowOps}

theorem toE4m3_step (hA : WholeCircuit.Pc8OpsAgree ops) {β : ℕ} (hβ : AccumProofs.StepWord β) :
    Prim.eval ops .toE4m3 [β] = (e4m3 (wordVal β)).val := by
  show ops.f32ToE4m3Sat β % 2 ^ 8 = _
  rw [hA.cast, e4m3_wordVal (fun h => absurd h (AccumProofs.finite?_exp hβ.2.2))
    (fun h0 => by rw [AccumProofs.wordQ_zero_word hβ h0]; rfl)]
  exact Nat.mod_eq_of_lt (Fin.isLt _)

/-- **A formed word is `nvf4NoiseElem`'s** where the noise atom is in the model: the accumulator `α·x` is
`f32Word` of the values' product (`α > 0`, `x`'s widening a step word, so a zero product is `+0`), the noise scale byte
is `e4m3(β)` (`β` a step word), and the operands and scale bytes are `noiseOperand`'s and `noiseScales`'. -/
theorem form4_sem (hA : WholeCircuit.Pc8OpsAgree ops) {α x β : ℕ} {e f : Fin 32 → Nib} {E F : ℕ → ℕ}
    (hα : AccumProofs.StepWord α) (hαp : 0 < wordVal α) (hβ : AccumProofs.StepWord β)
    (hx : AccumProofs.StepWord (ops.bf16ToF32 x)) (hE : ∀ l : Fin 32, E l = (e l).val)
    (hF : ∀ l : Fin 32, F l = (f l).val) {w : ℕ}
    (hw : nvf4NoiseElem (wordVal α) (wordVal (ops.bf16ToF32 x)) (wordVal β) e f = some w) :
    form4 ops α x (Prim.eval ops .toE4m3 [β]) E F = w := by
  have hαq := WholeCircuit.stepWord_wordQ hα
  have hxq := WholeCircuit.stepWord_wordQ hx
  have hacc : Prim.eval ops .mul [α, ops.bf16ToF32 x] = f32Word (wordVal α * wordVal (ops.bf16ToF32 x)) :=
    hA.mul α (ops.bf16ToF32 x) (wordVal α) (wordVal (ops.bf16ToF32 x)) hα.1 hx.1 hαq hxq fun h0 => by
      rcases mul_eq_zero.1 h0 with h0 | h0
      · exact absurd h0 hαp.ne'
      · rw [WholeCircuit.signBit_of_pos hαq hαp, AccumProofs.wordQ_zero_word hx (by rw [hxq, h0])]; rfl
  have hbE := toE4m3_step hA hβ
  have hs : sm120Nvf4.step (f32Word (wordVal α * wordVal (ops.bf16ToF32 x))) (noiseOperand sm120Nvf4 e)
      (noiseOperand sm120Nvf4 f) (noiseScales sm120Nvf4 (e4m3 (wordVal β)))
      (noiseScales sm120Nvf4 (fScale sm120Nvf4.fmt)) = some w := hw
  obtain ⟨q1, q2⟩ := formArgs4_getD (e4m3 (wordVal β)).val E F
  unfold form4
  rw [hacc, hbE]
  show atom4F false (f32Word (wordVal α * wordVal (ops.bf16ToF32 x)) :: formArgs4 (e4m3 (wordVal β)).val E F) %
    2 ^ 32 = w
  simp only [atom4F, Bool.false_eq_true, ↓reduceIte, List.headD_cons, List.tail_cons]
  set L := formArgs4 (e4m3 (wordVal β)).val E F
  have a1 : (List.ofFn fun l : Fin 64 => nib (L.getD l 0)) = noiseOperand sm120Nvf4 e := by
    rw [← noise_list hE]; exact congrArg List.ofFn (funext fun l => by rw [(q1 l l.isLt).1])
  have a2 : (List.ofFn fun l : Fin 64 => nib (L.getD (64 + l) 0)) = noiseOperand sm120Nvf4 f := by
    rw [← noise_list hF]; exact congrArg List.ofFn (funext fun l => by rw [(q1 l l.isLt).2])
  have a3 : (List.ofFn fun l : Fin 4 => byte (L.getD (128 + l) 0)) = noiseScales sm120Nvf4 (e4m3 (wordVal β)) := by
    rw [show (List.ofFn fun l : Fin 4 => byte (L.getD (128 + l) 0)) = List.ofFn fun _ : Fin 4 => e4m3 (wordVal β)
      from congrArg List.ofFn (funext fun l => by
        rw [(q2 l l.isLt).1]; exact Fin.ext (Nat.mod_eq_of_lt (Fin.isLt _))), List.ofFn_const]
    rfl
  have a4 : (List.ofFn fun l : Fin 4 => byte (L.getD (132 + l) 0)) =
      noiseScales sm120Nvf4 (fScale sm120Nvf4.fmt) := by
    rw [show (List.ofFn fun l : Fin 4 => byte (L.getD (132 + l) 0)) = List.ofFn fun _ : Fin 4 => byte 24
      from congrArg List.ofFn (funext fun l => by rw [(q2 l l.isLt).2]), List.ofFn_const]
    rfl
  rw [a1, a2, a3, a4, nvf4Word_of_step hs, Nat.mod_eq_of_lt (step4_lt hs)]

end Form

/-! ## A formed row -/

theorem take_drop16 {l : List ℕ} {a : ℕ} (h : a + 16 ≤ l.length) :
    (l.drop a).take 16 = (List.range 16).map fun s => l.getD (a + s) 0 := by
  apply List.ext_getElem (by simp; omega)
  intro i h1 h2
  simp only [List.length_map, List.length_range] at h2
  rw [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range,
    List.getD_eq_getElem _ _ (by omega)]

theorem flatten_getD {α : Type} {n : ℕ} (d : α) :
    ∀ (L : List (List α)), (∀ i (h : i < L.length), L[i].length = n) →
      ∀ i < L.length, ∀ s < n, L.flatten.getD (n * i + s) d = (L.getD i []).getD s d
  | [], _, i, hi, _, _ => absurd hi (by simp)
  | b :: L, hL, i, hi, s, hs => by
    have hb : b.length = n := hL 0 (by simp)
    rw [List.flatten_cons]
    cases i with
    | zero => rw [Nat.mul_zero, Nat.zero_add, List.getD_append _ _ _ _ (by omega), List.getD_cons_zero]
    | succ i =>
      rw [List.getD_append_right _ _ _ _ (by rw [hb, Nat.mul_succ]; omega), hb,
        show n * (i + 1) + s - n = n * i + s by rw [Nat.mul_succ]; omega, List.getD_cons_succ]
      exact flatten_getD d L (fun i h => hL (i + 1) (by simp; omega)) i (by simp at hi; omega) s hs

/-! ## A firing writes step words -/

/-- **`toF32RZ` writes step words**: `+0` where the magnitude truncates to zero, otherwise a nonzero magnitude at an
exponent field below 255 with the sign bit beside it, so never `−0`. -/
theorem toF32RZ_stepWord {n lsb : ℤ} {w : ℕ} (h : toF32RZ n lsb = some w) : AccumProofs.StepWord w := by
  unfold toF32RZ at h
  dsimp only at h
  generalize hg : (if -126 ≤ (bitLength n.natAbs : ℤ) - 1 + lsb then (bitLength n.natAbs : ℤ) - 1 + lsb - 23
    else -149) = grid at h
  have hgrid : -149 ≤ grid := by rw [← hg]; split_ifs <;> omega
  generalize (if 0 < grid - lsb then n.natAbs >>> (grid - lsb).toNat else n.natAbs <<< (-(grid - lsb)).toNat) = q at h
  split at h
  · cases h; exact Fp4Freeze.stepWord_zero
  next hq0 =>
  split at h
  · cases h
  next h127 =>
  obtain rfl := Option.some.inj h
  have hsb : (if n < 0 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  by_cases he : -126 ≤ (bitLength q : ℤ) - 1 + grid
  · rw [ite_eq_left he, Fp4Freeze.shift_or]
    have hE : ((bitLength q : ℤ) - 1 + grid + 127).toNat < 255 := by omega
    have hE1 : 1 ≤ ((bitLength q : ℤ) - 1 + grid + 127).toNat := by omega
    have hr : q % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
    have hlt : ((bitLength q : ℤ) - 1 + grid + 127).toNat * 2 ^ 23 + q % 2 ^ 23 < 2 ^ 31 := by
      calc ((bitLength q : ℤ) - 1 + grid + 127).toNat * 2 ^ 23 + q % 2 ^ 23 < 254 * 2 ^ 23 + 2 ^ 23 := by
            have := Nat.mul_le_mul_right (2 ^ 23) (show ((bitLength q : ℤ) - 1 + grid + 127).toNat ≤ 254 by omega)
            omega
        _ ≤ 2 ^ 31 := by norm_num
    rw [Fp4Freeze.or_sign hlt, ← Nat.add_assoc]
    exact Fp4Freeze.stepWord_fields hsb hE hr (by have := Nat.mul_le_mul_right (2 ^ 23) hE1; omega)
  · have hb : bitLength q ≤ 23 := by omega
    have hq : q < 2 ^ 23 := lt_of_lt_of_le (AccumProofs.bitLength_spec q) (Nat.pow_le_pow_right (by norm_num) hb)
    rw [ite_eq_right he, Fp4Freeze.or_sign (lt_trans hq (by norm_num)),
      show (if n < 0 then 1 else 0) * 2 ^ 31 + q = (if n < 0 then 1 else 0) * 2 ^ 31 + 0 * 2 ^ 23 + q by ring]
    exact Fp4Freeze.stepWord_fields hsb (by norm_num) hq (by omega)

/-- **Every word a firing writes is a step word** (below `2^32`, not `−0`, finite): `+0` where nothing anchors the grid
or the total is zero, `toF32RZ`'s word otherwise. -/
theorem step_stepWord (p : Fp4Atom) {acc w : ℕ} {a b : List Nib} {sa sb : List (Fin 256)}
    (h : p.step acc a b sa sb = some w) : AccumProofs.StepWord w := by
  unfold Fp4Atom.step at h
  obtain ⟨ts, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨cv, -, h⟩ := Option.bind_eq_some_iff.1 h
  dsimp only at h
  generalize ts.filterMap (fun t => if t.live then some (t.unit - p.fracBits) else none) ++
    (if cv.mag = 0 then [] else [cv.exp - ((p.accWindow - 1 : ℕ) : ℤ)]) = L at h
  cases L with
  | nil => cases h; exact Fp4Freeze.stepWord_zero
  | cons x xs =>
    dsimp only at h
    split_ifs at h <;> first | exact toF32RZ_stepWord h | (cases h; exact Fp4Freeze.stepWord_zero)

/-- **`formRow`'s noise atoms are step words**: each is a firing's word. So a formed row never reads a `−0` noise
atom. -/
theorem noiseElem_stepWord {α x β : ℚ} {e f : Fin 32 → Nib} {w : ℕ} (h : nvf4NoiseElem α x β e f = some w) :
    AccumProofs.StepWord w := by
  unfold nvf4NoiseElem noiseAtom at h
  exact step_stepWord _ h

/-! ## `−0` in a row -/

/-- A BF16 row word `v` the table reads against the word `u` `pc4Sem` reads for its value (`encRow16`): the same word,
or `−0` (`0x8000`) where `pc4Sem` writes `+0`. -/
def ZeroSign16 (v u : ℕ) : Prop := v = u ∨ (v = 0x8000 ∧ u = 0)

section ZeroSign16

variable {ops : RowOps} (hA : Pc4OpsAgree ops)
include hA

theorem bf16_zeroSign16 {v u : ℕ} (h : ZeroSign16 v u) :
    WholeCircuit.ZeroSign (ops.bf16ToF32 v) (ops.bf16ToF32 u) := by
  rcases h with rfl | ⟨rfl, rfl⟩
  · exact Or.inl rfl
  · exact Or.inr ⟨by rw [hA.bf16w _ (by norm_num)]; rfl, by rw [hA.bf16w _ (by norm_num)]; rfl⟩

/-- **`α` reads no row word's sign of zero**: both wirings widen `−0` to FP32's `−0`, and `|−0| = |+0|`,
`fma(−0, −0, ·) = fma(+0, +0, ·)`. -/
theorem alpha4_zeroSign16 {K : ℕ} (hK : 8 ≤ K) {V U : ℕ → ℕ} (h : ∀ l < K, ZeroSign16 (V l) (U l)) :
    alpha4 ops V K = alpha4 ops U K := by
  unfold alpha4 drho4
  rw [WholeCircuit.rowRho_congrW (w := .bf16) (x := V) (y := U) hK fun l hl _ hc =>
      WholeCircuit.fmaSq_zeroSign hA.toPc8OpsAgree (bf16_zeroSign16 hA (h l hl)) hc,
    WholeCircuit.rowMax_congrW (ab := .bf16Fabs) (x := V) (y := U) (K - 1) fun l hl =>
      WholeCircuit.fabs_zeroSign hA.toPc8OpsAgree (bf16_zeroSign16 hA (h l (by omega)))]

theorem beta4_zeroSign16 {K : ℕ} (hK : 8 ≤ K) {V U : ℕ → ℕ} (h : ∀ l < K, ZeroSign16 (V l) (U l)) :
    beta4 ops V K = beta4 ops U K := by
  unfold beta4 drho4
  rw [WholeCircuit.rowRho_congrW (w := .bf16) (x := V) (y := U) hK fun l hl _ hc =>
      WholeCircuit.fmaSq_zeroSign hA.toPc8OpsAgree (bf16_zeroSign16 hA (h l hl)) hc, alpha4_zeroSign16 hA hK h]

omit hA in
/-- An NVF4 firing reads its accumulator only as a value: `−0` fires as `+0`. -/
theorem nvf4Word_negZero (a b : List Nib) (sa sb : List (Fin 256)) :
    nvf4Word 0x80000000 a b sa sb = nvf4Word 0 a b sa sb := by
  have hv : fp32Value 0x80000000 = fp32Value 0 := rfl
  have h1 : ¬ ((0x80000000 : ℕ) >>> 23) % 256 = 255 := by decide
  have h0 : ¬ ((0 : ℕ) >>> 23) % 256 = 255 := by decide
  unfold nvf4Word Fp4Atom.step
  simp only [h1, h0, ↓reduceIte, hv]

/-- `α·(±0)`, `α` a step word, is `+0` or `−0` (`mul`, `mulZero`). -/
theorem mul_zero4 {a v : ℕ} (ha : AccumProofs.StepWord a) (hv : v = 0 ∨ v = 0x80000000) :
    Prim.eval ops .mul [a, v] = 0 ∨ Prim.eval ops .mul [a, v] = 0x80000000 := by
  have hqa := WholeCircuit.stepWord_wordQ ha
  have hqv : wordQ v = some 0 := by
    rcases hv with rfl | rfl
    · exact Freeze.wordQ_zero
    · exact PeelExactProofs.wordQ_negZero
  have hvl : v < 2 ^ 32 := by rcases hv with rfl | rfl <;> norm_num
  simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
  by_cases hs : signBit a = signBit v
  · left
    rw [hA.mul _ _ _ _ ha.1 hvl hqa hqv fun _ => hs, mul_zero]
    simp [f32Word]
  · exact Or.inr (hA.mulZero _ _ _ _ ha.1 hvl hqa hqv (mul_zero _) hs)

omit hA in
theorem atom4_zeroAcc {m : ℕ} (hm : m = 0 ∨ m = 0x80000000) (L : List ℕ) :
    Prim.eval ops .atom4 (m :: L) = Prim.eval ops .atom4 (0 :: L) := by
  rcases hm with rfl | rfl
  · rfl
  · simp only [Prim.eval, atom4F, Bool.false_eq_true, ↓reduceIte, List.headD_cons, List.tail_cons,
      nvf4Word_negZero]

/-- **The forming reads no row word's sign of zero**: the firing reads `α·(−0)` and `α·(+0)` alike. -/
theorem form4_zeroSign16 {a v u aE : ℕ} (ha : AccumProofs.StepWord a) (h : ZeroSign16 v u) (e f : ℕ → ℕ) :
    form4 ops a v aE e f = form4 ops a u aE e f := by
  rcases h with rfl | ⟨rfl, rfl⟩
  · rfl
  · unfold form4
    have h8 : ops.bf16ToF32 0x8000 = 0x80000000 := by rw [hA.bf16w _ (by norm_num)]; rfl
    have h0 : ops.bf16ToF32 0 = 0 := by rw [hA.bf16w _ (by norm_num)]; rfl
    simp only [List.cons_append, h8, h0]
    rw [atom4_zeroAcc (mul_zero4 hA ha (Or.inr rfl)), atom4_zeroAcc (mul_zero4 hA ha (Or.inl rfl))]

end ZeroSign16

/-- The named conditions on a row `X` of `k` values under which the row units compute `formRow`'s codes and scale
bytes: each value's BF16 word widens to a step word of that value, and `α` (positive) and `β` are step words. -/
structure Row4OK (ops : RowOps) (k : ℕ) (X : Fin k → ℚ) : Prop where
  widen : ∀ t : Fin k, AccumProofs.StepWord (ops.bf16ToF32 (bf16WordOf (X t))) ∧
    wordVal (ops.bf16ToF32 (bf16WordOf (X t))) = X t
  alpha : AccumProofs.StepWord (alpha4 ops (encRow16 k X) k)
  alpha_pos : 0 < wordVal (alpha4 ops (encRow16 k X) k)
  beta : AccumProofs.StepWord (beta4 ops (encRow16 k X) k)

section Row

variable {ops : RowOps}

/-- **A row unit's codes and scale bytes are `formRow`'s** at `pc4Sem`'s `α` and `β`, where `formRow` is in the model:
on a row unit's inputs `x` holding `X`'s BF16 words, `e`'s codes and the form basis `F`'s, block `q`'s scale byte is
`formRow`'s and position `t`'s code is `formRow`'s. -/
theorem formRow_table (hA : Pc4OpsAgree ops) {K : ℕ} (hK : 8 ≤ K) (h16 : 16 ∣ K) {X : Fin K → ℚ}
    {e : Fin 32 → Nib} {F : Fin K → Fin 32 → Nib} {x : ℕ → ℕ} (hR : Row4OK ops K X)
    (hx : ∀ l < K, ZeroSign16 (x l) (encRow16 K X l)) (he : ∀ l : Fin 32, x (K + l) = (e l).val)
    (hf : ∀ (t : Fin K) (l : Fin 32), x (K + 32 + 32 * t + l) = (F t l).val) {r : Row4}
    (hr : formRow (wordVal (alpha4 ops (encRow16 K X) K)) (wordVal (beta4 ops (encRow16 K X) K)) X e F = some r) :
    (∀ q < K / 16, scale4V ops x K q = (r.2.getD q 0).val) ∧ ∀ t < K, code4V ops x K t = (r.1.getD t 0).val := by
  simp only [formRow, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hr
  obtain ⟨ds, hds, rfl⟩ := hr
  obtain ⟨hlen, hds'⟩ := PeelExactProofs.mapM_ofFn _ 0 hds
  have hdW : ∀ t (ht : t < K), d4W ops x K t = ds.getD t 0 := by
    intro t ht
    have hw : nvf4NoiseElem (wordVal (alpha4 ops (encRow16 K X) K))
        (wordVal (ops.bf16ToF32 (bf16WordOf (X ⟨t, ht⟩)))) (wordVal (beta4 ops (encRow16 K X) K)) e (F ⟨t, ht⟩) =
        some (ds.getD t 0) := by
      rw [(hR.widen ⟨t, ht⟩).2]; exact hds' ⟨t, ht⟩
    have hxt : encRow16 K X t = bf16WordOf (X ⟨t, ht⟩) := by rw [encRow16, dite_eq_left ht]
    unfold d4W
    rw [alpha4_zeroSign16 hA hK hx, beta4_zeroSign16 hA hK hx, form4_zeroSign16 hA hR.alpha (hx t ht), hxt]
    exact form4_sem hA.toPc8OpsAgree hR.alpha hR.alpha_pos hR.beta (hR.widen ⟨t, ht⟩).1 he (hf ⟨t, ht⟩) hw
  have h16' : 16 ∣ ds.length := by rw [hlen]; exact h16
  have hc : (ds.toChunks 16).length = K / 16 := by rw [Freeze.length_toChunks_of_dvd (by omega) h16', hlen]
  have hblk : ∀ q (hq : q < K / 16), (ds.toChunks 16)[q]'(by omega) =
      (List.range 16).map fun s => d4W ops x K (16 * q + s) := by
    intro q hq
    rw [Freeze.getElem_toChunks (by omega), take_drop16 (by omega)]
    exact List.map_congr_left fun s hs => (hdW _ (by simp at hs; omega)).symm
  have hstep : ∀ q < K / 16, ∀ s < 16, AccumProofs.StepWord (d4W ops x K (16 * q + s)) := by
    intro q hq s hs
    rw [hdW _ (by omega)]
    exact noiseElem_stepWord (hds' ⟨16 * q + s, by omega⟩)
  refine ⟨fun q hq => ?_, fun t ht => ?_⟩
  · rw [List.getD_eq_getElem _ _ (by simp; omega), List.getElem_map, List.getElem_map, hblk q hq]
    exact (block4_sem hA (hstep q hq)).1
  · obtain ⟨q, s, hs, rfl⟩ : ∃ q s, s < 16 ∧ t = 16 * q + s :=
      ⟨t / 16, t % 16, Nat.mod_lt _ (by omega), (Nat.div_add_mod t 16).symm⟩
    have hq : q < K / 16 := by omega
    have hL : ∀ i (h : i < (((ds.toChunks 16).map nvf4Block).map Prod.fst).length),
        (((ds.toChunks 16).map nvf4Block).map Prod.fst)[i].length = 16 := by
      intro i h
      simp only [List.getElem_map, nvf4Block, List.length_map]
      exact Freeze.length_getElem_toChunks_of_dvd (by omega) h16' i (by simpa using h)
    rw [flatten_getD 0 _ hL q (by simp; omega) s hs,
      List.getD_eq_getElem (((ds.toChunks 16).map nvf4Block).map Prod.fst) [] (by simp; omega), List.getElem_map,
      List.getElem_map, hblk q hq]
    show code4W ops (d4W ops x K (16 * q + s)) (scale4V ops x K ((16 * q + s) / 16)) = _
    rw [show (16 * q + s) / 16 = q by omega]
    exact (block4_sem hA (hstep q hq)).2 s hs

/-- **RowA's codes and scale bytes are `formRow`'s** in a transcript whose one unit is correct, on inputs holding the
row's BF16 words, `E_A`'s codes and `F_A`'s (`row4A_code`, `row4A_scale`, `formRow_table`). -/
theorem row4A_form (hA : Pc4OpsAgree ops) {K : ℕ} (hK : 8 ≤ K) (h16 : 16 ∣ K)
    (T : ((row4A K).spec ops).toCircuit.Transcript) (hT : ((row4A K).spec ops).part.Correct T 0)
    {X : Fin K → ℚ} {e : Fin 32 → Nib} {F : Fin K → Fin 32 → Nib} (hR : Row4OK ops K X)
    (hx : ∀ l < K, ZeroSign16 (val T l) (encRow16 K X l)) (he : ∀ l : Fin 32, val T (K + l) = (e l).val)
    (hf : ∀ (t : Fin K) (l : Fin 32), val T (K + 32 + 32 * t + l) = (F t l).val) {r : Row4}
    (hr : formRow (wordVal (alpha4 ops (encRow16 K X) K)) (wordVal (beta4 ops (encRow16 K X) K)) X e F = some r) :
    (∀ q < K / 16, val T (scale4 K (rowIn K false) q) = (r.2.getD q 0).val) ∧
      ∀ t < K, val T (code4 K (rowIn K false) t) = (r.1.getD t 0).val := by
  obtain ⟨h1, h2⟩ := formRow_table hA hK h16 hR hx he hf hr
  exact ⟨fun q hq => (row4A_scale hK T hT hq).trans (h1 q hq),
    fun t ht => (row4A_code hK T hT (by omega)).trans (h2 t ht)⟩

/-- **RowB's codes and scale bytes are `formRow`'s**, as `row4A_form` with `E_B`'s codes and `F_B`'s. -/
theorem row4B_form (hA : Pc4OpsAgree ops) {K : ℕ} (hK : 8 ≤ K) (h16 : 16 ∣ K)
    (T : ((row4B K).spec ops).toCircuit.Transcript) (hT : ((row4B K).spec ops).part.Correct T 0)
    {X : Fin K → ℚ} {e : Fin 32 → Nib} {F : Fin K → Fin 32 → Nib} (hR : Row4OK ops K X)
    (hx : ∀ l < K, ZeroSign16 (val T l) (encRow16 K X l)) (he : ∀ l : Fin 32, val T (K + l) = (e l).val)
    (hf : ∀ (t : Fin K) (l : Fin 32), val T (K + 32 + 32 * t + l) = (F t l).val) {r : Row4}
    (hr : formRow (wordVal (alpha4 ops (encRow16 K X) K)) (wordVal (beta4 ops (encRow16 K X) K)) X e F = some r) :
    (∀ q < K / 16, val T (scale4 K (rowIn K true) q) = (r.2.getD q 0).val) ∧
      ∀ t < K, val T (code4 K (rowIn K true) t) = (r.1.getD t 0).val := by
  obtain ⟨h1, h2⟩ := formRow_table hA hK h16 hR hx he hf hr
  exact ⟨fun q hq => (row4B_scale hK T hT hq).trans (h1 q hq),
    fun t ht => (row4B_code hK T hT (by omega)).trans (h2 t ht)⟩

end Row

theorem length_flatten_const {α : Type} {n : ℕ} :
    ∀ (L : List (List α)), (∀ b ∈ L, b.length = n) → L.flatten.length = n * L.length
  | [], _ => by simp
  | b :: L, h => by
    rw [List.flatten_cons, List.length_append, h b (by simp),
      length_flatten_const L (fun b' hb' => h b' (by simp [hb'])), List.length_cons]
    ring

/-- A formed row of `K` positions, `16 ∣ K`, has `K` codes and `K/16` scale bytes. -/
theorem formRow_length {K : ℕ} (h16 : 16 ∣ K) {α β : ℚ} {X : Fin K → ℚ} {e : Fin 32 → Nib}
    {F : Fin K → Fin 32 → Nib} {r : Row4} (hr : formRow α β X e F = some r) : r.1.length = K ∧ r.2.length = K / 16 := by
  simp only [formRow, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hr
  obtain ⟨ds, hds, rfl⟩ := hr
  obtain ⟨hlen, -⟩ := PeelExactProofs.mapM_ofFn _ 0 hds
  have h16' : 16 ∣ ds.length := by rw [hlen]; exact h16
  have hc : (ds.toChunks 16).length = K / 16 := by rw [Freeze.length_toChunks_of_dvd (by omega) h16', hlen]
  refine ⟨?_, by simp [hc]⟩
  rw [length_flatten_const (n := 16) _ ?_]
  · simp only [List.length_map, hc]; omega
  · intro b hb
    simp only [List.map_map, List.mem_map] at hb
    obtain ⟨c, hcm, rfl⟩ := hb
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hcm
    simp [nvf4Block, Freeze.length_getElem_toChunks_of_dvd (by omega) h16' i hi]

/-! ## `P_A` and `P_B` -/

theorem chainN4_lt : ∀ (n c : ℕ) (a b : List Nib) (sa sb : List (Fin 256)) {w : ℕ}, c < 2 ^ 32 →
    sm120Nvf4.chainN n c a b sa sb = some w → w < 2 ^ 32
  | 0, c, _, _, _, _, w, hc, h => by
    simp only [Fp4Atom.chainN, Option.some.injEq] at h
    exact h ▸ hc
  | n + 1, c, a, b, sa, sb, w, hc, h => by
    simp only [Fp4Atom.chainN] at h
    obtain ⟨d, hd, h⟩ := Option.bind_eq_some_iff.1 h
    exact chainN4_lt n d _ _ _ _ (step4_lt hd) h

theorem cTilde4_lt {a b : List Nib} {sa sb : List (Fin 256)} {w : ℕ} (h : cTilde sm120Nvf4 a b sa sb = some w) :
    w < 2 ^ 32 := by
  unfold cTilde Fp4Atom.chain at h
  split_ifs at h
  exact chainN4_lt _ 0 _ _ _ _ (by norm_num) h

/-- `PouwETilde_v1` at a scale byte that is not NaN is `DeviceFp4`'s `Ẽ`. -/
theorem eTilde_prim (ops : RowOps) (e : Fin 32 → Nib) (a : Fin 256) (ha : a.val % 128 ≠ 127) (q : Fin 32) :
    Prim.eval ops .eTilde [(e q).val, a.val] = eTilde e a q := by
  simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ, eTildeWord, ha, ↓reduceIte, nib_val,
    code_val]
  refine Nat.mod_eq_of_lt (Nat.or_lt_two_pow (WholeCircuit.bf16WordOf_lt _) ?_)
  split <;> norm_num

section Sides

variable {ops : RowOps}

/-- **RowA's `P_A` row is the FP4 peel's**: at depth `K = 64·nw`, a correct RowA transcript holding a row `X`, its `E_A`
line `e`, the form basis `F` and the chain basis `F_B`, where `formRow` forms `X` to `r` (`row4A_form`'s conditions),
has on its `Ẽ` outputs and its `P_A` outputs the row `paRow?` forms from the line's `Ẽ` at `e4m3(β)` and the chain
words `af` of `r` against `F_B`'s columns, wherever they exist (`peelFp4`'s `P_A,i`). -/
theorem row4A_paWords (hA : Pc4OpsAgree ops) {K nw : ℕ} (hnw : 0 < nw) (hK : K = 64 * nw)
    (T : ((row4A K).spec ops).toCircuit.Transcript) (hT : ((row4A K).spec ops).part.Correct T 0)
    {X : Fin K → ℚ} {e : Fin 32 → Nib} {F FB : Matrix (Fin K) (Fin 32) Nib} (hR : Row4OK ops K X)
    (hx : ∀ l < K, ZeroSign16 (val T l) (encRow16 K X l)) (he : ∀ l : Fin 32, val T (K + l) = (e l).val)
    (hf : ∀ (t : Fin K) (l : Fin 32), val T (K + 32 + 32 * t + l) = (F t l).val)
    (hb : ∀ (t : Fin K) (q : Fin 32), val T (33 * K + 32 + 32 * t + q) = (FB t q).val) {r : Row4}
    (hr : formRow (wordVal (alpha4 ops (encRow16 K X) K)) (wordVal (beta4 ops (encRow16 K X) K)) X e F = some r)
    {af pa : List ℕ}
    (haf : (List.ofFn fun q => (cTilde sm120Nvf4 r.1 (colF FB q) r.2 (onesScales sm120Nvf4 K)).bind
      PeelExact.finite?).mapM id = some af)
    (hpa : PeelExact.paRow? (List.ofFn (eTilde e (e4m3 (wordVal (beta4 ops (encRow16 K X) K))))) af = some pa) :
    ∀ q < 32, val T (post4At K (rowIn K false) + 32 * (K / 64 + 1) + q) = pa.getD q 0 ∧
      val T (post4At K (rowIn K false) + q * (K / 64 + 1) + K / 64) = pa.getD (32 + q) 0 := by
  have h8 : 8 ≤ K := by omega
  have h16 : 16 ∣ K := ⟨4 * nw, by omega⟩
  obtain ⟨hS, hC⟩ := row4A_form hA h8 h16 T hT hR hx he hf hr
  have hlen := (formRow_length h16 hr).1
  obtain ⟨hpa1, hpa2⟩ := PeelExactProofs.paRow_spec hpa List.length_ofFn
  obtain ⟨hafl, hafq⟩ := PeelExactProofs.mapM_ofFn _ 0 haf
  intro q hq
  refine ⟨?_, ?_⟩
  · have he' : val T (K + q) = (e ⟨q, hq⟩).val := he ⟨q, hq⟩
    rw [row4A_et h8 T hT hq, he', beta4_zeroSign16 hA h8 hx, toE4m3_step hA.toPc8OpsAgree hR.beta,
      eTilde_prim ops e _ (e4m3_notNaN _) ⟨q, hq⟩, hpa1 q hq, WholeCircuit.getD_ofFn32 _ hq]
  · obtain ⟨c, hc1, hc2⟩ := Option.bind_eq_some_iff.1 (hafq ⟨q, hq⟩)
    have hcw : c = af.getD q 0 := by
      unfold PeelExact.finite? at hc2
      split_ifs at hc2
      exact Option.some.inj hc2
    have hB : ∀ t < K, val T (33 * K + 32 + 32 * t + q) = ((colF FB ⟨q, hq⟩).getD t 0).val := by
      intro t ht
      refine (hb ⟨t, ht⟩ ⟨q, hq⟩).trans ?_
      unfold colF
      rw [PeelExactProofs.getD_ofFn (fun t => FB t ⟨q, hq⟩) 0 ⟨t, ht⟩]
    rw [row4A_pa hnw hK T hT r hlen hC hS hq (colF FB ⟨q, hq⟩) hB hc1]
    exact WholeCircuit.bf16_word hA.toPc8OpsAgree (cTilde4_lt hc1) (by rw [hcw]; exact hpa2 q (by omega))

theorem eTilde_lt (e : Fin 32 → Nib) (a : Fin 256) (q : Fin 32) : eTilde e a q < 2 ^ 16 :=
  Nat.or_lt_two_pow (WholeCircuit.bf16WordOf_lt _) (by split <;> norm_num)

/-- **RowB's `P_B` row is the FP4 peel's**: as `row4A_paWords`, a correct RowB transcript holding a row `X`, its `E_B`
line `e`, the form basis `F`, the chain basis `F_A` and `G`'s rows `gL` has on its `P_B` outputs and its `−Ẽ` outputs
the row `pbRow?` forms from the line's `Ẽ` at `e4m3(β)`, the chain words `bf` of `r` against `F_A`'s columns and `gL`,
wherever they exist (`peelFp4`'s `P_B,j`). -/
theorem row4B_pbWords (hA : Pc4OpsAgree ops) {K nw : ℕ} (hnw : 0 < nw) (hK : K = 64 * nw)
    (T : ((row4B K).spec ops).toCircuit.Transcript) (hT : ((row4B K).spec ops).part.Correct T 0)
    {X : Fin K → ℚ} {e : Fin 32 → Nib} {F FA : Matrix (Fin K) (Fin 32) Nib} (hR : Row4OK ops K X)
    (hx : ∀ l < K, ZeroSign16 (val T l) (encRow16 K X l)) (he : ∀ l : Fin 32, val T (K + l) = (e l).val)
    (hf : ∀ (t : Fin K) (l : Fin 32), val T (K + 32 + 32 * t + l) = (F t l).val)
    (hb : ∀ (t : Fin K) (q : Fin 32), val T (33 * K + 32 + 32 * t + q) = (FA t q).val) {r : Row4}
    (hr : formRow (wordVal (alpha4 ops (encRow16 K X) K)) (wordVal (beta4 ops (encRow16 K X) K)) X e F = some r)
    {bf pb : List ℕ} {gL : List (List ℕ)}
    (hbf : (List.ofFn fun q => (cTilde sm120Nvf4 r.1 (colF FA q) r.2 (onesScales sm120Nvf4 K)).bind
      PeelExact.finite?).mapM id = some bf)
    (hgl : gL.length = 32)
    (hg : ∀ s < 32, ∀ q < 32, val T (65 * K + 32 + 32 * s + q) = (gL.getD s []).getD q 0 % 2 ^ 32)
    (hpb : PeelExact.pbRow? (List.ofFn (eTilde e (e4m3 (wordVal (beta4 ops (encRow16 K X) K))))) bf gL = some pb) :
    ∀ q < 32, val T (post4At K (rowIn K true) + 32 + q * (K / 64 + 35) + K / 64 + 34) = pb.getD q 0 ∧
      val T (post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + q) = pb.getD (32 + q) 0 := by
  have h8 : 8 ≤ K := by omega
  have h16 : 16 ∣ K := ⟨4 * nw, by omega⟩
  obtain ⟨hS, hC⟩ := row4B_form hA h8 h16 T hT hR hx he hf hr
  have hlen := (formRow_length h16 hr).1
  obtain ⟨hpb1, hpb2⟩ := PeelExactProofs.pbRow_spec hpb List.length_ofFn
  obtain ⟨-, hbfq⟩ := PeelExactProofs.mapM_ofFn _ 0 hbf
  have hET : ∀ s (hs : s < 32), Prim.eval ops .eTilde [val T (K + s), Prim.eval ops .toE4m3 [beta4 ops (val T) K]] =
      eTilde e (e4m3 (wordVal (beta4 ops (encRow16 K X) K))) ⟨s, hs⟩ := by
    intro s hs
    have he' : val T (K + s) = (e ⟨s, hs⟩).val := he ⟨s, hs⟩
    rw [he', beta4_zeroSign16 hA h8 hx, toE4m3_step hA.toPc8OpsAgree hR.beta, eTilde_prim ops e _ (e4m3_notNaN _) ⟨s, hs⟩]
  intro q hq
  refine ⟨?_, ?_⟩
  · obtain ⟨acc, s, hacc, hs, hrn⟩ := hpb1 q hq
    obtain ⟨c, hc1, hc2⟩ := Option.bind_eq_some_iff.1 (hbfq ⟨q, hq⟩)
    have hcw : c = bf.getD q 0 := by
      unfold PeelExact.finite? at hc2
      split_ifs at hc2
      exact Option.some.inj hc2
    have hB : ∀ t < K, val T (33 * K + 32 + 32 * t + q) = ((colF FA ⟨q, hq⟩).getD t 0).val := by
      intro t ht
      refine (hb ⟨t, ht⟩ ⟨q, hq⟩).trans ?_
      unfold colF
      rw [PeelExactProofs.getD_ofFn (fun t => FA t ⟨q, hq⟩) 0 ⟨t, ht⟩]
    have hlist : ((List.range (31 + 1)).map fun s =>
        ((List.ofFn (eTilde e (e4m3 (wordVal (beta4 ops (encRow16 K X) K))))).getD s 0, (gL.getD s []).getD q 0)) =
          List.ofFn fun p : Fin 32 =>
            (eTilde e (e4m3 (wordVal (beta4 ops (encRow16 K X) K))) p, (gL.getD p []).getD q 0) := by
      apply List.ext_getElem
      · simp
      · intro n h1 h2
        simp only [List.getElem_map, List.getElem_range, List.getElem_ofFn]
        rw [WholeCircuit.getD_ofFn32 _ (by simpa using h2)]
    have hacc' : pbAcc ops (fun s => ops.bf16ToF32 (Prim.eval ops .eTilde
          [val T (K + s), Prim.eval ops .toE4m3 [beta4 ops (val T) K]]))
        (fun s => val T (65 * K + 32 + 32 * s + q)) 31 = acc := by
      refine WholeCircuit.pbAcc_fold hA.toPc8OpsAgree
        (be := fun s => (List.ofFn (eTilde e (e4m3 (wordVal (beta4 ops (encRow16 K X) K))))).getD s 0)
        (G := fun s => (gL.getD s []).getD q 0) (fun s hs => ?_) (fun s hs => ?_) (fun s hs => hg s hs q hq) 31
        (by norm_num) acc ?_
      · dsimp only
        rw [hET s hs, WholeCircuit.getD_ofFn32 _ hs]
        exact hA.bf16w _ (eTilde_lt _ _ _)
      · dsimp only
        rw [hET s hs, hA.bf16w _ (eTilde_lt _ _ _), Nat.shiftLeft_eq]
        have := eTilde_lt e (e4m3 (wordVal (beta4 ops (encRow16 K X) K))) ⟨s, hs⟩
        calc _ < 2 ^ 16 * 2 ^ 16 := Nat.mul_lt_mul_of_pos_right this (by norm_num)
          _ = 2 ^ 32 := by norm_num
      · rw [hlist]
        unfold PeelExact.accChain? at hacc
        rwa [PeelExactProofs.zip_col hgl q] at hacc
    have hneg : Prim.eval ops .neg [c] = negWord (bf.getD q 0) := by
      simp only [Prim.eval, Prim.w, List.getD_cons_zero]
      rw [← hcw]
      exact hA.neg _ (cTilde4_lt hc1)
    have hadd : Prim.eval ops .add [acc, negWord (bf.getD q 0)] = s := by
      simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
      rw [f32Add_of_addWord hs]
      exact Nat.mod_eq_of_lt (WholeCircuit.addWord?_lt hs)
    rw [row4B_pb hnw hK T hT r hlen hC hS hq (colF FA ⟨q, hq⟩) hB hc1, hacc', hneg, hadd]
    exact WholeCircuit.bf16_word hA.toPc8OpsAgree (WholeCircuit.addWord?_lt hs) hrn
  · rw [row4B_net h8 T hT hq, hET q hq, hpb2 q hq, WholeCircuit.getD_ofFn32 _ hq]
    simp only [Prim.eval, Prim.w, List.getD_cons_zero, bf16NegWord, Nat.mod_mod]
    exact Nat.mod_eq_of_lt (Nat.xor_lt_two_pow (eTilde_lt _ _ _) (by norm_num))

end Sides

end Pouw.PearlC.RowCircuit
