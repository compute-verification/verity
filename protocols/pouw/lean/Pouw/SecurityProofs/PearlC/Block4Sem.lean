import Pouw.SecurityProofs.PearlC.Row4Words
import Pouw.SecurityProofs.PearlC.TileCheck8Words

/-!
# `pc4`'s blocks in DeviceFp4's terms (proofs)

`Row4Words` proves a `pc4` block's scale byte and codes as word formulas on `RowOps`: the scale byte
`e4m3(max(amax·f32(1/6), 2⁻⁹))` of the block's running `fmaxf` of `fabs` (`scale4W`, `blkMax4`) and each code
`e2m1(d·(1/scale))` (`code4W`). `DeviceFp4.nvf4Block` reads them on values: `max(e4m3(f32(amax·sixth)), 0x01)` of the
largest magnitude and `e2m1Word(f32(D_t·nvf4Recip s))`. Here they agree, under one named hypothesis on `ops`:

* `Pc4OpsAgree ops`: `Pc8OpsAgree`'s agreements, and `F32Fabs` clears the sign bit, `F32Fmaxf` of two finite words
  with the sign bit clear is the one of larger value (`a` at equal values), `F32Div` is `f32Word` of the quotient by
  a nonzero divisor, and `F32ToE2m1Sat` of a word that is not NaN is `e2m1Word`'s code.

The floors differ in order: the table floors the FP32 value at `2⁻⁹` and then casts, the protocol casts and then
floors the code at `0x01`. On a non-negative value the two agree (`e4m3_floor`), since `e4m3` is at least `0x01` from
`2⁻⁹` up and at most `0x01` below it. A block's words need only be step words (finite, not `−0`): the running max is
the fold's (`blkMax4_word`), the scale byte is `nvf4Scale`'s (`scale4W_sem`) and the codes are `nvf4Block`'s
(`code4W_sem`, `block4_sem`).
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev
  Pouw.PearlC.PeelExact

/-! ## E4M3 at the floor -/

theorem roundEven_le_of_lt {m : ℚ} {n : ℤ} (h : m < n + 1 / 2) : roundEven m ≤ n := by
  have hf : ⌊m⌋ < n + 1 := Int.floor_lt.2 (by push_cast; linarith)
  rcases (show ⌊m⌋ < n ∨ ⌊m⌋ = n by omega) with hl | he
  · have := roundEven_le m; omega
  · unfold roundEven; dsimp only
    rw [he, ite_eq_left (by linarith)]

/-- `e4m3`'s magnitude bits on a non-negative value are at most `126` (`448`). -/
theorem e4m3Mag_le {t : ℚ} (ht : 0 ≤ t) : e4m3Mag t ≤ 126 := by
  rcases ht.lt_or_eq with ht | rfl
  swap
  · simp [e4m3Mag]
  unfold e4m3Mag
  rw [ite_eq_right ht.ne']
  by_cases h464 : 464 ≤ t
  · rw [ite_eq_left h464]
  rw [ite_eq_right h464]
  rw [not_le] at h464
  dsimp only
  obtain ⟨hb1, hb2⟩ := binade_spec ht
  have hb9 : binade t < 9 := by
    have : (2 : ℚ) ^ binade t < 2 ^ (9 : ℤ) := lt_of_le_of_lt hb1 (by norm_num; linarith)
    exact (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 this
  have hE8 : max (binade t) (-6) ≤ 8 := max_le (by omega) (by norm_num)
  have hu : (0 : ℚ) < 2 ^ (max (binade t) (-6) - 3) := two_zpow_pos _
  have k16 : (2 : ℚ) ^ (max (binade t) (-6) - 3) * 16 = 2 ^ (max (binade t) (-6) + 1) := by
    rw [show (16 : ℚ) = 2 ^ (4 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero]; congr 1; ring
  have hlt : t < 2 ^ (max (binade t) (-6) + 1) :=
    lt_of_lt_of_le hb2 (zpow_le_zpow_right₀ one_le_two (by omega))
  have h16 : t / 2 ^ (max (binade t) (-6) - 3) < ((16 : ℤ) : ℚ) + 1 / 2 := by
    rw [div_lt_iff₀ hu]; push_cast; linarith
  have hR := roundEven_le_of_lt h16
  rcases (show max (binade t) (-6) ≤ 7 ∨ max (binade t) (-6) = 8 by omega) with h7 | h8
  · omega
  · have h14 : t / 2 ^ (max (binade t) (-6) - 3) < ((14 : ℤ) : ℚ) + 1 / 2 := by
      rw [h8, div_lt_iff₀ (by norm_num)]; norm_num; linarith
    have := roundEven_le_of_lt h14
    omega

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

/-- `f32Word` never writes a NaN. -/
theorem f32Word_notNaN (y : ℚ) : (f32Word y >>> 23) % 256 = 255 → f32Word y % 2 ^ 23 = 0 := by
  unfold f32Word
  dsimp only
  by_cases h0 : |y| = 0
  · rw [ite_eq_left h0]; simp
  rw [ite_eq_right h0]
  have hN := f32Word_N (lt_of_le_of_ne (abs_nonneg y) (Ne.symm h0))
  generalize (roundEven (|y| / 2 ^ (max (binade |y|) (-126) - 23))).toNat = N₀ at hN ⊢
  generalize max (binade |y|) (-126) = E₀
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

/-- **Core's primitives agree with the values `pc4`'s blocks read**, beyond `Pc8OpsAgree`'s: `F32Fabs` clears the
sign bit; `F32Fmaxf` of two finite words with the sign bit clear is the one of larger value, `a` at equal values
(PTX `max.f32`); `F32Div` of finite words by a nonzero divisor is `f32Word` of the quotient, an exact zero signed as
`F32Mul`'s; `F32ToE2m1Sat` of a word that is not NaN is `e2m1Word`'s code (it saturates `±inf` to `±6`, as `e2m1Word`
does reading them at `±480`). Core's Lean semantics will discharge them. -/
structure Pc4OpsAgree (ops : RowOps) : Prop extends WholeCircuit.Pc8OpsAgree ops where
  fabs : ∀ w, ops.f32Fabs w = w % 2 ^ 31
  fmaxf : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y →
    signBit a = false → signBit b = false → ops.f32Fmaxf a b % 2 ^ 32 = if x < y then b else a
  div : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y → y ≠ 0 →
    (x = 0 → signBit a = signBit b) → ops.f32DivV2 a b % 2 ^ 32 = f32Word (x / y)
  e2m1 : ∀ w < 2 ^ 32, ((w >>> 23) % 256 = 255 → w % 2 ^ 23 = 0) → ops.f32ToE2m1Sat w % 2 ^ 4 = (e2m1Word w).val

/-! ## A block -/

section Block

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
  simp [nvf4Block, hfold, List.getD_eq_getElem?_getD, hsl]

end Block

end Pouw.PearlC.RowCircuit
