import Proofs.Pouw.PearlC.Block4Sem

/-!
# A row's `α` and `β` by magnitudes (proofs)

`Stats8OK` (`pc8`) and `Row4OK`'s `alpha`, `alpha_pos` and `beta` (`pc4`) ask that a row's `α` and `β` be step words
(finite, not `−0`), with `β > 0` and `α > 0` respectively. Here they follow from the row's magnitudes, under
`Pc8OpsAgree`:

* `α = F32Div(q, F32Fmaxf(t, 2⁻³²))` is a step word on every word `t` (`alpha_spec`): `+0` where `t` is `+inf`, else
  `f32Word(q/d)` for a finite `d ≥ max(t, 2⁻³²)`, positive and within `2⁻²⁴` of `q/d`.
* `pc8` (`stats8OK_of_rho`): where `ρ`'s word is finite, `ρ ≥ 0` (`sqrtWord_nonneg`). At `ρ = 0`, or where
  `t = s + 4ρ` is `+inf` (so `α = +0`), the floor `F32Fma(ρ, α, −1)` is about `−1`. Else `t` is finite and
  `d ≥ t ≥ 4ρ` up to three roundings, since `s ≥ 0` (`rowMax_word`), so `ρ·α ≤ 113`; a floor of at least `0` then
  gives `ρ·α ≥ 1/2`, and `β = F32Mul(F32Mul(ρ, α), 2^(−19/2))` is a positive finite word. `pc8Sem`'s floor is `0`
  where `ρ`'s word is not finite, so the floor rule asks a finite `ρ`, as Python (`s5_stats`' partials) and the
  circuit (`ρ ≤ F32_MAX`) do.
* `pc4` (`row4OK_stats_of_beta`): where `β`'s word is finite and `RowAdmit4` holds, `β > 0`, so `α > 0`: at `α = +0`
  the product `δρ·α` is `±0` or not finite (`mulNonfin`), and `β` with it. `pc4Sem`'s `β` is `0` where `β`'s word is
  not finite, so `RowAdmit4` asks a finite `β`, as the circuit (`β ≤ F32_MAX`) and Python (`pearl_c4.stats`) do.
* `pc4`'s forming (`alpha4_mul_le`): under `Pc4OpsAgree`, whose `I32Le` orders words below `2³¹`, the running max `s`
  is the largest `|x_l|` (`rowMax_ge`; word order is value order on finite words with bit 31 clear, `wordQ_mono`). On
  a `Row4OK` row `ρ` is finite, since `β` is, so `δρ ≥ 0` and `t = s + 4·δρ` is `+inf` (so `α = +0`) or finite and at
  least `s` up to one rounding: `α·|x_l| ≤ 2689`, under which `formRow`'s noise atoms are finite (`PeelExists4`).

The squares-overflow row (`test_circuit_pc8`'s and `test_circuit_pc4`'s `squares-overflow`: finite words, every 8th one
`2¹²⁶`) is the vector both refusals are checked on: each sampled square overflows, so `ρ` is `+inf`, and `pc8Sem`
(`pc8Sem_squaresOverflow`) and `pc4Sem` (`pc4Sem_squaresOverflow`) refuse the row, as Python and the circuit do.
`Fp4Sem.RowOK` also refuses a word that is not a widened BF16 word and a non-finite word (`rowOK4_notBf16`,
`rowOK4_nonFinite`), as `pearl_c4.row_passes` does. `PearlCSem.RowOK` refuses a non-finite word and a row that is not
live (`rowOK8_nonFinite`, `rowOK8_notLive`): with the squares-overflow row, one weight row per clause of the rule a
tile's check asks of its weight rows, as `check_tile` does (`test_check_tile_refuses_a_weight_row_outside_the_rules`).
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.PeelExact
  Pouw.PearlC.AccumProofs Pouw.PearlC.WholeCircuit

/-! ## Words -/

theorem wordQ_none_exp {w : ℕ} (h : wordQ w = none) : (w >>> 23) % 256 = 255 := by
  by_contra he
  rw [wordQ_wordVal he] at h
  cases h

theorem finite?_of_wordQ {w : ℕ} {x : ℚ} (h : wordQ w = some x) : finite? w = some w := by
  unfold finite?
  rw [ite_eq_right (wordQ_exp h)]

theorem stepWord_of_lt31 {w : ℕ} {x : ℚ} (hw : w < 2 ^ 31) (h : wordQ w = some x) : StepWord w :=
  ⟨by omega, by omega, finite?_of_wordQ h⟩

theorem wordVal_of_wordQ {w : ℕ} {x : ℚ} (h : wordQ w = some x) : PearlC.wordVal w = x := by
  simp [PearlC.wordVal, h]

/-- A step word is finite. -/
theorem stepWord_wordQ {w : ℕ} (h : StepWord w) : ∃ v, wordQ w = some v := by
  obtain ⟨-, v, hv⟩ := PeelExactProofs.finite?_eq_some h.2.2
  exact ⟨v, hv⟩

theorem fields_mono {a b : ℕ} (hab : a ≤ b) (hb : b < 2 ^ 31) :
    ((if (a >>> 23) % 256 = 0 then 0 else 2 ^ 23) + a % 2 ^ 23) * 2 ^ (max ((a >>> 23) % 256) 1) ≤
      ((if (b >>> 23) % 256 = 0 then 0 else 2 ^ 23) + b % 2 ^ 23) * 2 ^ (max ((b >>> 23) % 256) 1) := by
  have hea : (a >>> 23) % 256 = a / 2 ^ 23 := by rw [Nat.shiftRight_eq_div_pow]; omega
  have heb : (b >>> 23) % 256 = b / 2 ^ 23 := by rw [Nat.shiftRight_eq_div_pow]; omega
  rw [hea, heb]
  have hle : a / 2 ^ 23 ≤ b / 2 ^ 23 := Nat.div_le_div_right hab
  rcases hle.lt_or_eq with hlt | heq
  · have hb1 : 1 ≤ b / 2 ^ 23 := by omega
    rw [max_eq_left hb1, ite_eq_right (show ¬b / 2 ^ 23 = 0 by omega)]
    calc _ ≤ 2 ^ 23 * 2 ^ (b / 2 ^ 23) := ?_
      _ ≤ _ := Nat.mul_le_mul_right _ (by omega)
    by_cases h0 : a / 2 ^ 23 = 0
    · rw [h0, ite_eq_left rfl, max_eq_right (by omega)]
      calc (0 + a % 2 ^ 23) * 2 ^ 1 ≤ 2 ^ 23 * 2 ^ 1 := Nat.mul_le_mul_right _ (by omega)
        _ ≤ _ := Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by norm_num) hb1)
    · rw [ite_eq_right h0, max_eq_left (by omega)]
      calc (2 ^ 23 + a % 2 ^ 23) * 2 ^ (a / 2 ^ 23) ≤ 2 ^ 24 * 2 ^ (a / 2 ^ 23) := Nat.mul_le_mul_right _ (by omega)
        _ = 2 ^ 23 * 2 ^ (a / 2 ^ 23 + 1) := by ring
        _ ≤ _ := Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by norm_num) (by omega))
  · rw [heq]
    apply Nat.mul_le_mul_right
    have : a % 2 ^ 23 ≤ b % 2 ^ 23 := by omega
    omega

/-- **Word order is value order** on finite words with the sign bit clear. -/
theorem wordQ_mono {a b : ℕ} {x y : ℚ} (hab : a ≤ b) (hb : b < 2 ^ 31) (hx : wordQ a = some x)
    (hy : wordQ b = some y) : x ≤ y := by
  rw [PeelExactProofs.wordQ_fields (wordQ_exp hx)] at hx
  rw [PeelExactProofs.wordQ_fields (wordQ_exp hy)] at hy
  have hsa : (a >>> 31) % 2 = 0 := by rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt (by omega)]
  have hsb : (b >>> 31) % 2 = 0 := by rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt hb]
  simp only [hsa, hsb, zero_ne_one, ↓reduceIte, one_mul] at hx hy
  obtain rfl := Option.some.inj hx
  obtain rfl := Option.some.inj hy
  have e : ∀ n : ℕ, (2 : ℚ) ^ ((n : ℤ) - 150) = 2 ^ n * 2 ^ (-150 : ℤ) := fun n => by
    rw [sub_eq_add_neg, zpow_add₀ two_ne_zero, zpow_natCast]
  rw [e, e, ← mul_assoc, ← mul_assoc]
  apply mul_le_mul_of_nonneg_right _ (by positivity)
  exact_mod_cast fields_mono hab hb

/-- A finite word of positive value has sign bit `0`. -/
theorem lt31_of_pos {w : ℕ} {x : ℚ} (hw : w < 2 ^ 32) (h : wordQ w = some x) (hx : 0 < x) : w < 2 ^ 31 := by
  by_contra hlt
  have hs : (w >>> 31) % 2 = 1 := by rw [Nat.shiftRight_eq_div_pow]; omega
  rw [PeelExactProofs.wordQ_fields (wordQ_exp h)] at h
  simp only [hs, ↓reduceIte] at h
  obtain rfl := Option.some.inj h
  have : (0 : ℚ) ≤ (((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) : ℚ) *
      (2 : ℚ) ^ (((max ((w >>> 23) % 256) 1 : ℕ) : ℤ) - 150) := by positivity
  linarith

/-- A positive value's word that is not finite is `+inf`. -/
theorem f32Word_pos_inf {y : ℚ} (hy : 0 < y) (h : wordQ (f32Word y) = none) : f32Word y = 0x7F800000 := by
  have h31 := f32Word_lt31 hy.le
  have he := wordQ_none_exp h
  have hm := f32Word_notNaN y he
  rw [Nat.shiftRight_eq_div_pow] at he
  omega

/-- A value in `[2⁻¹²⁶, 2¹²⁷)`: its word is finite, positive and within `2⁻²⁴` of it. -/
theorem f32Word_pos {y : ℚ} (hy : 2 ^ (-126 : ℤ) ≤ y) (hy' : y < 2 ^ 127) :
    ∃ v, wordQ (f32Word y) = some v ∧ |v - y| ≤ y / 2 ^ 24 ∧ 0 < v := by
  have hy0 : 0 < y := lt_of_lt_of_le (two_zpow_pos _) hy
  obtain ⟨t, ht, herr⟩ := AtomErr.f32Word_spec (y := y) (by rw [abs_of_pos hy0]; exact hy')
  rw [abs_of_pos hy0] at herr
  have h150 : (2 : ℚ) ^ (-150 : ℤ) ≤ y / 2 ^ 24 := by
    have e : (2 : ℚ) ^ (-150 : ℤ) = 2 ^ (-126 : ℤ) / 2 ^ 24 := by norm_num
    rw [e]
    exact div_le_div_of_nonneg_right hy (by positivity)
  rw [max_eq_left h150] at herr
  refine ⟨_, wordQ_of_fin ht, herr, ?_⟩
  have h1 := (abs_le.1 herr).1
  have h2 : y / 2 ^ 24 < y := div_lt_self hy0 (by norm_num)
  linarith

/-- A rounding below `2¹²⁷` is finite. -/
theorem rnWord_finite {v : ℚ} (nz : Bool) (hv : |v| < 2 ^ 127) : ∃ r, wordQ (rnWord v nz) = some r := by
  unfold rnWord
  by_cases hv0 : v = 0
  · rw [ite_eq_left hv0]
    cases nz
    · exact ⟨0, Freeze.wordQ_zero⟩
    · exact ⟨0, PeelExactProofs.wordQ_negZero⟩
  · rw [ite_eq_right hv0]
    obtain ⟨t, ht, -⟩ := AtomErr.f32Word_spec hv
    exact ⟨_, wordQ_of_fin ht⟩

/-- A finite word's positive value is at least `2⁻¹⁴⁹`. -/
theorem wordQ_pos_ge {w : ℕ} {x : ℚ} (h : wordQ w = some x) (hx : 0 < x) : 2 ^ (-149 : ℤ) ≤ x := by
  obtain ⟨m, rfl⟩ := wordQ_grid h
  have hm : (0 : ℚ) < m := by
    by_contra h'
    have : (m : ℚ) * 2 ^ (-149 : ℤ) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg (not_lt.1 h') (two_zpow_pos _).le
    linarith
  have hm1 : (1 : ℚ) ≤ m := by exact_mod_cast (show (0 : ℤ) < m by exact_mod_cast hm)
  nlinarith [two_zpow_pos (-149 : ℤ)]

theorem wordQ_c4 : wordQ 0x40800000 = some 4 := by decide +kernel
theorem wordQ_c448 : wordQ 0x43E00000 = some 448 := by decide +kernel
theorem wordQ_c2688 : wordQ 0x45280000 = some 2688 := by decide +kernel
theorem wordQ_cFloor : wordQ 0x2F800000 = some (1 / 4294967296) := by decide +kernel
theorem wordQ_cm1 : wordQ 0xBF800000 = some (-1) := by decide +kernel
theorem wordQ_cLine : wordQ 0x3AB504F3 = some (11863283 / 8589934592) := by decide +kernel
theorem wordQ_cQuarter : wordQ 0x3E800000 = some (1 / 4) := by decide +kernel
theorem wordQ_cKbeta : wordQ 0x3EA920A7 = some (11083943 / 33554432) := by decide +kernel
theorem wordQ_pinf : wordQ 0x7F800000 = none := by decide +kernel

/-- A nonzero value's finite word is within one rounding of it. -/
theorem f32Word_err {y v : ℚ} (hy : y ≠ 0) (h : wordQ (f32Word y) = some v) :
    |v - y| ≤ |y| / 2 ^ 24 + 2 ^ (-150 : ℤ) :=
  PeelExactProofs.rnWord_err (nz := false) (by rw [rnWord, ite_eq_right hy]; exact h)

/-- `f32Add` of finite words rounds their sum. -/
theorem f32Add_fin {a b : ℕ} {x y : ℚ} (ha : wordQ a = some x) (hb : wordQ b = some y) :
    ∃ nz, f32Add a b = rnWord (x + y) nz := ⟨_, by unfold f32Add; rw [ha, hb]⟩

/-- `f32Add` of a finite word and `+inf` is `+inf`. -/
theorem f32Add_pinf {s : ℕ} {sv : ℚ} (hs : wordQ s = some sv) : f32Add s 0x7F800000 = 0x7F800000 := by
  obtain ⟨t, ht, -⟩ := PeelExactProofs.wordQ_fin hs
  have hi : fp32Value 0x7F800000 = .inf false := rfl
  unfold f32Add
  rw [hs, wordQ_pinf]
  dsimp only
  rw [ht, hi]
  rfl

/-- `PouwF32SqrtRn_v1`'s word: `+inf`, NaN, a zero, or the rounding of a non-negative value. -/
theorem sqrtWord_cases (a : ℕ) : sqrtWord a = 0x7F800000 ∨ sqrtWord a = 0x7FFFFFFF ∨
    (sqrtWord a % 2 ^ 31 = 0 ∧ sqrtWord a < 2 ^ 32) ∨ ∃ y : ℚ, 0 ≤ y ∧ sqrtWord a = f32Word y := by
  unfold sqrtWord
  dsimp only
  split_ifs <;> first
    | exact Or.inl ‹_›
    | exact Or.inr (Or.inl rfl)
    | exact Or.inr (Or.inr (Or.inl ⟨‹_›, Nat.mod_lt _ (by norm_num)⟩))
    | exact Or.inr (Or.inr (Or.inr ⟨_, by positivity, rfl⟩))

/-- `PouwF32SqrtRn_v1`'s finite words are non-negative. -/
theorem sqrtWord_nonneg {a : ℕ} {r : ℚ} (h : wordQ (sqrtWord a % 2 ^ 32) = some r) : 0 ≤ r := by
  have hnan : wordQ 0x7FFFFFFF = none := by decide +kernel
  rw [wordQ_mod32] at h
  rcases sqrtWord_cases a with h0 | h0 | ⟨h0, h32⟩ | ⟨y, hy, h0⟩
  · rw [h0, wordQ_pinf] at h; cases h
  · rw [h0, hnan] at h; cases h
  · rcases (by omega : sqrtWord a = 0 ∨ sqrtWord a = 0x80000000) with h0 | h0
    · rw [h0, Freeze.wordQ_zero] at h; cases h; exact le_rfl
    · rw [h0, PeelExactProofs.wordQ_negZero] at h; cases h; exact le_rfl
  · rw [h0] at h
    exact wordQ_nonneg (f32Word_lt31 hy) h

/-- `PouwF32SqrtRn_v1` of a word that is not finite is not finite (`+inf` or NaN). -/
theorem sqrtWord_nonfin {a : ℕ} (h : wordQ a = none) : wordQ (sqrtWord a % 2 ^ 32) = none := by
  have hnan : wordQ 0x7FFFFFFF = none := by decide +kernel
  rw [← wordQ_mod32] at h
  have he := wordQ_none_exp h
  unfold sqrtWord
  dsimp only
  rw [ite_eq_left he]
  split_ifs with hi
  · rw [Nat.mod_mod, wordQ_mod32]; rwa [wordQ_mod32] at h
  · exact hnan

/-! ## The primitives on words -/

section Prims

variable {ops : RowOps}

theorem eval_lt32 (q : Prim) (hq : q.w = 32) (l : List ℕ) : q.eval ops l < 2 ^ 32 := by
  rw [← Prim.eval_mod, hq]; exact Nat.mod_lt _ (by norm_num)

/-- A product of finite words that is zero is `±0`. -/
theorem mul_zero_val (hA : Pc8OpsAgree ops) {a b : ℕ} {x y : ℚ} (ha32 : a < 2 ^ 32) (hb32 : b < 2 ^ 32)
    (ha : wordQ a = some x) (hb : wordQ b = some y) (h0 : x * y = 0) : wordQ (Prim.eval ops .mul [a, b]) = some 0 := by
  change wordQ (ops.f32MulV2 a b % 2 ^ 32) = some 0
  by_cases hs : signBit a = signBit b
  · rw [hA.mul a b x y ha32 hb32 ha hb (fun _ => hs), h0]
    simpa [f32Word] using Freeze.wordQ_zero
  · rw [hA.mulZero a b x y ha32 hb32 ha hb h0 hs]
    exact PeelExactProofs.wordQ_negZero

/-- A product of finite words whose exact value is in `[2⁻¹²⁶, 2¹²⁷)`: finite, positive, below `2³¹`, and within `2⁻²⁴`
of the product. -/
theorem mul_pos_val (hA : Pc8OpsAgree ops) {a b : ℕ} {x y : ℚ} (ha32 : a < 2 ^ 32) (hb32 : b < 2 ^ 32)
    (ha : wordQ a = some x) (hb : wordQ b = some y) (hlo : 2 ^ (-126 : ℤ) ≤ x * y) (hhi : x * y < 2 ^ 127) :
    Prim.eval ops .mul [a, b] < 2 ^ 31 ∧ ∃ v, wordQ (Prim.eval ops .mul [a, b]) = some v ∧ |v - x * y| ≤ x * y / 2 ^ 24 ∧
      0 < v := by
  have hp : 0 < x * y := lt_of_lt_of_le (two_zpow_pos _) hlo
  change ops.f32MulV2 a b % 2 ^ 32 < 2 ^ 31 ∧ ∃ v, wordQ (ops.f32MulV2 a b % 2 ^ 32) = some v ∧ _
  rw [hA.mul a b x y ha32 hb32 ha hb (fun h => absurd h hp.ne')]
  exact ⟨f32Word_lt31 hp.le, f32Word_pos hlo hhi⟩

/-- `F32Fabs` of a finite word: finite, below `2³¹`. -/
theorem f32Fabs_word (hA : Pc8OpsAgree ops) {w : ℕ} (hf : (wordQ w).isSome) :
    ops.f32Fabs w < 2 ^ 31 ∧ (wordQ (ops.f32Fabs w)).isSome := by
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hf
  obtain ⟨h31, hq⟩ := fabs_word hx
  rw [hA.fabs]
  exact ⟨h31, by rw [hq]; rfl⟩

/-- **`s`, the running max of `|x|`, is a finite word below `2³¹`** on a row of finite words. -/
theorem rowMax_word (hA : Pc8OpsAgree ops) {x : ℕ → ℕ} (hx : ∀ l, (wordQ (x l)).isSome) :
    ∀ t, rowMax ops .fabs x t < 2 ^ 31 ∧ (wordQ (rowMax ops .fabs x t)).isSome
  | 0 => f32Fabs_word hA (hx 0)
  | t + 1 => by
    have ih := rowMax_word hA hx t
    have hf := f32Fabs_word hA (hx (t + 1))
    rw [rowMax]
    change ops.selectI32 (Prim.eval ops .le [rowMax ops .fabs x t, ops.f32Fabs (x (t + 1))])
        (ops.f32Fabs (x (t + 1))) (rowMax ops .fabs x t) % 2 ^ 32 < 2 ^ 31 ∧
      (wordQ (ops.selectI32 (Prim.eval ops .le [rowMax ops .fabs x t, ops.f32Fabs (x (t + 1))])
        (ops.f32Fabs (x (t + 1))) (rowMax ops .fabs x t) % 2 ^ 32)).isSome
    rw [hA.sel]
    split_ifs
    · rw [Nat.mod_eq_of_lt (by omega)]; exact hf
    · rw [Nat.mod_eq_of_lt (by omega)]; exact ih

/-- **`α = F32Div(q, F32Fmaxf(t, 2⁻³²))` is a step word** on every word `t`, for a positive `q` in `[4, 2¹²]`: `+0`
where `t` is `+inf`, else positive and within `2⁻²⁴` of `q/d` for a finite `d` at least `2⁻³²` and `t`'s value. -/
theorem alpha_spec (hA : Pc8OpsAgree ops) {t Q : ℕ} {q : ℚ} (ht : t < 2 ^ 32) (hQ : wordQ Q = some q)
    (hQ31 : Q < 2 ^ 31) (hq4 : 4 ≤ q) (hq : q ≤ 2 ^ 12) :
    StepWord (Prim.eval ops .div [Q, Prim.eval ops .fmaxf [t, 0x2F800000]]) ∧
      ∃ a, wordQ (Prim.eval ops .div [Q, Prim.eval ops .fmaxf [t, 0x2F800000]]) = some a ∧ 0 ≤ a ∧
        (t = 0x7F800000 → a = 0) ∧
        (t ≠ 0x7F800000 → 0 < a ∧ ∃ dv, 2 ^ (-32 : ℤ) ≤ dv ∧ (∀ tv, wordQ t = some tv → tv ≤ dv) ∧
          |a - q / dv| ≤ q / dv / 2 ^ 24) := by
  have hQ32 : Q < 2 ^ 32 := by omega
  have hd : Prim.eval ops .fmaxf [t, 0x2F800000] =
      if t = 0x7F800000 ∨ ∃ x, wordQ t = some x ∧ 1 / 4294967296 < x then t else 0x2F800000 :=
    hA.fmaxfAny t 0x2F800000 (1 / 4294967296) ht (by norm_num) wordQ_cFloor (by norm_num)
  rw [hd]
  by_cases hinf : t = 0x7F800000
  · rw [ite_eq_left (Or.inl hinf), hinf]
    have h0 : Prim.eval ops .div [Q, 0x7F800000] = 0 := by
      change ops.f32DivV2 Q 0x7F800000 % 2 ^ 32 = 0
      rw [hA.divInf Q hQ32 (by rw [hQ]; rfl), signBit_of_lt hQ31]
      rfl
    rw [h0]
    exact ⟨Fp4Freeze.stepWord_zero, 0, Fp4Freeze.wordQ_zero, le_rfl, fun _ => rfl, fun h => absurd rfl h⟩
  obtain ⟨d, hd32, dv, hdv, hdv32, hdt, hdeq⟩ : ∃ d : ℕ, d < 2 ^ 32 ∧ ∃ dv, wordQ d = some dv ∧ 2 ^ (-32 : ℤ) ≤ dv ∧
      (∀ tv, wordQ t = some tv → tv ≤ dv) ∧
      (if t = 0x7F800000 ∨ ∃ x, wordQ t = some x ∧ 1 / 4294967296 < x then t else 0x2F800000) = d := by
    by_cases hgt : ∃ x, wordQ t = some x ∧ 1 / 4294967296 < x
    · obtain ⟨x, hx, hx'⟩ := hgt
      refine ⟨t, ht, x, hx, by norm_num at hx' ⊢; exact hx'.le, fun tv htv => ?_, ite_eq_left (Or.inr ⟨x, hx, hx'⟩)⟩
      rw [hx] at htv; cases htv; exact le_rfl
    · refine ⟨0x2F800000, by norm_num, 1 / 4294967296, wordQ_cFloor, by norm_num, fun tv htv => ?_, ?_⟩
      · by_contra h
        exact hgt ⟨tv, htv, not_le.1 h⟩
      · refine ite_eq_right ?_
        rintro (h | h)
        · exact hinf h
        · exact hgt h
  rw [hdeq]
  have hdpos : 0 < dv := lt_of_lt_of_le (two_zpow_pos _) hdv32
  have hdlt : dv < 2 ^ 128 := lt_of_abs_lt (wordQ_abs_lt hdv)
  have hα : Prim.eval ops .div [Q, d] = f32Word (q / dv) :=
    hA.div Q d q dv hQ32 hd32 hQ hdv hdpos.ne' (fun h => absurd h (by linarith))
  rw [hα]
  have hlo : 2 ^ (-126 : ℤ) ≤ q / dv := by
    rw [le_div_iff₀ hdpos]
    calc (2 : ℚ) ^ (-126 : ℤ) * dv ≤ 2 ^ (-126 : ℤ) * 2 ^ 128 := by gcongr
      _ = 4 := by norm_num
      _ ≤ q := hq4
  have hhi : q / dv < 2 ^ 127 := by
    rw [div_lt_iff₀ hdpos]
    calc q ≤ 2 ^ 12 := hq
      _ < 2 ^ 127 * 2 ^ (-32 : ℤ) := by norm_num
      _ ≤ 2 ^ 127 * dv := by gcongr
  obtain ⟨v, hv, herr, hvpos⟩ := f32Word_pos hlo hhi
  refine ⟨stepWord_of_lt31 (f32Word_lt31 (div_pos (by linarith) hdpos).le) hv, v, hv, hvpos.le,
    fun h => absurd h hinf, fun _ => ⟨hvpos, dv, hdv32, hdt, herr⟩⟩

/-- The floor `F32Fma(ρ, α, −1)` of finite words, where `ρ·α − 1` is below `2¹²⁷`: finite, within one rounding. -/
theorem floor_val (hA : Pc8OpsAgree ops) {ρw αw : ℕ} {r a : ℚ} (hρ32 : ρw < 2 ^ 32) (hα32 : αw < 2 ^ 32)
    (hρ : wordQ ρw = some r) (hα : wordQ αw = some a) (hv : |r * a - 1| < 2 ^ 127) :
    ∃ fl, wordQ (Prim.eval ops .fma [ρw, αw, 0xBF800000]) = some fl ∧
      |fl - (r * a - 1)| ≤ |r * a - 1| / 2 ^ 24 + 2 ^ (-150 : ℤ) := by
  have e : r * a + -1 = r * a - 1 := by ring
  generalize hnz : (decide (r * a = 0) && decide (-1 = (0 : ℚ)) && (signBit ρw != signBit αw) &&
    signBit 0xBF800000) = nz
  obtain ⟨fl, hfl⟩ := rnWord_finite (v := r * a + -1) nz (by rw [e]; exact hv)
  have hfm : fmaWord? ρw αw 0xBF800000 = some (rnWord (r * a + -1) nz) := by
    simp only [fmaWord?, hρ, hα, wordQ_cm1, hnz]
    exact finite?_of_wordQ hfl
  show ∃ fl, wordQ (ops.f32FmaV2 ρw αw 0xBF800000 % 2 ^ 32) = some fl ∧ _
  rw [hA.fma _ _ _ _ hρ32 hα32 (by norm_num) hfm]
  refine ⟨fl, hfl, ?_⟩
  have := PeelExactProofs.rnWord_err hfl
  rwa [e] at this

/-- A nonnegative value's word: `+inf`, or finite, nonnegative and within one rounding. -/
theorem f32Word_nonneg_cases {y : ℚ} (hy : 0 ≤ y) : f32Word y < 2 ^ 31 ∧ (f32Word y = 0x7F800000 ∨
    ∃ v, wordQ (f32Word y) = some v ∧ 0 ≤ v ∧ y - y / 2 ^ 24 - 2 ^ (-150 : ℤ) ≤ v) := by
  refine ⟨f32Word_lt31 hy, ?_⟩
  rcases hy.eq_or_lt with h0 | hpos
  · rw [← h0]
    right
    refine ⟨0, by simpa [f32Word] using Freeze.wordQ_zero, le_rfl, ?_⟩
    have := two_zpow_pos (-150 : ℤ)
    linarith
  rcases hq : wordQ (f32Word y) with _ | v
  · exact Or.inl (f32Word_pos_inf hpos hq)
  · right
    have herr := f32Word_err hpos.ne' hq
    rw [abs_of_pos hpos] at herr
    exact ⟨v, rfl, wordQ_nonneg (f32Word_lt31 hy) hq, by linarith [(abs_le.1 herr).1]⟩

/-- A product of finite nonnegative values: `+inf`, or finite and nonnegative. -/
theorem mul_nonneg_cases (hA : Pc8OpsAgree ops) {a b : ℕ} {x y : ℚ} (ha32 : a < 2 ^ 32) (hb32 : b < 2 ^ 32)
    (ha : wordQ a = some x) (hb : wordQ b = some y) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    Prim.eval ops .mul [a, b] = 0x7F800000 ∨ ∃ v, wordQ (Prim.eval ops .mul [a, b]) = some v ∧ 0 ≤ v := by
  rcases (mul_nonneg hx hy).eq_or_lt with h0 | hpos
  · exact Or.inr ⟨0, mul_zero_val hA ha32 hb32 ha hb h0.symm, le_rfl⟩
  have hm : Prim.eval ops .mul [a, b] = f32Word (x * y) :=
    hA.mul a b x y ha32 hb32 ha hb (fun h => absurd h hpos.ne')
  rw [hm]
  rcases (f32Word_nonneg_cases hpos.le).2 with h | ⟨v, hv, hv0, -⟩
  · exact Or.inl h
  · exact Or.inr ⟨v, hv, hv0⟩

/-- **The running max is the largest of its words**, under `I32Le`'s order on words with the sign bit clear: it is one
of the words read, and at least each. -/
theorem rowMax_ge (hA : Pc4OpsAgree ops) {w : Wire} {x : ℕ → ℕ} :
    ∀ t, (∀ l ≤ t, w.eval ops (x l) < 2 ^ 31) →
      (∃ l ≤ t, rowMax ops w x t = w.eval ops (x l)) ∧ ∀ l ≤ t, w.eval ops (x l) ≤ rowMax ops w x t
  | 0, _ => ⟨⟨0, le_rfl, rfl⟩, fun l hl => by rw [Nat.le_zero.1 hl]; rfl⟩
  | t + 1, hv => by
    obtain ⟨⟨l0, hl0, h0⟩, hge⟩ := rowMax_ge hA t fun l hl => hv l (by omega)
    have hm : rowMax ops w x t < 2 ^ 31 := h0 ▸ hv l0 (by omega)
    have hf := hv (t + 1) le_rfl
    have hr : rowMax ops w x (t + 1) = max (rowMax ops w x t) (w.eval ops (x (t + 1))) := by
      rw [rowMax]
      change ops.selectI32 (Prim.eval ops .le [rowMax ops w x t, w.eval ops (x (t + 1))]) (w.eval ops (x (t + 1)))
        (rowMax ops w x t) % 2 ^ 32 = _
      have hl : Prim.eval ops .le [rowMax ops w x t, w.eval ops (x (t + 1))] =
          ops.i32Le (rowMax ops w x t) (w.eval ops (x (t + 1))) % 2 := rfl
      rw [hA.sel, hl, Nat.mod_mod, hA.le _ _ hm hf]
      by_cases h : rowMax ops w x t ≤ w.eval ops (x (t + 1))
      · rw [ite_eq_left h, ite_eq_left rfl, Nat.mod_eq_of_lt (by omega), max_eq_right h]
      · rw [ite_eq_right h, ite_eq_right (by norm_num), Nat.mod_eq_of_lt (by omega), max_eq_left (by omega)]
    rw [hr]
    refine ⟨?_, fun l hl => ?_⟩
    · rcases le_total (rowMax ops w x t) (w.eval ops (x (t + 1))) with h | h
      · exact ⟨t + 1, le_rfl, max_eq_right h⟩
      · exact ⟨l0, by omega, by rw [max_eq_left h, h0]⟩
    · rcases Nat.lt_or_ge l (t + 1) with h | h
      · exact (hge l (by omega)).trans (le_max_left _ _)
      · rw [show l = t + 1 by omega]; exact le_max_right _ _

end Prims

/-! ## `pc8` -/

section Pc8

variable {ops : RowOps}

/-- A row of finite words re-encoded (`encRow`) is a row of finite words. -/
theorem encRow_words {k : ℕ} {w : ℕ → ℕ} (hw : ∀ l < k, (wordQ (w l)).isSome) :
    ∀ l, encRow k (rowQ k w) l < 2 ^ 32 ∧ (wordQ (encRow k (rowQ k w) l)).isSome := by
  intro l
  unfold encRow
  split_ifs with hl
  · obtain ⟨h32, -, hf⟩ := (rowQ_zeroSign hw ⟨l, hl⟩).2
    obtain ⟨-, x, hx⟩ := PeelExactProofs.finite?_eq_some hf
    exact ⟨h32, by rw [hx]; rfl⟩
  · exact ⟨by norm_num, by rw [Freeze.wordQ_zero]; rfl⟩

/-- **`Stats8OK` by magnitudes**: on a row of finite words whose `ρ` is finite and whose floor `F32Fma(ρ, α, −1)` is at
least `0`, `α` and `β` are step words and `β > 0`. -/
theorem stats8OK_of_rho (hA : Pc8OpsAgree ops) {k : ℕ} {w : ℕ → ℕ} (hw : ∀ l < k, (wordQ (w l)).isSome)
    (hρ : (wordQ (rowRho ops .id (encRow k (rowQ k w)) k)).isSome)
    (hfl : 0 ≤ PearlC.wordVal (floor8 ops (encRow k (rowQ k w)) k)) : Stats8OK ops k w := by
  have hx := encRow_words hw
  unfold Stats8OK
  unfold floor8 at hfl
  unfold beta8
  generalize encRow k (rowQ k w) = x at hx hρ hfl ⊢
  obtain ⟨r, hr⟩ := Option.isSome_iff_exists.1 hρ
  have hρ32 : rowRho ops .id x k < 2 ^ 32 := eval_lt32 .sqrt rfl _
  have hr0 : 0 ≤ r :=
    sqrtWord_nonneg (a := Prim.eval ops .div [sumSq ops .id x (k / 8), f32Word ((k / 8 : ℕ) : ℚ)]) hr
  obtain ⟨hs31, hsS⟩ := rowMax_word hA (fun l => (hx l).2) (k - 1)
  obtain ⟨sv, hsv⟩ := Option.isSome_iff_exists.1 hsS
  have hsv0 : 0 ≤ sv := wordQ_nonneg hs31 hsv
  obtain ⟨hαS, a, ha, ha0, hinf, hfin⟩ := alpha_spec hA (t := Prim.eval ops .add [rowMax ops .fabs x (k - 1),
    Prim.eval ops .mul [0x40800000, rowRho ops .id x k]]) (eval_lt32 .add rfl _) wordQ_c448 (by norm_num)
    (by norm_num) (by norm_num)
  change StepWord (alpha8 ops x k) at hαS
  change wordQ (alpha8 ops x k) = some a at ha
  have hα32 : alpha8 ops x k < 2 ^ 32 := hαS.1
  have hra0 : 0 ≤ r * a := mul_nonneg hr0 ha0
  have h127 : (114 : ℚ) ≤ 2 ^ 127 := by norm_num
  have key : |r * a - 1| < 2 ^ 127 → 1 / 2 ≤ r * a := fun hv => by
    obtain ⟨fl, hflq, herr⟩ := floor_val hA hρ32 hα32 hr ha hv
    rw [wordVal_of_wordQ hflq] at hfl
    have h150 : (2 : ℚ) ^ (-150 : ℤ) ≤ 1 / 4 := by norm_num
    by_contra hlt
    rw [abs_of_neg (by linarith : r * a - 1 < 0)] at herr
    have := (abs_le.1 herr).2
    linarith
  have hbd : r * a ≤ 113 := by
    rcases hr0.eq_or_lt with h0 | hrpos
    · rw [← h0, zero_mul]; norm_num
    by_cases htinf : Prim.eval ops .add [rowMax ops .fabs x (k - 1),
        Prim.eval ops .mul [0x40800000, rowRho ops .id x k]] = 0x7F800000
    · rw [hinf htinf, mul_zero]; norm_num
    obtain ⟨-, dv, hdv32, hdt, herrα⟩ := hfin htinf
    have h4r : (0 : ℚ) < 4 * r := by linarith
    have hm4 : Prim.eval ops .mul [0x40800000, rowRho ops .id x k] = f32Word (4 * r) :=
      hA.mul 0x40800000 (rowRho ops .id x k) 4 r (by norm_num) hρ32 wordQ_c4 hr (fun h => absurd h h4r.ne')
    rw [hm4] at htinf hdt
    rcases hq4 : wordQ (f32Word (4 * r)) with _ | m4v
    · refine absurd ?_ htinf
      change f32Add (rowMax ops .fabs x (k - 1)) (f32Word (4 * r)) % 2 ^ 32 = 0x7F800000
      rw [f32Word_pos_inf h4r hq4, f32Add_pinf hsv]
      all_goals norm_num
    have herr4 := f32Word_err h4r.ne' hq4
    rw [abs_of_pos h4r] at herr4
    have hE := two_zpow_pos (-150)
    have hE2 : (2 : ℚ) ^ (-149 : ℤ) = 2 * 2 ^ (-150 : ℤ) := by norm_num
    have hr149 := wordQ_pos_ge hr hrpos
    have hm4pos : 0 < m4v := by
      have := (abs_le.1 herr4).1
      rw [hE2] at hr149
      generalize (2 : ℚ) ^ (-150 : ℤ) = E at this hr149 hE
      linarith
    have hsum : 0 < sv + m4v := by linarith
    obtain ⟨nz, hadd⟩ := f32Add_fin hsv hq4
    have htw : Prim.eval ops .add [rowMax ops .fabs x (k - 1), f32Word (4 * r)] = f32Word (sv + m4v) := by
      change f32Add (rowMax ops .fabs x (k - 1)) (f32Word (4 * r)) % 2 ^ 32 = _
      have := f32Word_lt31 hsum.le
      rw [hadd, rnWord, ite_eq_right hsum.ne', Nat.mod_eq_of_lt (by omega)]
    rw [htw] at htinf hdt
    rcases hqt : wordQ (f32Word (sv + m4v)) with _ | tv
    · exact absurd (f32Word_pos_inf hsum hqt) htinf
    have herrt := f32Word_err hsum.ne' hqt
    rw [abs_of_pos hsum] at herrt
    have htv := hdt tv hqt
    have hE32 : (2 : ℚ) ^ (-150 : ℤ) ≤ 2 ^ (-32 : ℤ) / 1000000 := by norm_num
    have hrle : r ≤ dv * (1 / 4 + 1 / 1048576) := by
      have h1 := (abs_le.1 herr4).1
      have h2 := (abs_le.1 herrt).1
      generalize (2 : ℚ) ^ (-150 : ℤ) = E at h1 h2 hE hE32
      generalize (2 : ℚ) ^ (-32 : ℤ) = D at hE32 hdv32
      linarith
    have hdv0 : 0 < dv := lt_of_lt_of_le (two_zpow_pos _) hdv32
    have hale : a ≤ 448 / dv * (1 + 1 / 16777216) := by
      have := (abs_le.1 herrα).2
      linarith
    calc r * a ≤ (dv * (1 / 4 + 1 / 1048576)) * (448 / dv * (1 + 1 / 16777216)) :=
          mul_le_mul hrle hale ha0 (by positivity)
      _ = (1 / 4 + 1 / 1048576) * (1 + 1 / 16777216) * 448 := by field_simp
      _ ≤ 113 := by norm_num
  have hra : 1 / 2 ≤ r * a := key (abs_lt.2 ⟨by linarith, by linarith⟩)
  obtain ⟨hmb31, v1, hv1, herr1, -⟩ := mul_pos_val hA hρ32 hα32 hr ha
    (le_trans (by norm_num) hra) (by linarith)
  have hv1lo : 1 / 4 ≤ v1 := by
    have := (abs_le.1 herr1).1
    linarith
  have hv1hi : v1 ≤ 114 := by
    have := (abs_le.1 herr1).2
    linarith
  obtain ⟨hβ31, bv, hbv, -, hbpos⟩ := mul_pos_val hA (by omega) (by norm_num) hv1 wordQ_cLine
    (calc (2 : ℚ) ^ (-126 : ℤ) ≤ 1 / 4 * (11863283 / 8589934592) := by norm_num
      _ ≤ v1 * (11863283 / 8589934592) := by gcongr)
    (calc v1 * (11863283 / 8589934592) ≤ 114 * (11863283 / 8589934592) := by gcongr
      _ < 2 ^ 127 := by norm_num)
  exact ⟨hαS, stepWord_of_lt31 hβ31 hbv, by rw [wordVal_of_wordQ hbv]; exact hbpos⟩

end Pc8

/-! ## `pc4` -/

section Pc4

variable {ops : RowOps}

/-- **`Row4OK`'s `α` and `β` by magnitudes**: on any row whose `β` word is finite with `0 ≤ β` and `e4m3(β)` at least
code `8` (`RowAdmit4`'s first two clauses at `pc4Sem`'s `β`), `α` and `β` are step words and `α > 0`. -/
theorem row4OK_stats_of_beta (hA : Pc8OpsAgree ops) {k : ℕ} {x : ℕ → ℕ} (hβ : (wordQ (beta4 ops x k)).isSome)
    (h0 : 0 ≤ PearlC.wordVal (beta4 ops x k)) (h8 : 8 ≤ (e4m3 (PearlC.wordVal (beta4 ops x k))).val) :
    StepWord (alpha4 ops x k) ∧ 0 < PearlC.wordVal (alpha4 ops x k) ∧ StepWord (beta4 ops x k) := by
  obtain ⟨hαS, a, ha, -, hinf, hfin⟩ := alpha_spec hA (t := Prim.eval ops .add [rowMax ops .bf16Fabs x (k - 1),
    Prim.eval ops .mul [0x40800000, drho4 ops x k]]) (eval_lt32 .add rfl _) wordQ_c2688 (by norm_num)
    (by norm_num) (by norm_num)
  change StepWord (alpha4 ops x k) at hαS
  change wordQ (alpha4 ops x k) = some a at ha
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hβ
  rw [wordVal_of_wordQ hb] at h0 h8
  have hb0 : b ≠ 0 := by
    rintro rfl
    have : (e4m3 0).val = 0 := by decide +kernel
    omega
  have hbpos : 0 < b := lt_of_le_of_ne h0 (Ne.symm hb0)
  have hapos : 0 < a := by
    by_cases ht : Prim.eval ops .add [rowMax ops .bf16Fabs x (k - 1),
        Prim.eval ops .mul [0x40800000, drho4 ops x k]] = 0x7F800000
    · exfalso
      obtain rfl := hinf ht
      have hd32 : drho4 ops x k < 2 ^ 32 := eval_lt32 .mul rfl _
      have hm32 : Prim.eval ops .mul [drho4 ops x k, alpha4 ops x k] < 2 ^ 32 := eval_lt32 .mul rfl _
      rcases hd : wordQ (drho4 ops x k) with _ | d
      · have hm : wordQ (Prim.eval ops .mul [drho4 ops x k, alpha4 ops x k]) = none :=
          hA.mulNonfin (drho4 ops x k) (alpha4 ops x k) hd32 hd (by rw [ha]; rfl)
        have hn : wordQ (beta4 ops x k) = none :=
          hA.mulNonfin _ 0x3EA920A7 hm32 hm (by rw [wordQ_cKbeta]; rfl)
        rw [hb] at hn
        cases hn
      · have hm := mul_zero_val hA hd32 hαS.1 hd ha (mul_zero d)
        have hz : wordQ (beta4 ops x k) = some 0 := mul_zero_val hA hm32 (by norm_num) hm wordQ_cKbeta (zero_mul _)
        exact hb0 (Option.some.inj (hb.symm.trans hz))
    · exact (hfin ht).1
  exact ⟨hαS, by rw [wordVal_of_wordQ ha]; exact hapos,
    stepWord_of_lt31 (lt31_of_pos (eval_lt32 .mul rfl _) hb hbpos) hb⟩

/-- **`α·|x_l| ≤ 2689`** on a row that `Row4OK` admits, under `I32Le`'s order: `s` is the largest `|x_l|`
(`rowMax_ge`), `ρ` is finite since `β` is, so `δρ ≥ 0` and `t = s + 4·δρ` is `+inf` (`α = +0`) or finite and at least
`s` up to one rounding, while `α ≤ 2688/max(t, 2⁻³²)` up to one. -/
theorem alpha4_mul_le (hA : Pc4OpsAgree ops) {K : ℕ} {X : Fin K → ℚ} (hR : Row4OK ops K X) (l : Fin K) : wordVal (alpha4 ops (encRow16 K X) K) * |X l| ≤ 2689 := by
  have hK : 1 ≤ K := by have := l.isLt; omega
  have hxw : ∀ l' (h : l' < K), wordQ (ops.bf16ToF32 (encRow16 K X l')) = some (X ⟨l', h⟩) := by
    intro l' h
    rw [encRow16, dite_eq_left h]
    obtain ⟨v, hv⟩ := stepWord_wordQ (hR.widen ⟨l', h⟩).1
    rw [hv, ← wordVal_of_wordQ hv, (hR.widen ⟨l', h⟩).2]
  have hvw : ∀ l' (h : l' < K), Wire.eval ops .bf16Fabs (encRow16 K X l') < 2 ^ 31 ∧
      wordQ (Wire.eval ops .bf16Fabs (encRow16 K X l')) = some |X ⟨l', h⟩| := by
    intro l' h
    show ops.f32Fabs _ < 2 ^ 31 ∧ wordQ (ops.f32Fabs _) = some _
    rw [hA.fabs]
    exact fabs_word (hxw l' h)
  obtain ⟨⟨l0, hl0, hs0⟩, hge⟩ := rowMax_ge hA (w := .bf16Fabs) (x := encRow16 K X) (K - 1)
    fun l' hl' => (hvw l' (by omega)).1
  set s := rowMax ops .bf16Fabs (encRow16 K X) (K - 1) with hs
  have hs31 : s < 2 ^ 31 := by rw [hs0]; exact (hvw l0 (by omega)).1
  have hsv : wordQ s = some |X ⟨l0, by omega⟩| := by rw [hs0]; exact (hvw l0 (by omega)).2
  set sv := |X ⟨l0, by omega⟩|
  have hsv0 : 0 ≤ sv := abs_nonneg _
  have hxl : |X l| ≤ sv := wordQ_mono (hge l (by omega)) hs31 (hvw l l.isLt).2 hsv
  obtain ⟨αv, hαv⟩ := stepWord_wordQ hR.alpha
  obtain ⟨βv, hβv⟩ := stepWord_wordQ hR.beta
  have hρ32 : rowRho ops .bf16 (encRow16 K X) K < 2 ^ 32 := eval_lt32 .sqrt rfl _
  have hd32 : drho4 ops (encRow16 K X) K < 2 ^ 32 := eval_lt32 .mul rfl _
  rcases hρ : wordQ (rowRho ops .bf16 (encRow16 K X) K) with _ | r
  · exfalso
    have hd : wordQ (drho4 ops (encRow16 K X) K) = none :=
      hA.mulNonfin _ 0x3E800000 hρ32 hρ (by rw [wordQ_cQuarter]; rfl)
    have hm : wordQ (Prim.eval ops .mul [drho4 ops (encRow16 K X) K, alpha4 ops (encRow16 K X) K]) = none :=
      hA.mulNonfin (drho4 ops (encRow16 K X) K) (alpha4 ops (encRow16 K X) K) hd32 hd (by rw [hαv]; rfl)
    have hn : wordQ (beta4 ops (encRow16 K X) K) = none :=
      hA.mulNonfin _ 0x3EA920A7 (eval_lt32 .mul rfl _) hm (by rw [wordQ_cKbeta]; rfl)
    rw [hβv] at hn
    cases hn
  have hr0 : 0 ≤ r := sqrtWord_nonneg (a := Prim.eval ops .div [sumSq ops .bf16 (encRow16 K X) (K / 8),
    f32Word ((K / 8 : ℕ) : ℚ)]) hρ
  have hrlt : r < 2 ^ 128 := lt_of_abs_lt (wordQ_abs_lt hρ)
  obtain ⟨d, hd, hd0⟩ : ∃ d, wordQ (drho4 ops (encRow16 K X) K) = some d ∧ 0 ≤ d := by
    rcases (mul_nonneg_cases hA.toPc8OpsAgree hρ32 (by norm_num) hρ wordQ_cQuarter hr0 (by norm_num)) with h | h
    · exfalso
      have hpos : 0 < r * (1 / 4) := by
        rcases hr0.eq_or_lt with h0 | h0
        · have := mul_zero_val hA.toPc8OpsAgree hρ32 (by norm_num) hρ wordQ_cQuarter (by rw [← h0]; ring)
          change wordQ (Prim.eval ops .mul [_, _]) = _ at this
          rw [h, wordQ_pinf] at this
          cases this
        · positivity
      have hm : Prim.eval ops .mul [rowRho ops .bf16 (encRow16 K X) K, 0x3E800000] = f32Word (r * (1 / 4)) :=
        hA.mul _ _ r (1 / 4) hρ32 (by norm_num) hρ wordQ_cQuarter (fun h => absurd h hpos.ne')
      obtain ⟨q, hq⟩ := rnWord_finite (v := r * (1 / 4)) false (by rw [abs_of_pos hpos]; linarith)
      rw [rnWord, ite_eq_right hpos.ne', ← hm, h, wordQ_pinf] at hq
      cases hq
    · exact h
  have hm32 : Prim.eval ops .mul [0x40800000, drho4 ops (encRow16 K X) K] < 2 ^ 32 := eval_lt32 .mul rfl _
  set t := Prim.eval ops .add [s, Prim.eval ops .mul [0x40800000, drho4 ops (encRow16 K X) K]] with ht
  have htc : t = 0x7F800000 ∨ ∃ tv, wordQ t = some tv ∧ sv - sv / 2 ^ 24 - 2 ^ (-150 : ℤ) ≤ tv := by
    rcases mul_nonneg_cases hA.toPc8OpsAgree (by norm_num) hd32 wordQ_c4 hd (by norm_num) hd0 with hinf | ⟨mv, hmv, hmv0⟩
    · left
      rw [ht, hinf]
      change f32Add s 0x7F800000 % 2 ^ 32 = _
      rw [f32Add_pinf hsv]
      norm_num
    obtain ⟨nz, hadd⟩ := f32Add_fin hsv hmv
    have htw : t = rnWord (sv + mv) nz % 2 ^ 32 := by
      rw [ht]
      change f32Add s (Prim.eval ops .mul [0x40800000, drho4 ops (encRow16 K X) K]) % 2 ^ 32 = _
      rw [hadd]
    rcases (add_nonneg hsv0 hmv0).eq_or_lt with h0 | hpos
    · right
      refine ⟨0, ?_, ?_⟩
      · rw [htw, rnWord, ite_eq_left h0.symm]
        cases nz
        · exact Freeze.wordQ_zero
        · exact PeelExactProofs.wordQ_negZero
      · have : sv = 0 := by linarith
        rw [this]
        have := two_zpow_pos (-150 : ℤ)
        linarith
    rw [htw, rnWord, ite_eq_right hpos.ne', Nat.mod_eq_of_lt (by have := f32Word_lt31 hpos.le; omega)]
    rcases (f32Word_nonneg_cases hpos.le).2 with h | ⟨tv, htv, -, herr⟩
    · exact Or.inl h
    · right
      refine ⟨tv, htv, ?_⟩
      have : sv - sv / 2 ^ 24 ≤ (sv + mv) - (sv + mv) / 2 ^ 24 := by linarith
      linarith
  obtain ⟨-, a, ha, ha0, hinf, hfin⟩ := alpha_spec hA.toPc8OpsAgree (t := t) (eval_lt32 .add rfl _) wordQ_c2688 (by norm_num)
    (by norm_num) (by norm_num)
  change wordQ (alpha4 ops (encRow16 K X) K) = some a at ha
  rw [wordVal_of_wordQ ha]
  by_cases hti : t = 0x7F800000
  · rw [hinf hti, zero_mul]; norm_num
  obtain ⟨-, dv, hdv32, hdt, herrα⟩ := hfin hti
  obtain ⟨tv, htv, htvl⟩ := htc.resolve_left hti
  have htd := hdt tv htv
  have hdv0 : 0 < dv := lt_of_lt_of_le (two_zpow_pos _) hdv32
  have hE : (2 : ℚ) ^ (-150 : ℤ) ≤ 2 ^ (-32 : ℤ) / 2 ^ 100 := by norm_num
  have hx1 : |X l| ≤ dv * (1 + 1 / 8388608) := by
    have h2 : (2 : ℚ) ^ (-32 : ℤ) / 2 ^ 100 ≤ dv / 2 ^ 100 := div_le_div_of_nonneg_right hdv32 (by positivity)
    generalize (2 : ℚ) ^ (-150 : ℤ) = E at hE htvl
    generalize (2 : ℚ) ^ (-32 : ℤ) / 2 ^ 100 = D at hE h2
    linarith
  have hale : a ≤ 2688 / dv * (1 + 1 / 16777216) := by
    have := (abs_le.1 herrα).2
    linarith
  calc a * |X l| ≤ (2688 / dv * (1 + 1 / 16777216)) * (dv * (1 + 1 / 8388608)) :=
        mul_le_mul hale hx1 (abs_nonneg _) (by positivity)
    _ = 2688 * (1 + 1 / 16777216) * (1 + 1 / 8388608) := by field_simp
    _ ≤ 2689 := by norm_num

end Pc4

/-! ## The squares-overflow row -/

section Overflow

variable {ops : RowOps}

/-- A butterfly round over `n` sums that are all `+inf` gives `+inf` sums. -/
theorem bfly_pinf {h n : ℕ} {v : ℕ → ℕ} (hv : ∀ j < n, v j = 0x7F800000) :
    ∀ j < n, bfly ops h n v j = 0x7F800000 := by
  intro j hj
  have h1 : f32Add 0x7F800000 0x7F800000 = 0x7F800000 := by decide +kernel
  have h2 : f32Add 0x7F800000 0 = 0x7F800000 := by decide +kernel
  show f32Add (v j) (if j + h < n then v (j + h) else 0) % 2 ^ 32 = _
  rw [hv j hj]
  split_ifs with hjh
  · rw [hv _ hjh, h1]; rfl
  · rw [h2]; rfl

/-- The butterfly's sum of `n ≥ 1` partials that are all `+inf` is `+inf`. -/
theorem sumSq_pinf {w : Wire} {x : ℕ → ℕ} {n : ℕ} (hn : 0 < n)
    (hv : ∀ j < n, fmaAcc ops w x (accLast n j) = 0x7F800000) : sumSq ops w x n = 0x7F800000 :=
  bfly_pinf (bfly_pinf (bfly_pinf (bfly_pinf (bfly_pinf hv)))) 0 hn

/-- Where the sum of squares is `+inf`, `ρ`'s word is not finite. -/
theorem rowRho_nonfin (hA : Pc8OpsAgree ops) {w : Wire} {x : ℕ → ℕ} {K : ℕ} (hn : 0 < K / 8) (hn' : K / 8 < 2 ^ 127)
    (hs : sumSq ops w x (K / 8) = 0x7F800000) : wordQ (rowRho ops w x K) = none := by
  have h1 : (1 : ℚ) ≤ ((K / 8 : ℕ) : ℚ) := by exact_mod_cast hn
  obtain ⟨v, hv, -, -⟩ := f32Word_pos (y := ((K / 8 : ℕ) : ℚ)) (le_trans (by norm_num) h1) (by exact_mod_cast hn')
  have hd := hA.divNonfin 0x7F800000 (f32Word ((K / 8 : ℕ) : ℚ)) (by norm_num) wordQ_pinf (by rw [hv]; rfl)
  unfold rowRho
  rw [hs]
  exact sqrtWord_nonfin hd

/-- On a row whose every 8th word, read through `w`, is `2¹²⁶` (`0x7E800000`), each of at most 32 partials is
`fma(2¹²⁶, 2¹²⁶, +0)`, which overflows to `+inf`. -/
theorem fmaAcc_pinf (hA : Pc8OpsAgree ops) {w : Wire} {x : ℕ → ℕ} {n : ℕ} (hn : n ≤ 32)
    (hx : ∀ t < n, w.eval ops (x (8 * t)) = 0x7E800000) : ∀ j < n, fmaAcc ops w x (accLast n j) = 0x7F800000 := by
  intro j hj
  have hl : accLast n j = j := by
    unfold accLast
    rw [Nat.div_eq_of_lt (by omega : n - 1 - j < 32)]
    rfl
  have hq : wordQ 0x7E800000 = some ((2 : ℚ) ^ 126) := by decide +kernel
  have ho : f32Word ((2 : ℚ) ^ 126 * 2 ^ 126 + 0) = 0x7F800000 := by decide +kernel
  rw [hl, fmaAcc, ite_eq_left (by omega : j < 32)]
  show ops.f32FmaV2 (w.eval ops (x (8 * j))) (w.eval ops (x (8 * j))) 0 % 2 ^ 32 = _
  rw [hx j hj, hA.fmaOvf _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) hq hq Freeze.wordQ_zero
    (by rw [ho]; exact wordQ_pinf), ho]

/-- `ρ`'s word on a squares-overflow row of `8 ≤ K ≤ 256` words (every 8th one `2¹²⁶` through `w`) is not finite. -/
theorem rowRho_squaresOverflow (hA : Pc8OpsAgree ops) {w : Wire} {x : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K) (hK' : K ≤ 256)
    (hx : ∀ t < K / 8, w.eval ops (x (8 * t)) = 0x7E800000) : wordQ (rowRho ops w x K) = none :=
  rowRho_nonfin hA (by omega) (by omega) (sumSq_pinf (by omega) (fmaAcc_pinf hA (by omega) hx))

/-- **`pc8Sem` refuses the squares-overflow row** (`test_circuit_pc8`'s `squares-overflow`, which the circuit and Python
refuse): row `i` of a committed operand of `8 ≤ K ≤ 256` columns whose every 8th word is `2¹²⁶` (`0x7E800000`). Each
sampled square overflows, so `ρ`'s word is `+inf` and the floor is `0`. -/
theorem pc8Sem_squaresOverflow (hA : Pc8OpsAgree ops) {Q R S : Type} (p : Pipeline)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r)
    {m K i : ℕ} {W : Codes} (hK : 8 ≤ K) (hK' : K ≤ 256) (hi : i < m)
    (hW : ∀ t, 8 * t < K → W i (8 * t) = 0x7E800000) : ¬ (pc8Sem ops p noise).RowOK m K W i := by
  intro h
  have hf := (h.2 hi).2
  have hρ : wordQ (rowRho ops .id (encRow K (wordsVal m K W ⟨i, hi⟩)) K) = none := by
    refine rowRho_squaresOverflow hA hK hK' fun t ht => ?_
    have h8 : 8 * t < K := by omega
    show encRow K (wordsVal m K W ⟨i, hi⟩) (8 * t) = 0x7E800000
    rw [encRow, dite_eq_left h8]
    show f32Word ((wordQ (W i (8 * t))).getD 0) = _
    rw [hW t h8]
    decide +kernel
  simp only [pc8Sem, hρ, Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at hf
  norm_num at hf

/-- **`pc4Sem` refuses the squares-overflow row** (`test_circuit_pc4`'s `squares-overflow`, which the circuit and
Python refuse): row `i` of a committed operand of `8 ≤ K ≤ 256` columns whose every 8th word is `2¹²⁶` (`0x7E800000`,
the widened BF16 word `0x7E80`). `ρ` is `+inf`, so `δρ` and `β` are not finite, `pc4Sem`'s `β` is `0` and `RowAdmit4`
fails (`e4m3(0)` is code `0`). -/
theorem pc4Sem_squaresOverflow (hA : Pc4OpsAgree ops) {Q R S : Type}
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k)
    {m K i : ℕ} {W : Codes} (hK : 8 ≤ K) (hK' : K ≤ 256) (hi : i < m)
    (hW : ∀ t, 8 * t < K → W i (8 * t) = 0x7E800000) : ¬ (pc4Sem ops lines).RowOK m K W i := by
  intro h
  obtain ⟨-, h8, -⟩ := h.2 hi
  have hρ : wordQ (rowRho ops .bf16 (encRow16 K (wordsVal m K W ⟨i, hi⟩)) K) = none := by
    refine rowRho_squaresOverflow hA.toPc8OpsAgree hK hK' fun t ht => ?_
    have h8 : 8 * t < K := by omega
    have hb : encRow16 K (wordsVal m K W ⟨i, hi⟩) (8 * t) = 0x7E80 := by
      rw [encRow16, dite_eq_left h8]
      show bf16WordOf ((wordQ (W i (8 * t))).getD 0) = _
      rw [hW t h8]
      decide +kernel
    show ops.bf16ToF32 (encRow16 K (wordsVal m K W ⟨i, hi⟩) (8 * t)) = _
    rw [hb, hA.bf16w _ (by norm_num)]
    rfl
  simp only [pc4Sem] at h8
  generalize encRow16 K (wordsVal m K W ⟨i, hi⟩) = x at hρ h8
  have hd : wordQ (drho4 ops x K) = none :=
    hA.mulNonfin _ 0x3E800000 (eval_lt32 (ops := ops) .sqrt rfl _) hρ (by rw [wordQ_cQuarter]; rfl)
  obtain ⟨-, a, ha, -⟩ := alpha_spec hA.toPc8OpsAgree (t := Prim.eval ops .add [rowMax ops .bf16Fabs x (K - 1),
    Prim.eval ops .mul [0x40800000, drho4 ops x K]]) (eval_lt32 .add rfl _) wordQ_c2688 (by norm_num)
    (by norm_num) (by norm_num)
  have ha' : wordQ (alpha4 ops x K) = some a := ha
  have hm : wordQ (Prim.eval ops .mul [drho4 ops x K, alpha4 ops x K]) = none :=
    hA.mulNonfin _ _ (eval_lt32 .mul rfl _) hd (by show (wordQ (alpha4 ops x K)).isSome = true; rw [ha']; rfl)
  have hβ : wordQ (beta4 ops x K) = none :=
    hA.mulNonfin _ 0x3EA920A7 (eval_lt32 .mul rfl _) hm (by rw [wordQ_cKbeta]; rfl)
  have h0 : (e4m3 0).val = 0 := by decide +kernel
  simp only [hβ, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, h0] at h8
  omega

/-- **`Fp4Sem.RowOK` refuses a word that is not a widened BF16 word** (`test_pouw_pearl_c4`'s
`test_row_passes_asks_finite_widened_bf16_words`, which Python's `row_passes` refuses): `0x3F800001` (`1 + 2⁻²³`) is
finite, but its low 16 bits are not zero. -/
theorem rowOK4_notBf16 {Q R S : Type} (sem : Fp4Sem Q R S) {m K i l : ℕ} {W : Codes} (hl : l < K)
    (hW : W i l = 0x3F800001) : ¬ sem.RowOK m K W i := by
  intro h
  have h16 := (h.1 l hl).2
  rw [hW] at h16
  exact absurd h16 (by decide)

/-- **`Fp4Sem.RowOK` refuses a non-finite word** (the same test's `+inf`, `0x7F800000`). -/
theorem rowOK4_nonFinite {Q R S : Type} (sem : Fp4Sem Q R S) {m K i l : ℕ} {W : Codes} (hl : l < K)
    (hW : W i l = 0x7F800000) : ¬ sem.RowOK m K W i := by
  intro h
  have hf := (h.1 l hl).1
  rw [hW] at hf
  exact absurd hf (by decide +kernel)

/-- **`PearlCSem.RowOK` refuses a non-finite word** (`test_pouw_pearl_c`'s weight row with `+inf`, `0x7F800000`,
which `check_tile` refuses). -/
theorem rowOK8_nonFinite {Q R S : Type} (sem : PearlCSem Q R S) {m K i l : ℕ} {W : Codes} (hl : l < K)
    (hW : W i l = 0x7F800000) : ¬ sem.RowOK m K W i := by
  intro h
  have hf := h.1 l hl
  rw [hW] at hf
  exact absurd hf (by decide +kernel)

/-- **`PearlCSem.RowOK` refuses a row that is not live** (`test_pouw_pearl_c`'s weight row, which `check_tile` and
`live_row` refuse): of `128` words `1.0`, the three at `1`, `2` and `3` are `8.0`. The every-8th positions are all
`1.0`, so `ρ = 1`, and three entries reach `8ρ`, more than `128/64`. -/
theorem rowOK8_notLive {Q R S : Type} (sem : PearlCSem Q R S) {m i : ℕ} {W : Codes} (hi : i < m)
    (hW : ∀ l, W i l = if l = 1 ∨ l = 2 ∨ l = 3 then 0x41000000 else 0x3F800000) : ¬ sem.RowOK m 128 W i := by
  intro h
  have hl := (h.2 hi).1
  have hx : wordsVal m 128 W ⟨i, hi⟩ = fun l : Fin 128 => if l.val = 1 ∨ l.val = 2 ∨ l.val = 3 then 8 else 1 := by
    funext l
    simp only [wordsVal, hW]
    split_ifs
    · decide +kernel
    · decide +kernel
  rw [hx] at hl
  revert hl
  unfold LiveRow Stat.every8 Params.pi
  decide +kernel

end Overflow

end Pouw.PearlC.RowCircuit
