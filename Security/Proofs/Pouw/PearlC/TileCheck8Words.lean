import Proofs.Pouw.PearlC.TileCheck8
import Proofs.Pouw.PearlC.TileProtocol
import Proofs.Pouw.PearlC.TileGamma

/-!
# `pc8`'s tile check computes the words (proofs)

The row units' words read in the protocol's terms (`pc8Sem`), under the named hypotheses on which the table and the
protocol's word functions agree only conditionally:

* `Pc8OpsAgree ops`: core's primitives (`F32ToE4m3Sat`, `E4m3ToF32`, `F32Mul`, `F32Fma`, `F32Div`, `F32Fmaxf`,
  `F32Neg`, `F2fpBf16`, `F32Fabs`, `SelectI32`) agree with the protocol's word functions where both are defined. At
  core's instance `coreOps` they follow from core's definitions (`CoreOps.pc8OpsAgree_coreOps`);
* `Row8OK`: a row's words are finite and its `α` and `β` step words, with `β > 0` (`Stats8OK`). It is no hypothesis
  of L3: a credited row's follows from `PearlCSem.RowOK`, and a weight's from the check's own `weightOK`
  (`TileCheck.Accepts`), by magnitudes (`RowStats`, `TileCheckWords`). The table reads a `−0` word, and a word's bits
  above 32, as `pc8Sem` reads them (`ZeroSign`): `|−0| = |+0|`, `fma(−0, −0, ·) = fma(+0, +0, ·)`, and the step reads
  `α·(−0)` and `α·(+0)` alike (`alpha8_zeroSign`, `beta8_zeroSign`, `form8_zeroSign`);
* `Noise8OK`: no code of the noise is NaN (`Line8OK` at every line of its four factors). It is no hypothesis of L3:
  the check refuses a noise with a NaN code (`tileCheck8`'s `anchorsOK`), as Python's verifier does (`fp8_to_f32`
  refuses one), so an accepted tile's noise has it (`TileCheckWords`). On the other codes, `−0` included, the table's `E′` is `lineCodes`'s (`eTilde8_line`: `fp32.mul(β, −0)` is
  `−0`), never NaN (`lineCodes_notNaN`), and its `bf16(E′)` is the protocol's `codeBf16 E′` (`be8_word`).

On them the formed words are `CastWord`s (`formWord8_castWord`), so the cast is `e4m3` of the formed value: the step
writes a NaN only from a NaN and never writes `−0`, and its accumulator `α·x_t` (a product of step words), `E′` and the
basis' codes are not NaN. A row unit's codes are `pc8Sem`'s (`quant8`, so `quant8A`/`quant8B` are `unitChain`'s
`A′`/`B̃`) and its factor row is the peel's (`row8A_paWords`, `row8B_pbWords`). `entry8_words` chains RowA `i` and RowB
`j` into their tile (`TileCheck8`'s `chain8_*`): where the three units are correct and the peel's factor rows exist,
entry `(i, j)`'s `C̃` and `U` are the device's ticket and useful words (`tile_devAt`).

On the whole table a gate holds its assignment's low bits at its width (`val_in8`, `val_tile8`), so where the public
gates hold the anchors (`anchors8`) and the registered gates `rows8`'s values, an entry's registered words are the
device's (`entry8_accepts`), at two `Row8OK` rows and the named hypothesis at the entry (`Pc8EntryOK`: the peel's
factor rows exist). `TileCheckWords` composes these into L3's words for the plain `pc8` Program on a domain of whole-tile
shapes (`Shape8`). The Program's tile at a unit's shape is the audit grid's there (`tileSide`: `64`, or the side itself
below 64, so decode's `32 × 64`), and whole-tile means that it divides the shape: each of `m` and `n` at most 64 or a
multiple of 64 (`tileSide_dvd_iff`). `D_dev` (`pearlCDomainDev`) gives `Shape8`'s `k` but leaves `m` and `n` free
(`shape8_of_pearlCDomainDev`). A last tile narrower than the others (`m = 80`) is outside the Program. The cap is not
part of it: its verdict is a separate conjunct, decided by units of its own.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit Pouw.PearlC.PeelExact
  Pouw.PearlC.PeelExactP Matrix

/-! ## Words -/

section Words

theorem shr31_mod32 (w : ℕ) : ((w % 2 ^ 32) >>> 31) % 2 = (w >>> 31) % 2 := by
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]; omega

theorem wordQ_mod32 (w : ℕ) : wordQ (w % 2 ^ 32) = wordQ w := by
  have h1 : ((w % 2 ^ 32) >>> 23) % 256 = (w >>> 23) % 256 := by
    simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]; omega
  have h2 : w % 2 ^ 32 % 2 ^ 23 = w % 2 ^ 23 := by simp only [Nat.reducePow]; omega
  unfold wordQ fp32Value
  simp only [h1, h2, shr31_mod32]

theorem signBit_mod32 (w : ℕ) : signBit (w % 2 ^ 32) = signBit w := by
  unfold signBit
  rw [shr31_mod32]

theorem fmaWord?_mod32 (a b c : ℕ) : fmaWord? a (b % 2 ^ 32) c = fmaWord? a b c := by
  unfold fmaWord?
  rw [wordQ_mod32, signBit_mod32]

/-- A code's FP32 word has the code's value. -/
theorem wordQ_f32Word_code : ∀ c : Code, c.isNaN = false → wordQ (f32Word (codeVal c)) = some (codeVal c) := by
  decide +kernel

/-- A code's FP32 word narrows to BF16 exactly: its top half. -/
theorem rnBf16_f32Word_code :
    ∀ c : Code, c.isNaN = false → rnBf16? (f32Word (codeVal c)) = some (f32Word (codeVal c) >>> 16) := by
  decide +kernel

/-- A code's FP32 word is its top half widened back. -/
theorem bf16_shift_code : ∀ c : Code, c.isNaN = false → (f32Word (codeVal c) >>> 16) <<< 16 = f32Word (codeVal c) := by
  decide +kernel

theorem rnWord_lt (v : ℚ) (b : Bool) : rnWord v b < 2 ^ 32 := by
  unfold rnWord
  split_ifs
  · norm_num
  · norm_num
  · exact f32Word_lt v

theorem finite?_eq {w s : ℕ} (h : finite? w = some s) : s = w := by
  unfold finite? at h
  split_ifs at h
  exact (Option.some.inj h).symm

theorem addWord?_lt {a b s : ℕ} (h : addWord? a b = some s) : s < 2 ^ 32 := by
  unfold addWord? at h
  split at h
  · rw [finite?_eq h]; exact rnWord_lt _ _
  · simp at h

theorem promoteWord?_lt {t w s : ℕ} (h : promoteWord? t w = some s) : s < 2 ^ 32 := by
  unfold promoteWord? at h
  obtain ⟨s', hs', rfl⟩ := Option.map_eq_some_iff.1 h
  split_ifs
  · norm_num
  · exact addWord?_lt hs'

theorem foldlM_lt {α : Type} (f : ℕ → α → Option ℕ) (hf : ∀ t x s, f t x = some s → s < 2 ^ 32) :
    ∀ (l : List α) (t c : ℕ), t < 2 ^ 32 → l.foldlM f t = some c → c < 2 ^ 32
  | [], t, c, ht, h => by
    simp only [List.foldlM_nil, pure, Option.some.injEq] at h
    omega
  | x :: l, t, c, _, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.1 h
    exact foldlM_lt f hf l s c (hf t x s hs) h

/-- A chain's word is 32-bit. -/
theorem chainWordP?_lt {p : Pipeline} {G : ℕ} {atoms : List AtomIn} {c : ℕ} (h : chainWordP? p G atoms = some c) :
    c < 2 ^ 32 := by
  unfold chainWordP? at h
  refine foldlM_lt _ (fun t g s hs => ?_) _ 0 c (by norm_num) h
  obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 hs
  exact promoteWord?_lt hw

theorem stepWord_wordQ {w : ℕ} (hw : AccumProofs.StepWord w) : wordQ w = some (wordVal w) := by
  obtain ⟨y, hy⟩ : ∃ y, wordQ w = some y := ⟨_, PeelExactProofs.wordQ_fields (AccumProofs.finite?_exp hw.2.2)⟩
  simp [wordVal, hy]

theorem signBit_of_pos {w : ℕ} {y : ℚ} (h : wordQ w = some y) (hy : 0 < y) : signBit w = false := by
  have := (wordQ_neg_iff h (ne_of_gt hy)).not.1 (not_lt.2 hy.le)
  unfold signBit
  simpa using this

/-- A row word `v` the table reads against the word `u` `pc8Sem` reads for its value: the same word, or `−0` where
`pc8Sem` writes `+0`. -/
def ZeroSign (v u : ℕ) : Prop := v = u ∨ (v = 0x80000000 ∧ u = 0)

/-- **A finite 32-bit word is its value's `f32Word`, or `−0`**, and that `f32Word` is a step word. -/
theorem zeroSign_of_wordQ {w : ℕ} {x : ℚ} (hw : w < 2 ^ 32) (h : wordQ w = some x) :
    ZeroSign w (f32Word x) ∧ AccumProofs.StepWord (f32Word x) := by
  have hz : f32Word 0 = 0 := by simp [f32Word]
  by_cases h0 : w = 0x80000000
  · subst h0
    obtain rfl : x = 0 := (Option.some.inj (PeelExactProofs.wordQ_negZero.symm.trans h)).symm
    rw [hz]
    exact ⟨Or.inr ⟨rfl, rfl⟩, ⟨by norm_num, by norm_num, by decide⟩⟩
  · have hs : AccumProofs.StepWord w := ⟨hw, h0, by unfold finite?; rw [ite_eq_right (wordQ_exp_ne h)]⟩
    have hv : wordVal w = x := by simp [wordVal, h]
    rw [← hv, f32Word_wordVal hs]
    exact ⟨Or.inl rfl, hs⟩

end Words

/-! ## The named hypotheses -/

/-- **Core's primitives agree with the protocol's word functions** where both are defined: the E4M3 cast is
`e4m3Word`; the widening of a code other than NaN and `−0` is its value's FP32 word; FP32 multiply and FMA round as
`f32Word`, `fmaWord?` (a zero product of equal signs is `+0`, of opposite signs `−0`); negation flips the sign bit;
the BF16 narrowing is `rnBf16?`; the widening of `−0` is FP32's `−0`; `F32Fabs` clears the sign bit; the FMA of
`−0` by `−0` is that of `+0` by `+0` on every word, the infinities and NaNs included (both products are `+0`); FP32
divide by a nonzero word rounds as `f32Word` (an exact zero signed as `F32Mul`'s), and a finite word over `+inf` is the
zero of its sign; a product whose first factor is not finite is not finite; `F32Fmaxf` against a finite nonzero word
is the larger, a NaN losing; `SelectI32` selects on its condition's low bit; an FMA of finite words whose exact value
overflows writes its `f32Word` (`±inf`); and a quotient of a non-finite word by a finite one is not finite. Each is a
fact about IEEE binary32 (round to nearest even, gradual underflow) or core's integer primitives. At core's instance
`coreOps` all of them follow from core's definitions (`CoreOps.pc8OpsAgree_coreOps`). -/
structure Pc8OpsAgree (ops : RowOps) : Prop where
  cast : ∀ u, ops.f32ToE4m3Sat u = (e4m3Word u).val
  widen : ∀ c : Code, c.isNaN = false → c.val ≠ 0x80 → ops.e4m3ToF32 c.val % 2 ^ 32 = f32Word (codeVal c)
  mul : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y →
    (x * y = 0 → signBit a = signBit b) → ops.f32MulV2 a b % 2 ^ 32 = f32Word (x * y)
  mulZero : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y → x * y = 0 →
    signBit a ≠ signBit b → ops.f32MulV2 a b % 2 ^ 32 = 0x80000000
  fma : ∀ a b c w : ℕ, a < 2 ^ 32 → b < 2 ^ 32 → c < 2 ^ 32 → fmaWord? a b c = some w →
    ops.f32FmaV2 a b c % 2 ^ 32 = w
  neg : ∀ w < 2 ^ 32, ops.f32Neg w % 2 ^ 32 = negWord w
  bf16 : ∀ w x : ℕ, w < 2 ^ 32 → rnBf16? w = some x → ops.f2fpBf16 w % 2 ^ 16 = x
  widenNeg0 : ops.e4m3ToF32 0x80 % 2 ^ 32 = 0x80000000
  fabs : ∀ w, ops.f32Fabs w = w % 2 ^ 31
  fmaSq0 : ∀ c < 2 ^ 32, ops.f32FmaV2 0x80000000 0x80000000 c = ops.f32FmaV2 0 0 c
  div : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y → y ≠ 0 →
    (x = 0 → signBit a = signBit b) → ops.f32DivV2 a b % 2 ^ 32 = f32Word (x / y)
  divInf : ∀ a < 2 ^ 32, (wordQ a).isSome → ops.f32DivV2 a 0x7F800000 % 2 ^ 32 = if signBit a then 0x80000000 else 0
  mulNonfin : ∀ a b : ℕ, a < 2 ^ 32 → wordQ a = none → (wordQ b).isSome → wordQ (ops.f32MulV2 a b % 2 ^ 32) = none
  fmaxfAny : ∀ a b : ℕ, ∀ y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ b = some y → y ≠ 0 →
    ops.f32Fmaxf a b % 2 ^ 32 = if a = 0x7F800000 ∨ ∃ x, wordQ a = some x ∧ y < x then a else b
  sel : ∀ c a b, ops.selectI32 c a b = if c % 2 = 1 then a else b
  fmaOvf : ∀ a b c : ℕ, ∀ x y z : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → c < 2 ^ 32 → wordQ a = some x → wordQ b = some y →
    wordQ c = some z → wordQ (f32Word (x * y + z)) = none → ops.f32FmaV2 a b c % 2 ^ 32 = f32Word (x * y + z)
  divNonfin : ∀ a b : ℕ, a < 2 ^ 32 → wordQ a = none → (wordQ b).isSome → wordQ (ops.f32DivV2 a b % 2 ^ 32) = none

/-- A row of words as values, as `wordsVal` reads it. -/
def rowQ (k : ℕ) (w : ℕ → ℕ) : Fin k → ℚ := fun l => (wordQ (w l)).getD 0

/-- Row `t` of a row of finite words, as the table's gate holds it (its low 32 bits), against `pc8Sem`'s word. -/
theorem rowQ_zeroSign {K : ℕ} {w : ℕ → ℕ} (hw : ∀ l < K, (wordQ (w l)).isSome) (t : Fin K) :
    ZeroSign (w t % 2 ^ 32) (f32Word (rowQ K w t)) ∧ AccumProofs.StepWord (f32Word (rowQ K w t)) := by
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 (hw t t.isLt)
  rw [show rowQ K w t = x by simp [rowQ, hx]]
  exact zeroSign_of_wordQ (Nat.mod_lt _ (by norm_num)) (by rw [wordQ_mod32, hx])

theorem encRow_zeroSign {K : ℕ} {w : ℕ → ℕ} (hw : ∀ l < K, (wordQ (w l)).isSome) :
    ∀ l < K, ZeroSign (w l % 2 ^ 32) (encRow K (rowQ K w) l) := fun l hl => by
  simpa only [encRow, hl, ↓reduceDIte] using (rowQ_zeroSign hw ⟨l, hl⟩).1

/-- A `pc8` row's `α` and `β` as the table reads them: step words, and `β > 0`. -/
def Stats8OK (ops : RowOps) (k : ℕ) (w : ℕ → ℕ) : Prop :=
  AccumProofs.StepWord (alpha8 ops (encRow k (rowQ k w)) k) ∧ AccumProofs.StepWord (beta8 ops (encRow k (rowQ k w)) k) ∧
    0 < wordVal (beta8 ops (encRow k (rowQ k w)) k)

/-- A `pc8` row the table reads as `pc8Sem` does: its words finite (`PearlCSem.RowOK`'s and `OperandOK`'s first
clause; a `−0` word and the bits above 32 read as `pc8Sem` reads them, `alpha8_zeroSign`, `form8_zeroSign`), and
`Stats8OK`. -/
def Row8OK (ops : RowOps) (k : ℕ) (w : ℕ → ℕ) : Prop := (∀ l < k, (wordQ (w l)).isSome) ∧ Stats8OK ops k w

/-- The formed word `pc8Form` reads as a value. -/
def formWord8 (ops : RowOps) (p : Pipeline) (α x β : ℚ) (e f : Fin Params.pi.r → Code) : ℕ :=
  form8 ops p (f32Word α) (f32Word x) (fun l => eTilde8 ops (f32Word β) (encCodes e l)) (encCodes f)

/-- `e4m3Word` of an `f32Word` is never NaN: `f32Word` never writes a NaN, and the cast saturates the rest. -/
theorem e4m3Word_f32Word_notNaN (y : ℚ) : (e4m3Word (f32Word y)).isNaN = false := by
  have hN := f32Word_notNaN y
  unfold e4m3Word
  dsimp only
  split
  next x _ =>
    have := e4m3Mag_le (abs_nonneg x)
    simp only [Code.isNaN, beq_eq_false_iff_ne, ne_eq]
    split_ifs <;> omega
  next hq =>
    have h255 : (f32Word y >>> 23) % 256 = 255 := by
      by_contra h
      rw [PeelExactProofs.wordQ_fields h] at hq
      exact absurd hq (by simp)
    rw [ite_eq_left (hN h255)]
    simp only [Code.isNaN, beq_eq_false_iff_ne, ne_eq]
    split_ifs <;> omega

/-- `E′` is never NaN. -/
theorem lineCodes_notNaN (β : ℚ) (e : Fin Params.pi.r → Code) (q : Fin Params.pi.r) :
    (lineCodes β e q).isNaN = false := by
  unfold lineCodes mulCode
  split_ifs
  · decide +kernel
  · decide +kernel
  · exact e4m3Word_f32Word_notNaN _

/-- A zero code other than `−0` and the NaNs is `+0`, of sign `false`. -/
theorem code_sign_of_zero : ∀ c : Code, c.isNaN = false → c.val ≠ 0x80 → codeVal c = 0 → c.sign = false := by
  decide +kernel

/-- `codeBf16` is `bf16WordOf` of the code's value on every code but NaN and `−0`. -/
theorem codeBf16_eq : ∀ c : Code, c.isNaN = false → c.val ≠ 0x80 → codeBf16 c = bf16WordOf (codeVal c) := by
  decide +kernel

theorem codeBf16_lt : ∀ c : Code, codeBf16 c < 2 ^ 16 := by decide +kernel

theorem codeBf16_shift_lt : ∀ c : Code, codeBf16 c <<< 16 < 2 ^ 32 := by decide +kernel

/-- A code's BF16 word widened to FP32 narrows back to it. -/
theorem rnBf16_codeBf16 : ∀ c : Code, c.isNaN = false → rnBf16? (codeBf16 c <<< 16) = some (codeBf16 c) := by
  decide +kernel

/-- `−0`, `+0`, `1.0` and `−448` widen to `0x8000`, `0x0000`, `0x3F80` and `0xC3E0`
(`test_bf16_of_code_keeps_the_sign_of_minus_zero`). -/
theorem codeBf16_examples :
    [codeBf16 0x80, codeBf16 0, codeBf16 0x38, codeBf16 0xFE] = [0x8000, 0, 0x3F80, 0xC3E0] := by
  decide +kernel

/-- The codes of `E′ = e4m3(fp32.mul(β, e))` at five `β` words (`1.0`, `2^−10·√2`, `−1.0`, `+0`, `3.0`) on ten noise
codes, `±0` among them: a zero product takes the XOR of the signs, so `−0` keeps its sign at `β > 0`
(`test_eprime_keeps_the_sign_of_a_zero_noise_code`). -/
theorem lineCodes_examples :
    let es : List Code := [0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE, 0x08, 0x88]
    let row := fun (b : ℕ) => es.map fun c => (lineCodes (wordVal b) (fun _ : Fin 1 => c) 0).val
    [row 0x3F800000, row 0x3AB504F3, row 0xBF800000, row 0x00000000, row 0x40400000] =
      [[0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE, 0x08, 0x88],
       [0x00, 0x80, 0x00, 0x80, 0x01, 0x81, 0x32, 0xB2, 0x00, 0x80],
       [0x80, 0x00, 0x81, 0x01, 0xB8, 0x38, 0xFE, 0x7E, 0x88, 0x08],
       [0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80],
       [0x00, 0x80, 0x03, 0x83, 0x44, 0xC4, 0x7E, 0xFE, 0x14, 0x94]] := by
  decide +kernel

/-! ## A row's words in the protocol's terms -/

section WordCongr

variable {ops : RowOps}

/-- `rowMax` reads a word only through its wiring `ab`. -/
theorem rowMax_congrW {ab : Wire} {x y : ℕ → ℕ} :
    ∀ t, (∀ l ≤ t, ab.eval ops (x l) = ab.eval ops (y l)) → rowMax ops ab x t = rowMax ops ab y t
  | 0, h => by simp only [rowMax, h 0 le_rfl]
  | t + 1, h => by simp only [rowMax, rowMax_congrW t fun l hl => h l (by omega), h (t + 1) le_rfl]

theorem fmaAcc_lt (w : Wire) (x : ℕ → ℕ) (i : ℕ) : fmaAcc ops w x i < 2 ^ 32 := by
  rw [fmaAcc.eq_1]
  simp only [Prim.eval, Prim.w]
  exact Nat.mod_lt _ (by norm_num)

/-- `fmaAcc` reads a word only through `fma(x, x, ·)` on an accumulator's word. -/
theorem fmaAcc_congrW {w : Wire} {x y : ℕ → ℕ} (i : ℕ)
    (h : ∀ l ≤ 8 * i, ∀ c < 2 ^ 32, Prim.eval ops .fma [w.eval ops (x l), w.eval ops (x l), c] =
      Prim.eval ops .fma [w.eval ops (y l), w.eval ops (y l), c]) :
    fmaAcc ops w x i = fmaAcc ops w y i := by
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    rw [fmaAcc.eq_1 ops w x i, fmaAcc.eq_1 ops w y i]
    by_cases h32 : i < 32
    · simp only [h32, ↓reduceIte]
      exact h _ le_rfl 0 (by norm_num)
    · simp only [h32, ↓reduceIte]
      rw [ih (i - 32) (by omega) fun l hl => h l (by omega)]
      exact h _ le_rfl _ (fmaAcc_lt w y _)

theorem rowRho_congrW {w : Wire} {x y : ℕ → ℕ} {K : ℕ} (hK : 8 ≤ K)
    (h : ∀ l < K, ∀ c < 2 ^ 32, Prim.eval ops .fma [w.eval ops (x l), w.eval ops (x l), c] =
      Prim.eval ops .fma [w.eval ops (y l), w.eval ops (y l), c]) :
    rowRho ops w x K = rowRho ops w y K := by
  have hA : ∀ j < K / 8, fmaAcc ops w x (accLast (K / 8) j) = fmaAcc ops w y (accLast (K / 8) j) := by
    intro j hj
    have := accLast_lt hj
    exact fmaAcc_congrW _ fun l hl => h l (by omega)
  unfold rowRho sumSq
  rw [bfly_agree (bfly_agree (bfly_agree (bfly_agree (bfly_agree hA)))) 0 (by omega)]

end WordCongr

section Row8

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

theorem fabs_zeroSign {v u : ℕ} (h : ZeroSign v u) : Wire.eval ops .fabs v = Wire.eval ops .fabs u := by
  rcases h with rfl | ⟨rfl, rfl⟩
  · rfl
  · simp only [Wire.eval, hops.fabs]
    norm_num

theorem fmaSq_zeroSign {v u : ℕ} (h : ZeroSign v u) {c : ℕ} (hc : c < 2 ^ 32) :
    Prim.eval ops .fma [Wire.eval ops .id v, Wire.eval ops .id v, c] =
      Prim.eval ops .fma [Wire.eval ops .id u, Wire.eval ops .id u, c] := by
  rcases h with rfl | ⟨rfl, rfl⟩
  · rfl
  · simp only [Wire.eval, Prim.eval, List.getD_cons_zero, List.getD_cons_succ, hops.fmaSq0 c hc]

/-- **`α` reads no row word's sign of zero**: `|−0| = |+0|`, and `fma(−0, −0, ·) = fma(+0, +0, ·)`. -/
theorem alpha8_zeroSign {K : ℕ} (hK : 8 ≤ K) {V U : ℕ → ℕ} (h : ∀ l < K, ZeroSign (V l) (U l)) :
    alpha8 ops V K = alpha8 ops U K := by
  unfold alpha8
  rw [rowRho_congrW hK fun l hl _ hc => fmaSq_zeroSign hops (h l hl) hc,
    rowMax_congrW (K - 1) fun l hl => fabs_zeroSign hops (h l (by omega))]

theorem beta8_zeroSign {K : ℕ} (hK : 8 ≤ K) {V U : ℕ → ℕ} (h : ∀ l < K, ZeroSign (V l) (U l)) :
    beta8 ops V K = beta8 ops U K := by
  unfold beta8
  rw [rowRho_congrW hK fun l hl _ hc => fmaSq_zeroSign hops (h l hl) hc, alpha8_zeroSign hops hK h]

/-- `α·(±0)`, `α` a step word, reads as the accumulator `+0`: it is `f32Word 0` or `−0` (`mul`, `mulZero`). -/
theorem mul_zero8 {a v : ℕ} (ha : AccumProofs.StepWord a) (hv : v = 0 ∨ v = 0x80000000) :
    fp32Value (Prim.eval ops .mul [a, v]) = .fin Term.zero := by
  have hqa := stepWord_wordQ ha
  have hqv : wordQ v = some 0 := by
    rcases hv with rfl | rfl
    · exact Freeze.wordQ_zero
    · exact PeelExactProofs.wordQ_negZero
  have hvl : v < 2 ^ 32 := by rcases hv with rfl | rfl <;> norm_num
  simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
  by_cases hs : signBit a = signBit v
  · rw [hops.mul _ _ _ _ ha.1 hvl hqa hqv fun _ => hs, mul_zero, show f32Word 0 = 0 by simp [f32Word]]
    rfl
  · rw [hops.mulZero _ _ _ _ ha.1 hvl hqa hqv (mul_zero _) hs]
    rfl

/-- **The forming reads no row word's sign of zero**: the step reads `α·(−0)` and `α·(+0)` alike. -/
theorem form8_zeroSign (p : Pipeline) {a v u : ℕ} (ha : AccumProofs.StepWord a) (h : ZeroSign v u) (et f : ℕ → ℕ) :
    form8 ops p a v et f = form8 ops p a u et f := by
  rcases h with rfl | ⟨rfl, rfl⟩
  · rfl
  · unfold form8
    simp only [List.cons_append, atom8_eval]
    unfold Pipeline.step
    rw [mul_zero8 hops ha (Or.inr rfl), mul_zero8 hops ha (Or.inl rfl)]

/-- **A rescaled noise code is `lineCodes`'s**: `eTilde8` at `β`'s step word and a noise code other than NaN, `−0`
included (`fp32.mul(β, −0)` is `−0` at `β > 0`). -/
theorem eTilde8_line {b : ℕ} (hb : AccumProofs.StepWord b) (hpos : 0 < wordVal b) {e : Fin Params.pi.r → Code}
    (q : Fin Params.pi.r) (hn : (e q).isNaN = false) :
    eTilde8 ops b (encCodes e q) = (lineCodes (wordVal b) e q).val := by
  have hβ := stepWord_wordQ hb
  have hsb := signBit_of_pos hβ hpos
  have hd : decide (wordVal b < 0) = false := decide_eq_false (not_lt.2 hpos.le)
  have hc : encCodes e q = (e q).val := by simp only [encCodes, q.isLt, ↓reduceDIte]
  have hm : Prim.eval ops .mul [b, Prim.eval ops .e4m3ToF32 [encCodes e q]] = mulCode (wordVal b) (e q) := by
    rw [hc]
    by_cases h80 : (e q).val = 0x80
    · have he : e q = 0x80 := Fin.ext h80
      have hw : Prim.eval ops .e4m3ToF32 [(e q).val] = 0x80000000 := by
        simp only [Prim.eval, Prim.w, List.getD_cons_zero]
        rw [h80, hops.widenNeg0]
      rw [hw]
      simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
      rw [hops.mulZero _ _ _ _ hb.1 (by norm_num) hβ PeelExactProofs.wordQ_negZero (mul_zero _)
        (by rw [hsb]; decide)]
      unfold mulCode
      rw [he, show codeVal (0x80 : Code) = 0 by decide +kernel, mul_zero, ite_eq_left rfl, hd]
      decide
    · have hw : Prim.eval ops .e4m3ToF32 [(e q).val] = f32Word (codeVal (e q)) := by
        simp only [Prim.eval, Prim.w, List.getD_cons_zero]
        exact hops.widen _ hn h80
      rw [hw]
      simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
      have hz : wordVal b * codeVal (e q) = 0 → codeVal (e q) = 0 := fun h0 =>
        (mul_eq_zero.1 h0).resolve_left (ne_of_gt hpos)
      rw [hops.mul _ _ _ _ hb.1 (f32Word_lt _) hβ (wordQ_f32Word_code _ hn) fun h0 => by
        rw [hsb, hz h0]; rfl]
      unfold mulCode
      by_cases h0 : wordVal b * codeVal (e q) = 0
      · rw [ite_eq_left h0, hd, code_sign_of_zero _ hn h80 (hz h0), h0]
        simp [f32Word]
      · rw [ite_eq_right h0]
  unfold eTilde8 lineCodes
  rw [hm]
  simp only [Prim.eval, Prim.w, List.getD_cons_zero, hops.cast]
  exact Nat.mod_eq_of_lt (e4m3Word _).isLt

/-- **The widening of a code** other than NaN is its `codeBf16` word in FP32's top half, `−0` included. -/
theorem widen8_word {c : Code} (hn : c.isNaN = false) : Prim.eval ops .e4m3ToF32 [c.val] = codeBf16 c <<< 16 := by
  simp only [Prim.eval, Prim.w, List.getD_cons_zero]
  by_cases h80 : c.val = 0x80
  · rw [h80, hops.widenNeg0, show c = 0x80 from Fin.ext h80]
    decide +kernel
  · rw [hops.widen c hn h80, codeBf16_eq c hn h80]
    exact (bf16_shift_code c hn).symm

/-- **`bf16(E′)`** of a code other than NaN is its widening `codeBf16`, `−0` included. -/
theorem be8_word {c : Code} (hn : c.isNaN = false) :
    Prim.eval ops .f2fpBf16 [Prim.eval ops .e4m3ToF32 [c.val]] = codeBf16 c := by
  rw [widen8_word hops hn]
  simp only [Prim.eval, Prim.w, List.getD_cons_zero]
  exact hops.bf16 _ _ (codeBf16_shift_lt c) (rnBf16_codeBf16 c hn)

/-- **A row's code is `pc8Sem`'s**: `code8W` on a row holding `encRow`'s words (or `−0` where `encRow` writes `+0`),
the noise line's and a basis row's codes is `e4m3` of `pc8Form` at `alpha8`'s and `beta8`'s values, the entry and the
two rows, where the formed word is a `CastWord`. -/
theorem code8W_sem (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) {V : ℕ → ℕ} {x : Fin K → ℚ}
    (hx : ∀ l : Fin K, ZeroSign (V l) (encRow K x l)) (ha : AccumProofs.StepWord (alpha8 ops (encRow K x) K))
    (hb : AccumProofs.StepWord (beta8 ops (encRow K x) K)) (e f : Fin Params.pi.r → Code) (t : Fin K)
    (he : ∀ l < 32, V (K + l) = encCodes e l) (hf : ∀ l < 32, V (K + 32 + 32 * t + l) = encCodes f l)
    (hD : CastWord (formWord8 ops p (wordVal (alpha8 ops (encRow K x) K)) (x t)
      (wordVal (beta8 ops (encRow K x) K)) e f)) :
    code8W ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
        (fun l => V (K + 32 + 32 * t + l)) =
      (e4m3 ((pc8Form ops p).value (wordVal (alpha8 ops (encRow K x) K)) (x t)
        (wordVal (beta8 ops (encRow K x) K)) e f)).val := by
  have hV : ∀ l < K, ZeroSign (V l) (encRow K x l) := fun l hl => hx ⟨l, hl⟩
  have hform : form8 ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
      (fun l => V (K + 32 + 32 * t + l)) =
        formWord8 ops p (wordVal (alpha8 ops (encRow K x) K)) (x t) (wordVal (beta8 ops (encRow K x) K)) e f := by
    unfold formWord8
    rw [f32Word_wordVal ha, f32Word_wordVal hb, alpha8_zeroSign hops hK hV, beta8_zeroSign hops hK hV,
      form8_zeroSign hops p ha (hx t)]
    simp only [encRow, t.isLt, ↓reduceDIte, Fin.eta]
    exact form8_congr ops p _ _ (fun l hl => by rw [he l hl]) (fun l hl => hf l hl)
  rw [code8W_e4m3 ops hops.cast p _ _ _ _ (by rw [hform]; exact hD.1) (by rw [hform]; exact hD.2), hform]
  rfl

/-- `P_A`'s and `P_B`'s narrowing of a word that `rnBf16?` narrows. -/
theorem bf16_word {c x : ℕ} (hc : c < 2 ^ 32) (h : rnBf16? c = some x) : Prim.eval ops .f2fpBf16 [c] = x := by
  simp only [Prim.eval, Prim.w, List.getD_cons_zero]
  exact hops.bf16 _ _ hc h

/-- **RowB's FMA chain is `accChain?`'s**: `pbAcc` over FP32 words that are BF16 words widened, against words with
`G`'s low 32 bits, is the chain `accChain?` folds, term by term. -/
theorem pbAcc_fold {ew g be G : ℕ → ℕ} (hew : ∀ s < 32, ew s = be s <<< 16) (hlt : ∀ s < 32, ew s < 2 ^ 32)
    (hg : ∀ s < 32, g s = G s % 2 ^ 32) :
    ∀ n < 32, ∀ acc, ((List.range (n + 1)).map fun s => (be s, G s)).foldlM
      (fun acc x => fmaWord? (x.1 <<< 16) x.2 acc) 0 = some acc → pbAcc ops ew g n = acc := by
  intro n
  induction n with
  | zero =>
    intro _ acc h
    rw [show List.range (0 + 1) = [0] from rfl] at h
    simp only [List.map_cons, List.map_nil, List.foldlM_cons, List.foldlM_nil] at h
    obtain ⟨a, ha, h⟩ := Option.bind_eq_some_iff.1 h
    obtain rfl := Option.some.inj h
    show Prim.eval ops .fma [ew 0, g 0, 0] = a
    simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
    refine hops.fma _ _ _ _ (hlt 0 (by norm_num)) ?_ (by norm_num) ?_
    · rw [hg 0 (by norm_num)]; exact Nat.mod_lt _ (by norm_num)
    · rw [hew 0 (by norm_num), hg 0 (by norm_num), fmaWord?_mod32]; exact ha
  | succ n ih =>
    intro hn acc h
    rw [List.range_succ, List.map_append, List.foldlM_append] at h
    obtain ⟨a, ha, h⟩ := Option.bind_eq_some_iff.1 h
    have hpa := ih (by omega) a ha
    simp only [List.map_cons, List.map_nil, List.foldlM_cons, List.foldlM_nil] at h
    obtain ⟨b, hb, h⟩ := Option.bind_eq_some_iff.1 h
    obtain rfl := Option.some.inj h
    show Prim.eval ops .fma [ew (n + 1), g (n + 1), pbAcc ops ew g n] = b
    have hpl : pbAcc ops ew g n < 2 ^ 32 := by
      cases n with
      | zero => exact Nat.mod_lt _ (by norm_num)
      | succ n => exact Nat.mod_lt _ (by norm_num)
    simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
    refine hops.fma _ _ _ _ (hlt _ hn) ?_ hpl ?_
    · rw [hg _ hn]; exact Nat.mod_lt _ (by norm_num)
    · rw [hew _ hn, hg _ hn, fmaWord?_mod32, hpa]; exact hb

end Row8

/-! ## `pc8Sem`'s operands -/

section Sem8

variable (ops : RowOps) (p : Pipeline)

/-- `pc8Sem`'s `α` of a row of words. -/
def alphaQ8 (k : ℕ) (w : ℕ → ℕ) : ℚ := wordVal (alpha8 ops (encRow k (rowQ k w)) k)

/-- `pc8Sem`'s `β` of a row of words. -/
def betaQ8 (k : ℕ) (w : ℕ → ℕ) : ℚ := wordVal (beta8 ops (encRow k (rowQ k w)) k)

/-- A row's codes as `pc8Sem` forms them: the row of words `w`, its noise line `e` and the form basis `F`. -/
def quant8 {k : ℕ} (w : ℕ → ℕ) (e : Fin Params.pi.r → Code) (F : Matrix (Fin k) (Fin Params.pi.r) Code) :
    Fin k → Code :=
  fun l => e4m3 ((pc8Form ops p).value (alphaQ8 ops k w) (rowQ k w l) (betaQ8 ops k w) e (F l))

/-- `A′` as `pc8Sem`'s `unitChain` forms it from the activations' rows and the noise `z`. -/
def quant8A {m n k : ℕ} (z : Noise m n k Params.pi.r) (act : Codes) : Matrix (Fin m) (Fin k) Code :=
  fun i => quant8 ops p (act i) (z.EA i) z.FA

/-- `B̃` as `pc8Sem`'s `unitChain` forms it from the weights' rows and the noise `z`. -/
def quant8B {m n k : ℕ} (z : Noise m n k Params.pi.r) (wt : Codes) : Matrix (Fin n) (Fin k) Code :=
  fun j => quant8 ops p (wt j) (z.EB j) z.FB

end Sem8

/-- A basis' codes, over `Fin 32`. -/
def basis8 {k : ℕ} (F : Matrix (Fin k) (Fin Params.pi.r) Code) : Matrix (Fin k) (Fin 32) Code := fun l q => F l q

/-- A noise line's `bf16(E′)` words as the protocol reads them, `E′ = lineCodes β e`. -/
def be8 (β : ℚ) (e : Fin Params.pi.r → Code) : Fin 32 → ℕ := fun q => codeBf16 (lineCodes β e q)

/-! ## The row units' ports -/

section Ports

variable {M N K : ℕ}

theorem rowIn_false (K : ℕ) : rowIn K false = 65 * K + 32 := by simp [rowIn]

theorem rowIn_true (K : ℕ) : rowIn K true = 65 * K + 1056 := by simp [rowIn]

theorem portA8_x {i l : ℕ} (hl : l < K) : portA8 M N K i l = i * K + l := by
  unfold portA8; rw [ite_eq_left hl]

theorem portA8_e {i l : ℕ} (hl : l < 32) : portA8 M N K i (K + l) = eaAt M K + 32 * i + l := by
  unfold portA8; rw [ite_eq_right (by omega), ite_eq_left (by omega)]; omega

theorem portA8_f {i t l : ℕ} (ht : t < K) (hl : l < 32) :
    portA8 M N K i (K + 32 + 32 * t + l) = faAt M N K + 32 * t + l := by
  unfold portA8; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega)]; omega

theorem portA8_b {i t q : ℕ} (hq : q < 32) :
    portA8 M N K i (33 * K + 32 + 32 * t + q) = fbAt M N K + 32 * t + q := by
  unfold portA8; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]; omega

theorem portB8_x {j l : ℕ} (hl : l < K) : portB8 M N K j l = bAt M K + j * K + l := by
  unfold portB8; rw [ite_eq_left hl]

theorem portB8_e {j l : ℕ} (hl : l < 32) : portB8 M N K j (K + l) = ebAt M N K + 32 * j + l := by
  unfold portB8; rw [ite_eq_right (by omega), ite_eq_left (by omega)]; omega

theorem portB8_f {j t l : ℕ} (ht : t < K) (hl : l < 32) :
    portB8 M N K j (K + 32 + 32 * t + l) = fbAt M N K + 32 * t + l := by
  unfold portB8; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega)]; omega

theorem portB8_b {j t q : ℕ} (ht : t < K) (hq : q < 32) :
    portB8 M N K j (33 * K + 32 + 32 * t + q) = faAt M N K + 32 * t + q := by
  unfold portB8; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega)]; omega

theorem portB8_g {j s q : ℕ} (hs : s < 32) (hq : q < 32) :
    portB8 M N K j (65 * K + 32 + 32 * s + q) = gAt M N K + 32 * s + q := by
  unfold portB8; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]; omega

end Ports

/-! ## A row unit's codes and factor row -/

theorem encCodes_of_lt {e : Fin Params.pi.r → Code} {l : ℕ} (hl : l < 32) : encCodes e l = (e ⟨l, hl⟩).val := by
  unfold encCodes
  exact dite_eq_left hl

/-! ## The formed word is a `CastWord` -/

section Cast8

/-- A word that is not a NaN reads as a value that is not NaN. -/
theorem fp32Value_isNaN_false {w : ℕ} (h : (w >>> 23) % 256 = 255 → w % 2 ^ 23 = 0) :
    (fp32Value w).isNaN = false := by
  unfold fp32Value
  dsimp only
  split_ifs with h1 h2 <;> first | rfl | exact absurd (h h1) h2

/-- `packTerm` never writes a NaN. -/
theorem packTerm_notNaN (t : Term) : (packTerm t >>> 23) % 256 = 255 → packTerm t % 2 ^ 23 = 0 := by
  by_cases h0 : t.mag = 0
  · unfold packTerm; rw [ite_eq_left h0]; decide
  by_cases h1 : 127 < t.exp
  · unfold packTerm; rw [ite_eq_right h0, ite_eq_left h1]; split <;> decide
  rw [AccumProofs.packTerm_fin t h0 h1 (AccumProofs.packTerm_exp t h1)]
  have hm := AccumProofs.and_mask_lt t.mag
  have hs : (if t.neg = true then 1 else 0) ≤ 1 := by split <;> norm_num
  have he : (t.exp + 127 - if t.mag &&& 1 <<< 23 = 0 then 1 else 0).toNat < 255 := by split_ifs <;> omega
  generalize (t.exp + 127 - if t.mag &&& 1 <<< 23 = 0 then 1 else 0).toNat = ef at he ⊢
  generalize (if t.neg = true then 1 else 0) = s at hs ⊢
  generalize t.mag &&& 0x7FFFFF = m at hm ⊢
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow] at hm ⊢
  omega

/-- A group step from a value that is not NaN, over finite products, is not NaN: one infinity at most. -/
theorem groupStepTotal_notNaN (w : ℕ) (z : ℤ) {v : Value} (hv : v.isNaN = false) {ps : List Value}
    (hps : ∀ q ∈ ps, ∃ t, q = .fin t) : (groupStepTotal w z v ps).isNaN = false := by
  have hp : ∀ f : Value → Bool, (∀ t, f (.fin t) = false) → ps.any f = false := fun f hf => by
    rw [List.any_eq_false]
    intro q hq
    obtain ⟨t, rfl⟩ := hps q hq
    simp [hf]
  unfold groupStepTotal
  dsimp only
  rw [List.any_cons, List.any_cons, List.any_cons, hp _ (fun _ => rfl), hp _ (fun _ => rfl),
    hp _ (fun _ => rfl)]
  rcases v with _ | s | t
  · exact absurd hv (by decide)
  · cases s <;> rfl
  · simp only [Value.isNaN, Value.isInf, Bool.or_self, Bool.and_self, Bool.false_eq_true, ↓reduceIte]
    split_ifs <;> rfl

/-- The groups folded from a value that is not NaN, over codes that are not NaN, are not NaN: E4M3 has no
infinities, so every product is finite. -/
theorem fold_notNaN (p : Pipeline) : ∀ (gs : List ℕ) (v : Value) (ab : List (Code × Code)), v.isNaN = false →
    (∀ x ∈ ab, x.1.isNaN = false ∧ x.2.isNaN = false) → (p.fold gs v ab).isNaN = false
  | [], _, _, hv, _ => hv
  | _ :: gs, _, _, hv, hab => fold_notNaN p gs _ _
      (groupStepTotal_notNaN _ _ hv fun q hq => by
        obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hq
        obtain ⟨h1, h2⟩ := hab x (List.mem_of_mem_take hx)
        exact ⟨product x.1 x.2, by simp [productTotal, h1, h2]⟩)
      fun x hx => hab x (List.mem_of_mem_drop hx)

/-- **A step's word is a `CastWord`** from an accumulator that is not NaN, over codes that are not NaN: it writes a
NaN only from a NaN, and never writes `−0` (`step_ne_negZero`). -/
theorem step_castWord (p : Pipeline) {acc : ℕ} {a b : List Code} (hacc : (fp32Value acc).isNaN = false)
    (ha : ∀ c ∈ a, c.isNaN = false) (hb : ∀ c ∈ b, c.isNaN = false) : CastWord (p.step acc a b) := by
  refine ⟨?_, fun h0 => ?_⟩
  · have hv := fold_notNaN p p.groups _ (a.zip b) hacc fun x hx =>
      ⟨ha _ (List.of_mem_zip hx).1, hb _ (List.of_mem_zip hx).2⟩
    unfold Pipeline.step
    generalize p.fold p.groups (fp32Value acc) (a.zip b) = v at hv
    rcases v with _ | s | t
    · exact absurd hv (by decide)
    · cases s <;> decide
    · exact packTerm_notNaN t
  · have hw : AccumProofs.StepWord (p.step acc a b) :=
      ⟨AccumProofs.step_lt _ _ _ _, AccumProofs.step_ne_negZero _ _ _ _, by
        unfold finite?; rw [ite_eq_right (wordQ_exp_ne h0)]⟩
    rw [AccumProofs.wordQ_zero_word hw h0]
    rfl

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

/-- The product of two step words is not NaN: `f32Word` of the product, or `−0` (`mulZero`). -/
theorem mul8_notNaN {a b : ℕ} (ha : AccumProofs.StepWord a) (hb : AccumProofs.StepWord b) :
    (fp32Value (Prim.eval ops .mul [a, b])).isNaN = false := by
  have hqa := stepWord_wordQ ha
  have hqb := stepWord_wordQ hb
  simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
  by_cases hz : wordVal a * wordVal b = 0 ∧ signBit a ≠ signBit b
  · rw [hops.mulZero _ _ _ _ ha.1 hb.1 hqa hqb hz.1 hz.2]; decide
  · rw [hops.mul _ _ _ _ ha.1 hb.1 hqa hqb fun h0 => by by_contra hs; exact hz ⟨h0, hs⟩]
    exact fp32Value_isNaN_false (f32Word_notNaN _)

/-- **The formed word is a `CastWord`** on a `Row8OK` row and `Line8OK` noise and basis lines: its accumulator `α·x_t`
is the product of step words, its noise codes are `lineCodes`'s and its basis codes the basis', none of them NaN. -/
theorem formWord8_castWord (p : Pipeline) {K : ℕ} {w : ℕ → ℕ} (hA : Row8OK ops K w) {e : Fin Params.pi.r → Code}
    (hL : Line8OK e) (t : Fin K) {f : Fin Params.pi.r → Code} (hF : Line8OK f) :
    CastWord (formWord8 ops p (alphaQ8 ops K w) (rowQ K w t) (betaQ8 ops K w) e f) := by
  have hx := (rowQ_zeroSign hA.1 t).2
  unfold formWord8 form8 alphaQ8 betaQ8
  rw [f32Word_wordVal hA.2.1, f32Word_wordVal hA.2.2.1, List.cons_append, atom8_eval]
  refine step_castWord p (mul8_notNaN hops hA.2.1 hx) ?_ ?_
  · intro c hc
    obtain ⟨l, rfl⟩ := List.mem_ofFn.1 hc
    rw [eTilde8_line hops hA.2.2.1 hA.2.2.2 ⟨l, l.isLt⟩ (hL _), code_val]
    exact lineCodes_notNaN _ _ _
  · intro c hc
    obtain ⟨l, rfl⟩ := List.mem_ofFn.1 hc
    rw [encCodes_of_lt l.isLt, code_val]
    exact hF _

end Cast8

section RowWords

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

/-- **A row unit's code is `quant8`'s**: `code8W` on a row holding the row's words, its noise line's and a form basis
row's codes. -/
theorem code8_quant (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) {V w : ℕ → ℕ} (hx : ∀ l < K, V l = w l % 2 ^ 32)
    (hA : Row8OK ops K w) {e : Fin Params.pi.r → Code} (hL : Line8OK e) (F : Matrix (Fin K) (Fin Params.pi.r) Code)
    (hF : ∀ t, Line8OK (F t)) (he : ∀ l < 32, V (K + l) = encCodes e l) (t : Fin K)
    (hf : ∀ l < 32, V (K + 32 + 32 * t + l) = encCodes (F t) l) :
    code8W ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
        (fun l => V (K + 32 + 32 * t + l)) = (quant8 ops p w e F t).val :=
  code8W_sem hops p hK (x := rowQ K w) (fun l => by rw [hx l l.isLt]; exact encRow_zeroSign hA.1 l l.isLt) hA.2.1
    hA.2.2.1 e (F t) t he hf (formWord8_castWord hops p hA hL t (hF t))

/-- **A row unit's rescaled noise code is `lineCodes`'s** at `pc8Sem`'s `β` of the row's words. -/
theorem eTilde8_row {K : ℕ} (hK : 8 ≤ K) {V w : ℕ → ℕ} (hx : ∀ l < K, V l = w l % 2 ^ 32) (hA : Row8OK ops K w)
    {e : Fin Params.pi.r → Code} (hL : Line8OK e) (he : ∀ l < 32, V (K + l) = encCodes e l) {s : ℕ}
    (hs : s < 32) : eTilde8 ops (beta8 ops V K) (V (K + s)) = (lineCodes (betaQ8 ops K w) e ⟨s, hs⟩).val := by
  have hV : ∀ l < K, ZeroSign (V l) (encRow K (rowQ K w) l) := fun l hl => by
    rw [hx l hl]; exact encRow_zeroSign hA.1 l hl
  rw [beta8_zeroSign hops hK hV, he s hs]
  exact eTilde8_line hops hA.2.2.1 hA.2.2.2 ⟨s, hs⟩ (hL ⟨s, hs⟩)

end RowWords

theorem getD_ofFn32 (f : Fin 32 → ℕ) {s : ℕ} (hs : s < 32) : (List.ofFn f).getD s 0 = f ⟨s, hs⟩ :=
  PeelExactProofs.getD_ofFn f 0 ⟨s, hs⟩

theorem bf16WordOf_lt (v : ℚ) : bf16WordOf v < 2 ^ 16 := by
  have := f32Word_lt v
  unfold bf16WordOf
  rw [Nat.shiftRight_eq_div_pow]
  simp only [Nat.reducePow] at this ⊢
  omega

section RowWords

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

/-- **RowA's codes and `P_A` row are the protocol's**: a correct RowA transcript holding a row of words, its noise line,
its form basis `F` and its chain basis `Fc` has `quant8`'s codes, and, where the peel's chain words `af` against `Fc`'s
columns exist, the `P_A` row `paRow?` forms from them and the line's `bf16(E′)`. -/
theorem row8A_paWords (p : Pipeline) {nw : ℕ} (hnw : 0 < nw)
    (XA : ((row8A p (128 * nw)).spec ops).toCircuit.Transcript)
    (hX : ((row8A p (128 * nw)).spec ops).part.Correct XA 0) {w : ℕ → ℕ} {e : Fin Params.pi.r → Code}
    {F Fc : Matrix (Fin (128 * nw)) (Fin Params.pi.r) Code} (hx : ∀ l < 128 * nw, val XA l = w l % 2 ^ 32)
    (he : ∀ l < 32, val XA (128 * nw + l) = encCodes e l)
    (hf : ∀ t : Fin (128 * nw), ∀ l < 32, val XA (128 * nw + 32 + 32 * t + l) = encCodes (F t) l)
    (hb : ∀ t : Fin (128 * nw), ∀ q < 32, val XA (33 * (128 * nw) + 32 + 32 * t + q) = encCodes (Fc t) q)
    (hA : Row8OK ops (128 * nw) w) (hL : Line8OK e) (hF : ∀ t, Line8OK (F t))
    {af pa : List ℕ}
    (haf : (List.ofFn fun q : Fin 32 => chainWordP? p 4 (rowAtoms (quant8 ops p w e F) ((basis8 Fc)ᵀ q))).mapM id =
      some af)
    (hpa : paRow? (List.ofFn (be8 (betaQ8 ops (128 * nw) w) e)) af = some pa) :
    (∀ t : Fin (128 * nw), val XA (code8 (128 * nw) (rowIn (128 * nw) false) t) = (quant8 ops p w e F t).val) ∧
      ∀ q < 64, val XA (paOut8 (128 * nw) q) = pa.getD q 0 := by
  have hK8 : 8 ≤ 128 * nw := by omega
  have hcode : ∀ t : Fin (128 * nw),
      val XA (code8 (128 * nw) (rowIn (128 * nw) false) t) = (quant8 ops p w e F t).val := fun t =>
    (row8A_code (ops := ops) p hK8 XA hX t.isLt).trans (code8_quant hops p hK8 hx hA hL F hF he t (hf t))
  refine ⟨hcode, fun q hq => ?_⟩
  obtain ⟨hpa1, hpa2⟩ := PeelExactProofs.paRow_spec hpa List.length_ofFn
  obtain ⟨hafl, hafq⟩ := PeelExactProofs.mapM_ofFn _ 0 haf
  have hdiv : 128 * nw / 128 = nw := by omega
  unfold paOut8
  by_cases hq32 : q < 32
  · rw [ite_eq_left hq32, hdiv, row8A_be p hnw rfl XA hX hq32, eTilde8_row hops hK8 hx hA hL he hq32,
      be8_word hops (lineCodes_notNaN _ _ _), hpa1 q hq32, getD_ofFn32 _ hq32]
    rfl
  · have hq' : q - 32 < 32 := by omega
    have hc := hafq ⟨q - 32, hq'⟩
    rw [ite_eq_right hq32, hdiv,
      row8A_pa p hnw rfl XA hX (quant8 ops p w e F) ((basis8 Fc)ᵀ ⟨q - 32, hq'⟩) hcode hq'
        (fun t => (hb t (q - 32) hq').trans (encCodes_of_lt hq')) hc,
      bf16_word hops (chainWordP?_lt hc) (hpa2 (q - 32) (by omega)), show 32 + (q - 32) = q by omega]

/-- **RowB's codes and `P_B` row are the protocol's**: a correct RowB transcript holding a row of words, its noise line,
its form basis `F`, its chain basis `Fc` and `G` has `quant8`'s codes, and, where the peel's chain words `bf` against
`Fc`'s columns and `G`'s rows `gL` exist, the `P_B` row `pbRow?` forms from them and the line's `bf16(E′)`. -/
theorem row8B_pbWords (p : Pipeline) {nw : ℕ} (hnw : 0 < nw)
    (XB : ((row8B p (128 * nw)).spec ops).toCircuit.Transcript)
    (hX : ((row8B p (128 * nw)).spec ops).part.Correct XB 0) {w : ℕ → ℕ} {e : Fin Params.pi.r → Code}
    {F Fc : Matrix (Fin (128 * nw)) (Fin Params.pi.r) Code} (hx : ∀ l < 128 * nw, val XB l = w l % 2 ^ 32)
    (he : ∀ l < 32, val XB (128 * nw + l) = encCodes e l)
    (hf : ∀ t : Fin (128 * nw), ∀ l < 32, val XB (128 * nw + 32 + 32 * t + l) = encCodes (F t) l)
    (hb : ∀ t : Fin (128 * nw), ∀ q < 32, val XB (33 * (128 * nw) + 32 + 32 * t + q) = encCodes (Fc t) q)
    (hA : Row8OK ops (128 * nw) w) (hL : Line8OK e) (hF : ∀ t, Line8OK (F t))
    {bf pb : List ℕ} {gL : List (List ℕ)}
    (hbf : (List.ofFn fun q : Fin 32 => chainWordP? p 4 (rowAtoms (quant8 ops p w e F) ((basis8 Fc)ᵀ q))).mapM id =
      some bf)
    (hgl : gL.length = 32)
    (hg : ∀ s < 32, ∀ q < 32, val XB (65 * (128 * nw) + 32 + 32 * s + q) = (gL.getD s []).getD q 0 % 2 ^ 32)
    (hpb : pbRow? (List.ofFn (be8 (betaQ8 ops (128 * nw) w) e)) bf gL = some pb) :
    (∀ t : Fin (128 * nw), val XB (code8 (128 * nw) (rowIn (128 * nw) true) t) = (quant8 ops p w e F t).val) ∧
      ∀ q < 64, val XB (pbOut8 (128 * nw) q) = pb.getD q 0 := by
  have hK8 : 8 ≤ 128 * nw := by omega
  have hcode : ∀ t : Fin (128 * nw),
      val XB (code8 (128 * nw) (rowIn (128 * nw) true) t) = (quant8 ops p w e F t).val := fun t =>
    (row8B_code (ops := ops) p hK8 XB hX t.isLt).trans (code8_quant hops p hK8 hx hA hL F hF he t (hf t))
  refine ⟨hcode, fun q hq => ?_⟩
  obtain ⟨hpb1, hpb2⟩ := PeelExactProofs.pbRow_spec hpb List.length_ofFn
  obtain ⟨-, hbfq⟩ := PeelExactProofs.mapM_ofFn _ 0 hbf
  have hdiv : 128 * nw / 128 = nw := by omega
  unfold pbOut8
  by_cases hq32 : q < 32
  · obtain ⟨acc, s, hacc, hs, hrn⟩ := hpb1 q hq32
    have hc := hbfq ⟨q, hq32⟩
    have hew : ∀ s (hs : s < 32), Prim.eval ops .e4m3ToF32
        [eTilde8 ops (beta8 ops (val XB) (128 * nw)) (val XB (128 * nw + s))] =
          codeBf16 (lineCodes (betaQ8 ops (128 * nw) w) e ⟨s, hs⟩) <<< 16 := fun s hs => by
      rw [eTilde8_row hops hK8 hx hA hL he hs, widen8_word hops (lineCodes_notNaN _ _ _)]
    have hlist : ((List.range (31 + 1)).map fun s =>
        ((List.ofFn (be8 (betaQ8 ops (128 * nw) w) e)).getD s 0, (gL.getD s []).getD q 0)) =
          List.ofFn fun p : Fin 32 => (be8 (betaQ8 ops (128 * nw) w) e p, (gL.getD p []).getD q 0) := by
      apply List.ext_getElem
      · simp
      · intro n h1 h2
        simp only [List.getElem_map, List.getElem_range, List.getElem_ofFn]
        rw [getD_ofFn32 _ (by simpa using h2)]
    have hpa : pbAcc ops (fun s => Prim.eval ops .e4m3ToF32
        [eTilde8 ops (beta8 ops (val XB) (128 * nw)) (val XB (128 * nw + s))])
        (fun s => val XB (65 * (128 * nw) + 32 + 32 * s + q)) 31 = acc := by
      refine pbAcc_fold hops (be := fun s => (List.ofFn (be8 (betaQ8 ops (128 * nw) w) e)).getD s 0)
        (G := fun s => (gL.getD s []).getD q 0) (fun s hs => ?_) (fun s hs => ?_) (fun s hs => hg s hs q hq32) 31
        (by norm_num) acc ?_
      · dsimp only
        rw [hew s hs, getD_ofFn32 _ hs]
        rfl
      · dsimp only
        rw [hew s hs]
        exact codeBf16_shift_lt _
      · rw [hlist]
        unfold accChain? at hacc
        rwa [PeelExactProofs.zip_col hgl q] at hacc
    have hneg : Prim.eval ops .neg [bf.getD q 0] = negWord (bf.getD q 0) := by
      simp only [Prim.eval, Prim.w, List.getD_cons_zero]
      exact hops.neg _ (chainWordP?_lt hc)
    have hadd : Prim.eval ops .add [acc, negWord (bf.getD q 0)] = s := by
      simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
      rw [f32Add_of_addWord hs]
      exact Nat.mod_eq_of_lt (addWord?_lt hs)
    rw [ite_eq_left hq32, hdiv,
      row8B_pb p hnw rfl XB hX (quant8 ops p w e F) ((basis8 Fc)ᵀ ⟨q, hq32⟩) hcode hq32
        (fun t => (hb t q hq32).trans (encCodes_of_lt hq32)) hc, hpa, hneg, hadd]
    exact bf16_word hops (addWord?_lt hs) hrn
  · have hq' : q - 32 < 32 := by omega
    have h2 := hpb2 (q - 32) hq'
    rw [show 32 + (q - 32) = q by omega, getD_ofFn32 _ hq'] at h2
    rw [ite_eq_right hq32, hdiv, row8B_nbe p hnw rfl XB hX hq', eTilde8_row hops hK8 hx hA hL he hq',
      be8_word hops (lineCodes_notNaN _ _ _), h2]
    have hx16 := codeBf16_lt (lineCodes (betaQ8 ops (128 * nw) w) e ⟨q - 32, hq'⟩)
    have hxor := Nat.xor_lt_two_pow hx16 (show 0x8000 < 2 ^ 16 by norm_num)
    simp only [Prim.eval, Prim.w, List.getD_cons_zero, bf16NegWord, Nat.mod_mod]
    exact Nat.mod_eq_of_lt hxor

end RowWords

theorem gram8_eq (p : Pipeline) {m n k : ℕ} (z : Noise m n k Params.pi.r) {s q : ℕ} (hs : s < 32) (hq : q < 32) :
    gram8 p z s q = (chainWordP? p 4 (rowAtoms ((basis8 z.FB)ᵀ ⟨s, hs⟩) ((basis8 z.FA)ᵀ ⟨q, hq⟩))).getD 0 := by
  unfold gram8
  exact dite_eq_left ⟨hs, hq⟩

/-! ## An entry's words -/

section Entry8

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

/-- **Entry `(i, j)`'s words are the protocol's.** Where RowA `i`, RowB `j` and their tile are correct and the inputs
hold row `i` of the activations, row `j` of the weights, their noise lines, the bases and `G`: under the named
hypotheses at the two rows, and where the peel's factor rows exist, `C̃_ij` and `U_ij` are the device's ticket and
useful words on `pc8Sem`'s `A′` and `B̃` (`quant8A`, `quant8B`), the bases' codes and the lines' `bf16(E′)`. -/
theorem entry8_words (p : Pipeline) (pr : Prices) {M K N TM TN nw : ℕ} (hnw : 0 < nw) (hK : K = 128 * nw)
    (hTM : TM ∣ M) (hTN : TN ∣ N) (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript)
    (z : Noise M N K Params.pi.r) (act wt : Codes) (i : Fin M) (j : Fin N)
    (hXT : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_T ops p M K N TM TN) (tile_index hTM hTN i.isLt j.isLt).1⟩)
    (hXA : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_A ops p M K N TM TN) i.isLt⟩)
    (hXB : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_B ops p M K N TM TN) j.isLt⟩)
    (hxa : ∀ l < K, val X ((i : ℕ) * K + l) = act i l % 2 ^ 32)
    (hea : ∀ l < 32, val X (eaAt M K + 32 * (i : ℕ) + l) = encCodes (z.EA i) l)
    (hxb : ∀ l < K, val X (bAt M K + (j : ℕ) * K + l) = wt j l % 2 ^ 32)
    (heb : ∀ l < 32, val X (ebAt M N K + 32 * (j : ℕ) + l) = encCodes (z.EB j) l)
    (hfa : ∀ t : Fin K, ∀ l < 32, val X (faAt M N K + 32 * (t : ℕ) + l) = encCodes (z.FA t) l)
    (hfb : ∀ t : Fin K, ∀ l < 32, val X (fbAt M N K + 32 * (t : ℕ) + l) = encCodes (z.FB t) l)
    (hg : ∀ s < 32, ∀ q < 32, val X (gAt M N K + 32 * s + q) = gram8 p z s q)
    (hA : Row8OK ops K (act i)) (hB : Row8OK ops K (wt j)) (hz : Noise8OK z)
    {c : ℕ} {pa pb : List ℕ}
    (hr : PeelExactProofsP.peelRowsP p 4 (quant8A ops p z act) (quant8B ops p z wt) (basis8 z.FA) (basis8 z.FB)
      (fun i => be8 (betaQ8 ops K (act i)) (z.EA i)) (fun j => be8 (betaQ8 ops K (wt j)) (z.EB j)) i j =
        some (c, pa, pb)) :
    wordQ (val X (cGlob8 p M N K TM TN i j)) = (devAt p 4 pr).ticket (quant8A ops p z act) (quant8B ops p z wt) i j ∧
      wordQ (val X (uGlob8 p M N K TM TN i j)) = (devAt p 4 pr).useful (quant8A ops p z act) (quant8B ops p z wt)
        (basis8 z.FA) (basis8 z.FB) (fun i => be8 (betaQ8 ops K (act i)) (z.EA i))
        (fun j => be8 (betaQ8 ops K (wt j)) (z.EB j)) i j := by
  subst hK
  obtain ⟨af, bf, gL, -, haf, hbf, hgg, hpa, hpb⟩ := PeelExactProofsP.peelRowsP_spec hr
  obtain ⟨hgl, hgrow⟩ := PeelExactProofs.mapM_ofFn _ [] hgg
  have hinA : ∀ a < rowIn (128 * nw) false,
      val ((whole8 ops p M (128 * nw) N TM TN).cut X (blockA8 ops p M (128 * nw) N) i) a =
        val X (portA8 M N (128 * nw) i a) := fun a ha => cutA8_input ops p M (128 * nw) N TM TN X i.isLt ha
  have hinB : ∀ a < rowIn (128 * nw) true,
      val ((whole8 ops p M (128 * nw) N TM TN).cut X (blockB8 ops p M (128 * nw) N) j) a =
        val X (portB8 M N (128 * nw) j a) := fun a ha => cutB8_input ops p M (128 * nw) N TM TN X j.isLt ha
  have hrA := rowIn_false (128 * nw)
  have hrB := rowIn_true (128 * nw)
  obtain ⟨codeA, paA⟩ := row8A_paWords hops p hnw _
    ((whole8 ops p M (128 * nw) N TM TN).cut_correct X (blockA8 ops p M (128 * nw) N) i)
    (fun l hl => (hinA l (by omega)).trans (by rw [portA8_x hl, hxa l hl]))
    (fun l hl => (hinA _ (by omega)).trans (by rw [portA8_e hl, hea l hl]))
    (fun t l hl => (hinA _ (by have := t.isLt; omega)).trans (by rw [portA8_f t.isLt hl, hfa t l hl]))
    (fun t q hq => (hinA _ (by have := t.isLt; omega)).trans (by rw [portA8_b hq, hfb t q hq]))
    hA (hz.1 i) hz.2.1 haf hpa
  obtain ⟨codeB, pbB⟩ := row8B_pbWords hops p hnw _
    ((whole8 ops p M (128 * nw) N TM TN).cut_correct X (blockB8 ops p M (128 * nw) N) j)
    (fun l hl => (hinB l (by omega)).trans (by rw [portB8_x hl, hxb l hl]))
    (fun l hl => (hinB _ (by omega)).trans (by rw [portB8_e hl, heb l hl]))
    (fun t l hl => (hinB _ (by have := t.isLt; omega)).trans (by rw [portB8_f t.isLt hl, hfb t l hl]))
    (fun t q hq => (hinB _ (by have := t.isLt; omega)).trans (by rw [portB8_b t.isLt hq, hfa t q hq]))
    hB (hz.2.2.1 j) hz.2.2.2 hbf hgl
    (fun s hs q hq => by
      have h := (PeelExactProofs.mapM_ofFn _ 0 (hgrow ⟨s, hs⟩)).2 ⟨q, hq⟩
      refine (hinB _ (by omega)).trans ?_
      rw [portB8_g hs hq, hg s hs q hq, gram8_eq p z hs hq, Nat.mod_eq_of_lt (chainWordP?_lt h), h]
      rfl)
    hpb
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM i.isLt)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN j.isLt)
  have hk := Nat.dvd_mul_right 128 nw
  obtain ⟨h1, h2⟩ := tile_devAt p pr hnw _
    ((whole8 ops p M (128 * nw) N TM TN).cut_correct X (blockT8 p M (128 * nw) N TM TN) (i / TM * (N / TN) + j / TN))
    (quant8A ops p z act) (quant8B ops p z wt) (basis8 z.FA) (basis8 z.FB)
    (fun i => be8 (betaQ8 ops (128 * nw) (act i)) (z.EA i)) (fun j => be8 (betaQ8 ops (128 * nw) (wt j)) (z.EB j)) i j
    hti htj (fun l => (chain8_A ops p M (128 * nw) N TM TN X hTM hTN i.isLt j.isLt hk hXA l.isLt).trans (codeA l))
    (fun l => (chain8_B ops p M (128 * nw) N TM TN X hTM hTN i.isLt j.isLt hk hXB l.isLt).trans (codeB l)) hr
    (fun q hq => (chain8_PA ops p M (128 * nw) N TM TN X hTM hTN i.isLt j.isLt hXA hq).trans (paA q hq))
    (fun q hq => (chain8_PB ops p M (128 * nw) N TM TN X hTM hTN i.isLt j.isLt hXB hq).trans (pbB q hq))
  rw [chain8_C ops p M (128 * nw) N TM TN X hTM hTN i.isLt j.isLt (by omega) hXT,
    chain8_U ops p M (128 * nw) N TM TN X hTM hTN i.isLt j.isLt hXT]
  exact ⟨h1, h2⟩

end Entry8

/-! ## The anchors and the registered values, region by region -/

section Regions8

variable (p : Pipeline)

theorem layout8 (M N K : ℕ) : bAt M K = eaAt M K + 32 * M ∧ ebAt M N K = bAt M K + N * K ∧
    faAt M N K = ebAt M N K + 32 * N ∧ fbAt M N K = faAt M N K + 32 * K ∧ gAt M N K = fbAt M N K + 32 * K ∧
    in8 M N K = gAt M N K + 1024 := by
  unfold eaAt bAt ebAt faAt fbAt gAt in8
  exact ⟨by ring, by ring, by ring, by ring, by ring, rfl⟩

theorem encCodes_lt8 {r : ℕ} (e : Fin r → Code) (l : ℕ) : encCodes e l < 2 ^ 8 := by
  unfold encCodes
  split_ifs
  · exact (e _).isLt
  · norm_num

theorem gram8_lt {m n k : ℕ} (z : Noise m n k Params.pi.r) (s q : ℕ) : gram8 p z s q < 2 ^ 32 := by
  unfold gram8
  split_ifs
  · rcases h : chainWordP? p 4 _ with _ | c
    · norm_num
    · exact chainWordP?_lt h
  · norm_num

variable {m n k : ℕ} (z : Noise m n k Params.pi.r)

theorem anchors8_ea {i l : ℕ} (hi : i < m) (hl : l < 32) :
    anchors8 p z (eaAt m k + 32 * i + l) = encCodes (z.EA ⟨i, hi⟩) l := by
  obtain ⟨h1, -⟩ := layout8 m n k
  have hq : (eaAt m k + 32 * i + l - eaAt m k) / 32 = i := by omega
  have hr : (eaAt m k + 32 * i + l - eaAt m k) % 32 = l := by omega
  unfold anchors8
  rw [ite_eq_right (by omega), ite_eq_left (by omega)]
  simp only [hq, hr, hi, ↓reduceDIte]

theorem anchors8_eb {j l : ℕ} (hj : j < n) (hl : l < 32) :
    anchors8 p z (ebAt m n k + 32 * j + l) = encCodes (z.EB ⟨j, hj⟩) l := by
  obtain ⟨h1, h2, h3, -⟩ := layout8 m n k
  have hq : (ebAt m n k + 32 * j + l - ebAt m n k) / 32 = j := by omega
  have hr : (ebAt m n k + 32 * j + l - ebAt m n k) % 32 = l := by omega
  unfold anchors8
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega)]
  simp only [hq, hr, hj, ↓reduceDIte]

theorem anchors8_fa {t l : ℕ} (ht : t < k) (hl : l < 32) :
    anchors8 p z (faAt m n k + 32 * t + l) = encCodes (z.FA ⟨t, ht⟩) l := by
  obtain ⟨h1, h2, h3, h4, -⟩ := layout8 m n k
  have hq : (faAt m n k + 32 * t + l - faAt m n k) / 32 = t := by omega
  have hr : (faAt m n k + 32 * t + l - faAt m n k) % 32 = l := by omega
  unfold anchors8
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left (by omega)]
  simp only [hq, hr, ht, ↓reduceDIte]

theorem anchors8_fb {t l : ℕ} (ht : t < k) (hl : l < 32) :
    anchors8 p z (fbAt m n k + 32 * t + l) = encCodes (z.FB ⟨t, ht⟩) l := by
  obtain ⟨h1, h2, h3, h4, h5, -⟩ := layout8 m n k
  have hq : (fbAt m n k + 32 * t + l - fbAt m n k) / 32 = t := by omega
  have hr : (fbAt m n k + 32 * t + l - fbAt m n k) % 32 = l := by omega
  unfold anchors8
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_left (by omega)]
  simp only [hq, hr, ht, ↓reduceDIte]

theorem anchors8_g {s q : ℕ} (hs : s < 32) (hq : q < 32) :
    anchors8 p z (gAt m n k + 32 * s + q) = gram8 p z s q := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := layout8 m n k
  have hq' : (gAt m n k + 32 * s + q - gAt m n k) / 32 = s := by omega
  have hr' : (gAt m n k + 32 * s + q - gAt m n k) % 32 = q := by omega
  unfold anchors8
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega), hq', hr']

end Regions8

section Rows8

variable (p : Pipeline) {M N K TM TN : ℕ} (act wt tk us : Codes)

theorem rows8_a {i l : ℕ} (hi : i < M) (hl : l < K) : rows8 p M N K TM TN act wt tk us (i * K + l) = act i l := by
  have hK : 0 < K := by omega
  unfold rows8
  rw [ite_eq_left (show i * K + l < eaAt M K from mul_add_lt hi hl),
    show (i * K + l) / K = i by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (i * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem rows8_b {j l : ℕ} (hj : j < N) (hl : l < K) :
    rows8 p M N K TM TN act wt tk us (bAt M K + j * K + l) = wt j l := by
  have hK : 0 < K := by omega
  obtain ⟨h1, h2, -⟩ := layout8 M N K
  have := mul_add_lt hj hl
  unfold rows8
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    show bAt M K + j * K + l - bAt M K = j * K + l by omega,
    show (j * K + l) / K = j by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (j * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

/-- The registered value on a tile gate of entry `(i, j)`'s block: its ticket word at `C̃`, its useful word at `U`. -/
theorem rows8_block {i j r : ℕ} (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) (hr : r < blockLen K) :
    rows8 p M N K TM TN act wt tk us
        (tile8 p M N K TM TN (i / TM * (N / TN) + j / TN) (entryBase K TM TN (i % TM) (j % TN) + r)) =
      if r = 5 * (K / 128) - 1 then tk i j else if r = blockLen K - 1 then us i j else 0 := by
  obtain ⟨-, hie, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hbl := blockLen_pos K
  have hL : tileN K TM TN - tileIn K TM TN = TM * TN * blockLen K := by unfold tileN; omega
  have he := mul_add_lt (L := TN) hti htj
  have hb := mul_add_lt (L := blockLen K) he hr
  have hLp : 0 < TM * TN * blockLen K := by omega
  have hg : tile8 p M N K TM TN (i / TM * (N / TN) + j / TN) (entryBase K TM TN (i % TM) (j % TN) + r) =
      tiles8 p M N K + ((i / TM * (N / TN) + j / TN) * (TM * TN * blockLen K) +
        ((i % TM * TN + j % TN) * blockLen K + r)) := by
    unfold tile8 tiles8 entryBase; rw [hL]; omega
  have hq1 : ((i / TM * (N / TN) + j / TN) * (TM * TN * blockLen K) + ((i % TM * TN + j % TN) * blockLen K + r)) /
      (TM * TN * blockLen K) = i / TM * (N / TN) + j / TN := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hLp, Nat.div_eq_of_lt hb, Nat.zero_add]
  have hr1 : ((i / TM * (N / TN) + j / TN) * (TM * TN * blockLen K) + ((i % TM * TN + j % TN) * blockLen K + r)) %
      (TM * TN * blockLen K) = (i % TM * TN + j % TN) * blockLen K + r := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  have hq2 : ((i % TM * TN + j % TN) * blockLen K + r) / blockLen K = i % TM * TN + j % TN := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hbl, Nat.div_eq_of_lt hr, Nat.zero_add]
  have hr2 : ((i % TM * TN + j % TN) * blockLen K + r) % blockLen K = r := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]
  have htn : 0 < TN := by omega
  have hq3 : (i % TM * TN + j % TN) / TN = i % TM := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ htn, Nat.div_eq_of_lt htj, Nat.zero_add]
  have hr3 : (i % TM * TN + j % TN) % TN = j % TN := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt htj]
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := layout8 M N K
  have ht8 : in8 M N K ≤ tiles8 p M N K := by unfold tiles8; omega
  unfold rows8 entry8
  rw [hg, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]
  simp only [Nat.add_sub_cancel_left, hL, hq1, hr1, hq2, hr2, hq3, hr3, hie, hje]

theorem rows8_C {i j : ℕ} (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) (hK : 0 < K / 128) :
    rows8 p M N K TM TN act wt tk us (cGlob8 p M N K TM TN i j) = tk i j := by
  have hcg : cGate K TM TN (i % TM) (j % TN) = entryBase K TM TN (i % TM) (j % TN) + (5 * (K / 128) - 1) := by
    unfold cGate; omega
  unfold cGlob8
  rw [hcg, rows8_block p act wt tk us hTM hTN hi hj (by unfold blockLen; omega), ite_eq_left rfl]

theorem rows8_U {i j : ℕ} (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) :
    rows8 p M N K TM TN act wt tk us (uGlob8 p M N K TM TN i j) = us i j := by
  have hbl := blockLen_pos K
  have hug : uGate K TM TN (i % TM) (j % TN) = entryBase K TM TN (i % TM) (j % TN) + (blockLen K - 1) := by
    unfold uGate; omega
  unfold uGlob8
  rw [hug, rows8_block p act wt tk us hTM hTN hi hj (by omega), ite_eq_right (by unfold blockLen; omega),
    ite_eq_left rfl]

end Rows8

/-! ## The gates' widths -/

theorem tileGate_w (p : Pipeline) {K TM TN g : ℕ} (hg : tileIn K TM TN ≤ g) : (tileGate p K TM TN g).w = 32 := by
  have h1 : ¬ g < (TM + TN) * K := by unfold tileIn at hg; omega
  unfold tileGate
  rw [ite_eq_right h1, ite_eq_right (not_lt.2 hg)]
  unfold entryGate
  split_ifs <;> rfl

theorem val_bv {S : Spec} (X : S.toCircuit.Transcript) {g : ℕ} (hg : g < S.N) {v : ℕ}
    (h : X ⟨g, hg⟩ = BitVec.ofNat _ v) : val X g = v % 2 ^ (S.gate g).w := by
  rw [val_of_lt X hg, h, BitVec.toNat_ofNat]

section Widths8

variable (ops : RowOps) (p : Pipeline) {M K N TM TN : ℕ}

theorem inW8_a {g : ℕ} (h : g < eaAt M K) : inW8 M N K g = 32 := by
  unfold inW8; rw [ite_eq_left h]

theorem inW8_e {g : ℕ} (h1 : eaAt M K ≤ g) (h2 : g < bAt M K) : inW8 M N K g = 8 := by
  unfold inW8; rw [ite_eq_right (not_lt.2 h1), ite_eq_left h2]

theorem inW8_b {g : ℕ} (h1 : bAt M K ≤ g) (h2 : g < ebAt M N K) : inW8 M N K g = 32 := by
  obtain ⟨e1, -⟩ := layout8 M N K
  unfold inW8; rw [ite_eq_right (by omega), ite_eq_right (not_lt.2 h1), ite_eq_left h2]

theorem inW8_f {g : ℕ} (h1 : ebAt M N K ≤ g) (h2 : g < gAt M N K) : inW8 M N K g = 8 := by
  obtain ⟨e1, e2, -⟩ := layout8 M N K
  unfold inW8; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (not_lt.2 h1), ite_eq_left h2]

theorem inW8_g {g : ℕ} (h1 : gAt M N K ≤ g) : inW8 M N K g = 32 := by
  obtain ⟨e1, e2, e3, e4, e5, -⟩ := layout8 M N K
  unfold inW8; rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]

theorem in8_le_N : in8 M N K ≤ (whole8 ops p M K N TM TN).spec.toCircuit.N := by
  rw [whole8_N]; omega

/-- An input gate's value is its assignment's low `inW8` bits. -/
theorem val_in8 (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {g v : ℕ} (hg : g < in8 M N K)
    (hgN : g < (whole8 ops p M K N TM TN).spec.toCircuit.N) (h : X ⟨g, hgN⟩ = BitVec.ofNat _ v) :
    val X g = v % 2 ^ inW8 M N K g := by
  rw [val_bv X hgN h]
  show v % 2 ^ ((whole8 ops p M K N TM TN).gate g).w = _
  unfold Whole.gate
  rw [ite_eq_left (show g < (whole8 ops p M K N TM TN).nIn from hg)]
  rfl

theorem tile8_lt {e a : ℕ} (he : e < M / TM * (N / TN)) (ha1 : tileIn K TM TN ≤ a) (ha2 : a < tileN K TM TN) :
    tile8 p M N K TM TN e a < (whole8 ops p M K N TM TN).spec.toCircuit.N := by
  rw [tile8_eq ops p M K N TM TN]
  exact (whole8 ops p M K N TM TN).inst_lt (whole8_T ops p M K N TM TN) he ha1 ha2

/-- A tile gate's value is its assignment's low 32 bits. -/
theorem val_tile8 (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {e a v : ℕ}
    (he : e < M / TM * (N / TN)) (ha1 : tileIn K TM TN ≤ a) (ha2 : a < tileN K TM TN)
    (h : X ⟨_, tile8_lt ops p he ha1 ha2⟩ = BitVec.ofNat _ v) : val X (tile8 p M N K TM TN e a) = v % 2 ^ 32 := by
  rw [val_bv X _ h]
  show v % 2 ^ ((whole8 ops p M K N TM TN).gate (tile8 p M N K TM TN e a)).w = _
  rw [tile8_eq ops p M K N TM TN, (whole8 ops p M K N TM TN).gate_inst (whole8_T ops p M K N TM TN) he ha1 ha2]
  show v % 2 ^ (tileGate p K TM TN a).w = _
  rw [tileGate_w p ha1]

omit ops p in
theorem cGate_bounds {i j : ℕ} (hi : i < TM) (hj : j < TN) (hK : 0 < K / 128) :
    tileIn K TM TN ≤ cGate K TM TN i j ∧ cGate K TM TN i j < tileN K TM TN := by
  have hb := entry_bounds K TM TN hi hj (r := 5 * (K / 128) - 1) (by unfold blockLen; omega)
  have hcg : cGate K TM TN i j = entryBase K TM TN i j + (5 * (K / 128) - 1) := by unfold cGate; omega
  rw [hcg]; exact hb

omit ops p in
theorem uGate_bounds {i j : ℕ} (hi : i < TM) (hj : j < TN) :
    tileIn K TM TN ≤ uGate K TM TN i j ∧ uGate K TM TN i j < tileN K TM TN := by
  have hbl := blockLen_pos K
  have hb := entry_bounds K TM TN hi hj (r := blockLen K - 1) (by omega)
  have hug : uGate K TM TN i j = entryBase K TM TN i j + (blockLen K - 1) := by unfold uGate; omega
  rw [hug]; exact hb

end Widths8

/-! ## An accepted entry -/

/-- A whole-tile `pc8` shape: the tile at its own shape divides it, `tileSide m ∣ m` and `tileSide n ∣ n`
(`pc8.Shape.check`), and `k` is a positive multiple of 128. -/
def Shape8 (sh : Shape) : Prop := tileSide sh.m ∣ sh.m ∧ tileSide sh.n ∣ sh.n ∧ ∃ nw, 0 < nw ∧ sh.k = 128 * nw

/-- The tile at a shape divides it where the dimension is at most `64` or a multiple of `64`. -/
theorem tileSide_dvd_iff (d : ℕ) : tileSide d ∣ d ↔ d ≤ 64 ∨ 64 ∣ d := by
  unfold tileSide
  rcases le_total d 64 with h | h
  · rw [min_eq_left h]; exact ⟨fun _ => Or.inl h, fun _ => dvd_refl d⟩
  · rw [min_eq_right h]
    exact ⟨Or.inr, fun h' => h'.elim (fun _ => ⟨1, by omega⟩) id⟩

/-- **The named hypothesis at entry `(i, j)`** of a unit with noise `z`, activations `act` and weights `wt`: the peel's
factor rows exist (`peelRowsP`, which `devAt`'s useful word reads; where it fails the useful value is `none`). -/
structure Pc8EntryOK (ops : RowOps) (p : Pipeline) {m n k : ℕ} (z : Noise m n k Params.pi.r) (act wt : Codes)
    (i : Fin m) (j : Fin n) : Prop where
  peel : (PeelExactProofsP.peelRowsP p 4 (quant8A ops p z act) (quant8B ops p z wt) (basis8 z.FA) (basis8 z.FB)
    (fun i => be8 (betaQ8 ops k (act i)) (z.EA i)) (fun j => be8 (betaQ8 ops k (wt j)) (z.EB j)) i j).isSome

section Accept8

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

/-- **An accepted entry's registered words are the protocol's.** On `pc8`'s whole table at a whole-tile shape: where
every unit in the cone of a set `T` holding entry `(i, j)`'s `C̃` and `U` is correct, the public gates hold the anchors
of the noise `z` and the registered gates the activations `act`, the weights `wt` and the words `tk`, `us`, then at
`Row8OK` rows `i` of `act` and `j` of `wt`, under `Noise8OK z` and the named hypothesis at `(i, j)`, the registered
words `tk i j` and `us i j` read as the device's ticket and useful words on `pc8Sem`'s operands. -/
theorem entry8_accepts (p : Pipeline) (pr : Prices) {M K N TM TN nw : ℕ} (hnw : 0 < nw) (hK : K = 128 * nw)
    (hM : TM ∣ M) (hN : TN ∣ N) (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript)
    (z : Noise M N K Params.pi.r) (act wt tk us : Codes) (i : Fin M) (j : Fin N)
    {T : Finset (Fin (whole8 ops p M K N TM TN).spec.toCircuit.N)}
    (hT : ∀ g : Fin (whole8 ops p M K N TM TN).spec.toCircuit.N,
      (g.val = cGlob8 p M N K TM TN i j ∨ g.val = uGlob8 p M N K TM TN i j) → g ∈ T)
    (hcone : ∀ v ∈ (whole8 ops p M K N TM TN).part.cone T, (whole8 ops p M K N TM TN).part.Correct X v)
    (hpub : ∀ g (hg : g < (whole8 ops p M K N TM TN).spec.toCircuit.N),
      ((eaAt M K ≤ g ∧ g < bAt M K) ∨ (ebAt M N K ≤ g ∧ g < in8 M N K)) → X ⟨g, hg⟩ = BitVec.ofNat _ (anchors8 p z g))
    (hreg : ∀ g (hg : g < (whole8 ops p M K N TM TN).spec.toCircuit.N),
      (g < eaAt M K ∨ (bAt M K ≤ g ∧ g < ebAt M N K) ∨
        ∃ i < M, ∃ j < N, g = cGlob8 p M N K TM TN i j ∨ g = uGlob8 p M N K TM TN i j) →
      X ⟨g, hg⟩ = BitVec.ofNat _ (rows8 p M N K TM TN act wt tk us g))
    (hA : Row8OK ops K (act i)) (hB : Row8OK ops K (wt j)) (hz : Noise8OK z) (hE : Pc8EntryOK ops p z act wt i j) :
    wordQ (tk i j) = (devAt p 4 pr).ticket (quant8A ops p z act) (quant8B ops p z wt) i j ∧
      wordQ (us i j) = (devAt p 4 pr).useful (quant8A ops p z act) (quant8B ops p z wt)
        (basis8 z.FA) (basis8 z.FB) (fun i => be8 (betaQ8 ops K (act i)) (z.EA i))
        (fun j => be8 (betaQ8 ops K (wt j)) (z.EB j)) i j := by
  obtain ⟨⟨c, pa, pb⟩, hr⟩ := Option.isSome_iff_exists.1 hE.peel
  have hK0 : 0 < K / 128 := by subst hK; omega
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := layout8 M N K
  have hNin := in8_le_N ops p (M := M) (K := K) (N := N) (TM := TM) (TN := TN)
  have hea0 : eaAt M K = M * K := rfl
  have hxa : ∀ l < K, val X ((i : ℕ) * K + l) = act i l % 2 ^ 32 := fun l hl => by
    have := mul_add_lt i.isLt hl
    have hg : (i : ℕ) * K + l < eaAt M K := by omega
    rw [val_in8 ops p X (by omega) (by omega) (hreg _ _ (Or.inl hg)), rows8_a p act wt tk us i.isLt hl, inW8_a hg]
  have hxb : ∀ l < K, val X (bAt M K + (j : ℕ) * K + l) = wt j l % 2 ^ 32 := fun l hl => by
    have := mul_add_lt j.isLt hl
    rw [val_in8 ops p X (by omega) (by omega) (hreg _ _ (Or.inr (Or.inl ⟨by omega, by omega⟩))),
      rows8_b p act wt tk us j.isLt hl, inW8_b (by omega) (by omega)]
  have hea : ∀ l < 32, val X (eaAt M K + 32 * (i : ℕ) + l) = encCodes (z.EA i) l := fun l hl => by
    have := i.isLt
    rw [val_in8 ops p X (by omega) (by omega) (hpub _ _ (Or.inl ⟨by omega, by omega⟩)), anchors8_ea p z i.isLt hl,
      inW8_e (by omega) (by omega), Nat.mod_eq_of_lt (encCodes_lt8 _ _)]
  have heb : ∀ l < 32, val X (ebAt M N K + 32 * (j : ℕ) + l) = encCodes (z.EB j) l := fun l hl => by
    have := j.isLt
    rw [val_in8 ops p X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors8_eb p z j.isLt hl,
      inW8_f (by omega) (by omega), Nat.mod_eq_of_lt (encCodes_lt8 _ _)]
  have hfa : ∀ t : Fin K, ∀ l < 32, val X (faAt M N K + 32 * (t : ℕ) + l) = encCodes (z.FA t) l := fun t l hl => by
    have := t.isLt
    rw [val_in8 ops p X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors8_fa p z t.isLt hl,
      inW8_f (by omega) (by omega), Nat.mod_eq_of_lt (encCodes_lt8 _ _)]
  have hfb : ∀ t : Fin K, ∀ l < 32, val X (fbAt M N K + 32 * (t : ℕ) + l) = encCodes (z.FB t) l := fun t l hl => by
    have := t.isLt
    rw [val_in8 ops p X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors8_fb p z t.isLt hl,
      inW8_f (by omega) (by omega), Nat.mod_eq_of_lt (encCodes_lt8 _ _)]
  have hg : ∀ s < 32, ∀ q < 32, val X (gAt M N K + 32 * s + q) = gram8 p z s q := fun s hs q hq => by
    rw [val_in8 ops p X (by omega) (by omega) (hpub _ _ (Or.inr ⟨by omega, by omega⟩)), anchors8_g p z hs hq,
      inW8_g (by omega), Nat.mod_eq_of_lt (gram8_lt p z s q)]
  obtain ⟨w1, w2⟩ := entry8_words hops p pr hnw hK hM hN X z act wt i j
    (hcone _ (cone8T ops p M K N TM TN hM hN i.isLt j.isLt hK0 fun g hg => hT g (Or.inl hg)))
    (hcone _ (cone8A ops p M K N TM TN hM hN i.isLt j.isLt fun g hg => hT g (Or.inr hg)))
    (hcone _ (cone8B ops p M K N TM TN hM hN i.isLt j.isLt fun g hg => hT g (Or.inr hg)))
    hxa hea hxb heb hfa hfb hg hA hB hz hr
  obtain ⟨he, -, -⟩ := tile_index hM hN i.isLt j.isLt
  have hti := Nat.mod_lt (i : ℕ) (Nat.pos_of_dvd_of_pos hM (Nat.zero_lt_of_lt i.isLt))
  have htj := Nat.mod_lt (j : ℕ) (Nat.pos_of_dvd_of_pos hN (Nat.zero_lt_of_lt j.isLt))
  have hc := cGate_bounds (K := K) hti htj hK0
  have hu := uGate_bounds (K := K) hti htj
  have vC : val X (cGlob8 p M N K TM TN i j) = tk i j % 2 ^ 32 := by
    unfold cGlob8
    rw [val_tile8 ops p X he hc.1 hc.2 (hreg _ _ (Or.inr (Or.inr ⟨i, i.isLt, j, j.isLt, Or.inl rfl⟩)))]
    rw [show tile8 p M N K TM TN (i / TM * (N / TN) + j / TN) (cGate K TM TN (i % TM) (j % TN)) =
      cGlob8 p M N K TM TN i j from rfl, rows8_C p act wt tk us hM hN i.isLt j.isLt hK0]
  have vU : val X (uGlob8 p M N K TM TN i j) = us i j % 2 ^ 32 := by
    unfold uGlob8
    rw [val_tile8 ops p X he hu.1 hu.2 (hreg _ _ (Or.inr (Or.inr ⟨i, i.isLt, j, j.isLt, Or.inr rfl⟩)))]
    rw [show tile8 p M N K TM TN (i / TM * (N / TN) + j / TN) (uGate K TM TN (i % TM) (j % TN)) =
      uGlob8 p M N K TM TN i j from rfl, rows8_U p act wt tk us hM hN i.isLt j.isLt]
  rw [vC, wordQ_mod32] at w1
  rw [vU, wordQ_mod32] at w2
  exact ⟨w1, w2⟩

end Accept8

/-! ## The audit grid and `D_dev` -/

theorem auditTiling_rows_lt (tm tn : ℕ) (L : Layout) (g : ℕ) :
    ∀ i ∈ (auditTiling tm tn L).rows g, i < (L.shape ((auditTiling tm tn L).unit g)).m := fun _ hi =>
  lt_of_lt_of_le (Finset.mem_Ico.1 hi).2 (min_le_right _ _)

theorem auditTiling_cols_lt (tm tn : ℕ) (L : Layout) (g : ℕ) :
    ∀ j ∈ (auditTiling tm tn L).cols g, j < (L.shape ((auditTiling tm tn L).unit g)).n := fun _ hj =>
  lt_of_lt_of_le (Finset.mem_Ico.1 hj).2 (min_le_right _ _)

/-- `D_dev`'s `k` is whole-tile for `pc8`, a positive multiple of 128. Its `m` is free and its `n` only a multiple of 64,
so `Shape8` also needs the tile at the unit's shape to divide it. -/
theorem shape8_of_pearlCDomainDev {d : PearlCDevice} {L : Layout} (h : pearlCDomainDev d L) {u : ℕ} (hu : u < L.N)
    (hm : tileSide (L.shape u).m ∣ (L.shape u).m) (hn : tileSide (L.shape u).n ∣ (L.shape u).n) :
    Shape8 (L.shape u) := by
  obtain ⟨-, -, hk0, ⟨c, hc⟩, -, -⟩ := h u hu
  exact ⟨hm, hn, c, by omega, hc⟩

end Pouw.PearlC.WholeCircuit
