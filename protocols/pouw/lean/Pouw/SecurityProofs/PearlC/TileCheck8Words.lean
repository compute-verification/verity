import Pouw.SecurityProofs.PearlC.TileCheck8
import Pouw.SecurityProofs.PearlC.TileProtocol

/-!
# `pc8`'s tile check computes the words (proofs)

The row units' words read in the protocol's terms (`pc8Sem`), under the named hypotheses on which the table and the
protocol's word functions agree only conditionally:

* `Pc8OpsAgree ops`: core's primitives (`F32ToE4m3Sat`, `E4m3ToF32`, `F32Mul`, `F32Fma`, `F32Neg`, `F2fpBf16`) agree
  with the protocol's word functions where both are defined. Core's Lean semantics will discharge them;
* `Row8OK`: a row's words are step words (no `−0`), as are its `α` and `β`, and `β > 0`;
* `Line8OK`: the noise lines' entries are E4M3 values, and their rescaled codes `E′` are neither NaN nor `−0`. The
  protocol reads `bf16(E′)` as `bf16WordOf (codeVal E′)`, which loses `−0`'s sign, while the table keeps it;
* `CastWord` of the formed words, so the cast is `e4m3` of the formed value.

On them a row unit's codes are `pc8Sem`'s (`quant8`, so `quant8A`/`quant8B` are `unitChain`'s `A′`/`B̃`) and its
factor row is the peel's (`row8A_paWords`, `row8B_pbWords`). `entry8_words` chains RowA `i` and RowB `j` into their
tile (`TileCheck8`'s `chain8_*`): where the three units are correct and the peel's factor rows exist, entry `(i, j)`'s
`C̃` and `U` are the device's ticket and useful words (`tile_devAt`).
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

theorem stepWord_getD {w : ℕ} (hw : AccumProofs.StepWord w) : (wordQ w).getD 0 = wordVal w := by
  rw [stepWord_wordQ hw]; rfl

theorem signBit_of_pos {w : ℕ} {y : ℚ} (h : wordQ w = some y) (hy : 0 < y) : signBit w = false := by
  have := (wordQ_neg_iff h (ne_of_gt hy)).not.1 (not_lt.2 hy.le)
  unfold signBit
  simpa using this

end Words

/-! ## The named hypotheses -/

/-- **Core's primitives agree with the protocol's word functions** where both are defined: the E4M3 cast is
`e4m3Word`; the widening of a code other than NaN and `−0` is its value's FP32 word; FP32 multiply and FMA round as
`f32Word`, `fmaWord?` (a zero product of equal signs is `+0`); negation flips the sign bit; the BF16 narrowing is
`rnBf16?`. Each is a fact about IEEE binary32 (round to nearest even, gradual underflow), open until core's Lean
semantics of the primitives. -/
structure Pc8OpsAgree (ops : RowOps) : Prop where
  cast : ∀ u, ops.f32ToE4m3Sat u = (e4m3Word u).val
  widen : ∀ c : Code, c.isNaN = false → c.val ≠ 0x80 → ops.e4m3ToF32 c.val % 2 ^ 32 = f32Word (codeVal c)
  mul : ∀ a b : ℕ, ∀ x y : ℚ, a < 2 ^ 32 → b < 2 ^ 32 → wordQ a = some x → wordQ b = some y →
    (x * y = 0 → signBit a = signBit b) → ops.f32MulV2 a b % 2 ^ 32 = f32Word (x * y)
  fma : ∀ a b c w : ℕ, a < 2 ^ 32 → b < 2 ^ 32 → c < 2 ^ 32 → fmaWord? a b c = some w →
    ops.f32FmaV2 a b c % 2 ^ 32 = w
  neg : ∀ w < 2 ^ 32, ops.f32Neg w % 2 ^ 32 = negWord w
  bf16 : ∀ w x : ℕ, w < 2 ^ 32 → rnBf16? w = some x → ops.f2fpBf16 w % 2 ^ 16 = x

/-- A noise entry the table reads exactly: its E4M3 code is neither NaN nor `−0`, and has its value. -/
def NoiseCode (v : ℚ) : Prop := (e4m3 v).isNaN = false ∧ (e4m3 v).val ≠ 0x80 ∧ codeVal (e4m3 v) = v

/-- A noise line rescaled by `β` as the table reads it: every entry a `NoiseCode`, and every rescaled code `E′` neither
NaN nor `−0`. -/
def Line8OK (β : ℚ) (e : Fin Params.pi.r → ℚ) : Prop :=
  ∀ q, NoiseCode (e q) ∧ (lineCodes β e q).isNaN = false ∧ (lineCodes β e q).val ≠ 0x80

/-- A row of words as values, as `wordsVal` reads it. -/
def rowQ (k : ℕ) (w : ℕ → ℕ) : Fin k → ℚ := fun l => (wordQ (w l)).getD 0

/-- A `pc8` row the table reads as `pc8Sem` does: its words step words (so each is its value's word), its `α` and `β`
step words, and `β > 0`. -/
def Row8OK (ops : RowOps) (k : ℕ) (w : ℕ → ℕ) : Prop :=
  (∀ l < k, AccumProofs.StepWord (w l)) ∧ AccumProofs.StepWord (alpha8 ops (encRow k (rowQ k w)) k) ∧
    AccumProofs.StepWord (beta8 ops (encRow k (rowQ k w)) k) ∧ 0 < wordVal (beta8 ops (encRow k (rowQ k w)) k)

/-- The formed word `pc8Form` reads as a value. -/
def formWord8 (ops : RowOps) (p : Pipeline) (α x β : ℚ) (e f : Fin Params.pi.r → ℚ) : ℕ :=
  form8 ops p (f32Word α) (f32Word x) (fun l => eTilde8 ops (f32Word β) (encCodes e l)) (encCodes f)

/-! ## A row's words in the protocol's terms -/

section Row8

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

/-- **A rescaled noise code is `lineCodes`'s**: `eTilde8` at `β`'s step word and an exact noise entry's code. -/
theorem eTilde8_line {b : ℕ} (hb : AccumProofs.StepWord b) (hpos : 0 < wordVal b) {e : Fin Params.pi.r → ℚ}
    (q : Fin Params.pi.r) (he : NoiseCode (e q)) : eTilde8 ops b (encCodes e q) = (lineCodes (wordVal b) e q).val := by
  obtain ⟨hn, h80, hv⟩ := he
  have hβ := stepWord_wordQ hb
  have hw : Prim.eval ops .e4m3ToF32 [encCodes e q] = f32Word (e q) := by
    simp only [Prim.eval, Prim.w, List.getD_cons_zero, encCodes, q.isLt, ↓reduceDIte]
    rw [hops.widen _ hn h80, hv]
  have hm : Prim.eval ops .mul [b, f32Word (e q)] = f32Word (wordVal b * e q) := by
    simp only [Prim.eval, Prim.w, List.getD_cons_zero, List.getD_cons_succ]
    refine hops.mul _ _ _ _ hb.1 (f32Word_lt _) hβ (by rw [← hv]; exact wordQ_f32Word_code _ hn) fun h0 => ?_
    have h0' : e q = 0 := (mul_eq_zero.1 h0).resolve_left (ne_of_gt hpos)
    rw [signBit_of_pos hβ hpos, h0']
    rfl
  unfold eTilde8 lineCodes
  rw [hw, hm]
  simp only [Prim.eval, Prim.w, List.getD_cons_zero, hops.cast]
  exact Nat.mod_eq_of_lt (e4m3Word _).isLt

/-- **`bf16(E′)`** of a code other than NaN and `−0` is `bf16WordOf` of its value. -/
theorem be8_word {c : Code} (hn : c.isNaN = false) (h80 : c.val ≠ 0x80) :
    Prim.eval ops .f2fpBf16 [Prim.eval ops .e4m3ToF32 [c.val]] = bf16WordOf (codeVal c) := by
  have hw : Prim.eval ops .e4m3ToF32 [c.val] = f32Word (codeVal c) := by
    simp only [Prim.eval, Prim.w, List.getD_cons_zero]
    exact hops.widen c hn h80
  rw [hw]
  simp only [Prim.eval, Prim.w, List.getD_cons_zero]
  exact hops.bf16 _ _ (f32Word_lt _) (rnBf16_f32Word_code c hn)

/-- **A row's code is `pc8Sem`'s**: `code8W` on a row holding `encRow`'s words, the noise line's and a basis row's codes
is `e4m3` of `pc8Form` at `alpha8`'s and `beta8`'s values, the entry and the two rows, where the formed word is a
`CastWord`. -/
theorem code8W_sem (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) {V : ℕ → ℕ} {x : Fin K → ℚ}
    (hx : ∀ l : Fin K, V l = encRow K x l) (ha : AccumProofs.StepWord (alpha8 ops (encRow K x) K))
    (hb : AccumProofs.StepWord (beta8 ops (encRow K x) K)) (e f : Fin Params.pi.r → ℚ) (t : Fin K)
    (he : ∀ l < 32, V (K + l) = encCodes e l) (hf : ∀ l < 32, V (K + 32 + 32 * t + l) = encCodes f l)
    (hD : CastWord (formWord8 ops p (wordVal (alpha8 ops (encRow K x) K)) (x t)
      (wordVal (beta8 ops (encRow K x) K)) e f)) :
    code8W ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
        (fun l => V (K + 32 + 32 * t + l)) =
      (e4m3 ((pc8Form ops p).value (wordVal (alpha8 ops (encRow K x) K)) (x t)
        (wordVal (beta8 ops (encRow K x) K)) e f)).val := by
  have hV : ∀ l < K, V l = encRow K x l := fun l hl => hx ⟨l, hl⟩
  have hform : form8 ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
      (fun l => V (K + 32 + 32 * t + l)) =
        formWord8 ops p (wordVal (alpha8 ops (encRow K x) K)) (x t) (wordVal (beta8 ops (encRow K x) K)) e f := by
    unfold formWord8
    rw [f32Word_wordVal ha, f32Word_wordVal hb, alpha8_congr (ops := ops) hK hV, beta8_congr (ops := ops) hK hV,
      hx t]
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
def quant8 {k : ℕ} (w : ℕ → ℕ) (e : Fin Params.pi.r → ℚ) (F : Matrix (Fin k) (Fin Params.pi.r) ℚ) : Fin k → Code :=
  fun l => e4m3 ((pc8Form ops p).value (alphaQ8 ops k w) (rowQ k w l) (betaQ8 ops k w) e (F l))

/-- `A′` as `pc8Sem`'s `unitChain` forms it from the activations' rows and the noise `z`. -/
def quant8A {m n k : ℕ} (z : Noise m n k Params.pi.r) (act : Codes) : Matrix (Fin m) (Fin k) Code :=
  fun i => quant8 ops p (act i) (z.EA i) z.FA

/-- `B̃` as `pc8Sem`'s `unitChain` forms it from the weights' rows and the noise `z`. -/
def quant8B {m n k : ℕ} (z : Noise m n k Params.pi.r) (wt : Codes) : Matrix (Fin n) (Fin k) Code :=
  fun j => quant8 ops p (wt j) (z.EB j) z.FB

end Sem8

/-- A basis' entries as E4M3 codes. -/
def basis8 {k : ℕ} (F : Matrix (Fin k) (Fin Params.pi.r) ℚ) : Matrix (Fin k) (Fin 32) Code := fun l q => e4m3 (F l q)

/-- A noise line's `bf16(E′)` words as the protocol reads them, `E′ = lineCodes β e`. -/
def be8 (β : ℚ) (e : Fin Params.pi.r → ℚ) : Fin 32 → ℕ := fun q => bf16WordOf (codeVal (lineCodes β e q))

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

theorem encRow_rowQ {K : ℕ} {w : ℕ → ℕ} (hw : ∀ l < K, AccumProofs.StepWord (w l)) :
    ∀ l < K, encRow K (rowQ K w) l = w l := by
  intro l hl
  simp only [encRow, hl, ↓reduceDIte, rowQ, stepWord_getD (hw l hl), f32Word_wordVal (hw l hl)]

section RowWords

variable {ops : RowOps} (hops : Pc8OpsAgree ops)
include hops

/-- **A row unit's code is `quant8`'s**: `code8W` on a row holding the row's words, its noise line's and a form basis
row's codes. -/
theorem code8_quant (p : Pipeline) {K : ℕ} (hK : 8 ≤ K) {V w : ℕ → ℕ} (hx : ∀ l < K, V l = w l)
    (hA : Row8OK ops K w) (e : Fin Params.pi.r → ℚ) (F : Matrix (Fin K) (Fin Params.pi.r) ℚ)
    (he : ∀ l < 32, V (K + l) = encCodes e l) (t : Fin K) (hf : ∀ l < 32, V (K + 32 + 32 * t + l) = encCodes (F t) l)
    (hD : CastWord (formWord8 ops p (alphaQ8 ops K w) (rowQ K w t) (betaQ8 ops K w) e (F t))) :
    code8W ops p (alpha8 ops V K) (V t) (fun l => eTilde8 ops (beta8 ops V K) (V (K + l)))
        (fun l => V (K + 32 + 32 * t + l)) = (quant8 ops p w e F t).val :=
  code8W_sem hops p hK (x := rowQ K w) (fun l => by rw [hx l l.isLt, encRow_rowQ hA.1 l l.isLt]) hA.2.1 hA.2.2.1 e
    (F t) t he hf hD

/-- `bf16(E′)` of a noise code: its value's widening. -/
theorem widen8_word {c : Code} (hn : c.isNaN = false) (h80 : c.val ≠ 0x80) :
    Prim.eval ops .e4m3ToF32 [c.val] = f32Word (codeVal c) := by
  simp only [Prim.eval, Prim.w, List.getD_cons_zero]
  exact hops.widen c hn h80

/-- **A row unit's rescaled noise code is `lineCodes`'s** at `pc8Sem`'s `β` of the row's words. -/
theorem eTilde8_row {K : ℕ} (hK : 8 ≤ K) {V w : ℕ → ℕ} (hx : ∀ l < K, V l = w l) (hA : Row8OK ops K w)
    {e : Fin Params.pi.r → ℚ} (hL : Line8OK (betaQ8 ops K w) e) (he : ∀ l < 32, V (K + l) = encCodes e l) {s : ℕ}
    (hs : s < 32) : eTilde8 ops (beta8 ops V K) (V (K + s)) = (lineCodes (betaQ8 ops K w) e ⟨s, hs⟩).val := by
  have hV : ∀ l < K, V l = encRow K (rowQ K w) l := fun l hl => by rw [hx l hl, encRow_rowQ hA.1 l hl]
  rw [beta8_congr (ops := ops) hK hV, he s hs]
  exact eTilde8_line hops hA.2.2.1 hA.2.2.2 ⟨s, hs⟩ (hL ⟨s, hs⟩).1

end RowWords

theorem encCodes_of_lt {e : Fin Params.pi.r → ℚ} {l : ℕ} (hl : l < 32) : encCodes e l = (e4m3 (e ⟨l, hl⟩)).val := by
  unfold encCodes
  exact dite_eq_left hl

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
    (hX : ((row8A p (128 * nw)).spec ops).part.Correct XA 0) {w : ℕ → ℕ} {e : Fin Params.pi.r → ℚ}
    {F Fc : Matrix (Fin (128 * nw)) (Fin Params.pi.r) ℚ} (hx : ∀ l < 128 * nw, val XA l = w l)
    (he : ∀ l < 32, val XA (128 * nw + l) = encCodes e l)
    (hf : ∀ t : Fin (128 * nw), ∀ l < 32, val XA (128 * nw + 32 + 32 * t + l) = encCodes (F t) l)
    (hb : ∀ t : Fin (128 * nw), ∀ q < 32, val XA (33 * (128 * nw) + 32 + 32 * t + q) = encCodes (Fc t) q)
    (hA : Row8OK ops (128 * nw) w) (hL : Line8OK (betaQ8 ops (128 * nw) w) e)
    (hD : ∀ t : Fin (128 * nw), CastWord (formWord8 ops p (alphaQ8 ops (128 * nw) w) (rowQ (128 * nw) w t)
      (betaQ8 ops (128 * nw) w) e (F t)))
    {af pa : List ℕ}
    (haf : (List.ofFn fun q : Fin 32 => chainWordP? p 4 (rowAtoms (quant8 ops p w e F) ((basis8 Fc)ᵀ q))).mapM id =
      some af)
    (hpa : paRow? (List.ofFn (be8 (betaQ8 ops (128 * nw) w) e)) af = some pa) :
    (∀ t : Fin (128 * nw), val XA (code8 (128 * nw) (rowIn (128 * nw) false) t) = (quant8 ops p w e F t).val) ∧
      ∀ q < 64, val XA (paOut8 (128 * nw) q) = pa.getD q 0 := by
  have hK8 : 8 ≤ 128 * nw := by omega
  have hcode : ∀ t : Fin (128 * nw),
      val XA (code8 (128 * nw) (rowIn (128 * nw) false) t) = (quant8 ops p w e F t).val := fun t =>
    (row8A_code (ops := ops) p hK8 XA hX t.isLt).trans (code8_quant hops p hK8 hx hA e F he t (hf t) (hD t))
  refine ⟨hcode, fun q hq => ?_⟩
  obtain ⟨hpa1, hpa2⟩ := PeelExactProofs.paRow_spec hpa List.length_ofFn
  obtain ⟨hafl, hafq⟩ := PeelExactProofs.mapM_ofFn _ 0 haf
  have hdiv : 128 * nw / 128 = nw := by omega
  unfold paOut8
  by_cases hq32 : q < 32
  · rw [ite_eq_left hq32, hdiv, row8A_be p hnw rfl XA hX hq32, eTilde8_row hops hK8 hx hA hL he hq32,
      be8_word hops (hL _).2.1 (hL _).2.2, hpa1 q hq32, getD_ofFn32 _ hq32]
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
    (hX : ((row8B p (128 * nw)).spec ops).part.Correct XB 0) {w : ℕ → ℕ} {e : Fin Params.pi.r → ℚ}
    {F Fc : Matrix (Fin (128 * nw)) (Fin Params.pi.r) ℚ} (hx : ∀ l < 128 * nw, val XB l = w l)
    (he : ∀ l < 32, val XB (128 * nw + l) = encCodes e l)
    (hf : ∀ t : Fin (128 * nw), ∀ l < 32, val XB (128 * nw + 32 + 32 * t + l) = encCodes (F t) l)
    (hb : ∀ t : Fin (128 * nw), ∀ q < 32, val XB (33 * (128 * nw) + 32 + 32 * t + q) = encCodes (Fc t) q)
    (hA : Row8OK ops (128 * nw) w) (hL : Line8OK (betaQ8 ops (128 * nw) w) e)
    (hD : ∀ t : Fin (128 * nw), CastWord (formWord8 ops p (alphaQ8 ops (128 * nw) w) (rowQ (128 * nw) w t)
      (betaQ8 ops (128 * nw) w) e (F t)))
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
    (row8B_code (ops := ops) p hK8 XB hX t.isLt).trans (code8_quant hops p hK8 hx hA e F he t (hf t) (hD t))
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
          f32Word (codeVal (lineCodes (betaQ8 ops (128 * nw) w) e ⟨s, hs⟩)) := fun s hs => by
      rw [eTilde8_row hops hK8 hx hA hL he hs, widen8_word hops (hL _).2.1 (hL _).2.2]
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
        exact (bf16_shift_code _ (hL _).2.1).symm
      · dsimp only
        rw [hew s hs]
        exact f32Word_lt _
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
      be8_word hops (hL _).2.1 (hL _).2.2, h2]
    have hx16 := bf16WordOf_lt (codeVal (lineCodes (betaQ8 ops (128 * nw) w) e ⟨q - 32, hq'⟩))
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
    (hxa : ∀ l < K, val X ((i : ℕ) * K + l) = act i l)
    (hea : ∀ l < 32, val X (eaAt M K + 32 * (i : ℕ) + l) = encCodes (z.EA i) l)
    (hxb : ∀ l < K, val X (bAt M K + (j : ℕ) * K + l) = wt j l)
    (heb : ∀ l < 32, val X (ebAt M N K + 32 * (j : ℕ) + l) = encCodes (z.EB j) l)
    (hfa : ∀ t : Fin K, ∀ l < 32, val X (faAt M N K + 32 * (t : ℕ) + l) = encCodes (z.FA t) l)
    (hfb : ∀ t : Fin K, ∀ l < 32, val X (fbAt M N K + 32 * (t : ℕ) + l) = encCodes (z.FB t) l)
    (hg : ∀ s < 32, ∀ q < 32, val X (gAt M N K + 32 * s + q) = gram8 p z s q)
    (hA : Row8OK ops K (act i)) (hB : Row8OK ops K (wt j))
    (hLA : Line8OK (betaQ8 ops K (act i)) (z.EA i)) (hLB : Line8OK (betaQ8 ops K (wt j)) (z.EB j))
    (hDA : ∀ t : Fin K, CastWord (formWord8 ops p (alphaQ8 ops K (act i)) (rowQ K (act i) t) (betaQ8 ops K (act i))
      (z.EA i) (z.FA t)))
    (hDB : ∀ t : Fin K, CastWord (formWord8 ops p (alphaQ8 ops K (wt j)) (rowQ K (wt j) t) (betaQ8 ops K (wt j))
      (z.EB j) (z.FB t)))
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
    hA hLA hDA haf hpa
  obtain ⟨codeB, pbB⟩ := row8B_pbWords hops p hnw _
    ((whole8 ops p M (128 * nw) N TM TN).cut_correct X (blockB8 ops p M (128 * nw) N) j)
    (fun l hl => (hinB l (by omega)).trans (by rw [portB8_x hl, hxb l hl]))
    (fun l hl => (hinB _ (by omega)).trans (by rw [portB8_e hl, heb l hl]))
    (fun t l hl => (hinB _ (by have := t.isLt; omega)).trans (by rw [portB8_f t.isLt hl, hfb t l hl]))
    (fun t q hq => (hinB _ (by have := t.isLt; omega)).trans (by rw [portB8_b t.isLt hq, hfa t q hq]))
    hB hLB hDB hbf hgl
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

end Pouw.PearlC.WholeCircuit
