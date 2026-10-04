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
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit Pouw.PearlC.PeelExact
  Pouw.PearlC.PeelExactP

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

end Pouw.PearlC.WholeCircuit
