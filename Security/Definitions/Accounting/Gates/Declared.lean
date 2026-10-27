import Definitions.Accounting.Gates.Ranges

/-!
# Declared bounds, earned: what a declared interval may rest on, and the int8 circuits that need nothing else (trusted)

A declared bound is never taken on trust. An input is declared at its type's whole range (`TypeRange`), so the one thing
asked of an input assignment is that each slot carry a word of its type (`Typed`). A budgeted gate's declared interval
holds its evaluator's value on every operand in that operand's derived interval (`Declared`): the evaluator's type width
(a saturating cast, a word read at its width), a producer's proven interval (`Ranges`), or a range check's clamp, which
is budgeted. A run input narrower than its type gets its bound only through such a gate. Gathers and the free gates are
derived (`Proven`), never declared.

Two circuits of the exact-int8 profile (`verity_vllm/program/registry/int8_exact.py`) whose window then holds by
theorem: `int8Linear`, one coordinate of `Int8Linear{K,N}` (`Int8RowQuant`, then `ScaledMmS8_v1`'s s8·s8 dot), and
`lutPv`, one channel of `Int8LutAttentionRow`'s `P V'` (P from `LutIndex`, `Exp2Lut127` and `S32LowByte`, V' an s8
code, both zero-padded to `32·S` keys). The 32-bit primitives whose value no bound reads (`F32Div`, `F32Rint`, the
clamp's `F32Fmaxf` and `F32Fminf`, the row scale's fold, each reading the row's `Bf16ToF32` widening, an affine
`×2^16` folded into them) are each `word32 f` for an arbitrary `f`: the window holds whatever they compute.
-/

namespace Accounting.Gates

variable {ι : Type}

/-- A type's whole range: a signed word's `[−2^w, 2^w − 1]` or an unsigned word's `[0, 2^w − 1]`. -/
def TypeRange (lo hi : ℤ) : Prop := ∃ w : ℕ, (lo = -2 ^ w ∧ hi = 2 ^ w - 1) ∨ (lo = 0 ∧ hi = 2 ^ w - 1)

/-- **Every declared bound earned**: every input is declared at its type's whole range, and every budgeted gate's
declared interval holds its evaluator's value on every operand in its operand's derived interval. -/
def Declared (C : Dag ι) : Prop :=
  ∀ i (h : i < C.length),
    (∀ b s, C[i].op = .input b s → TypeRange C[i].lo C[i].hi) ∧
      ∀ cls f, C[i].op = .budgeted cls f → ∀ vs : List ℤ,
        (∀ j ∈ C[i].args, ((intervals (C.take i)).getD j (0, 0)).Mem (vs.getD j 0)) →
          Iv.Mem (C[i].lo, C[i].hi) (f (C[i].args.map fun j => vs.getD j 0))

/-- Each input slot carries a word of its type: its value lies in the declared range, its type's, of every input gate
that reads it. -/
def Typed (C : Dag ι) (x : ι → ℤ) : Prop := ∀ g ∈ C, ∀ b s, g.op = .input b s → g.lo ≤ x s ∧ x s ≤ g.hi

/-- A 32-bit primitive no bound reads: any function `f` of its operands, read as the s32 word it returns. -/
def word32 (f : List ℤ → ℤ) : Op ι := .budgeted .rounding fun v => wrapS32 (f v)

/-- A gate of no particular place, at engine step 0, uncommitted. -/
def plainGate (op : Op ι) (args : List ℕ) (lo hi : ℤ) (prim : String) : Gate ι :=
  ⟨op, args, lo, hi, 0, prim, .other, false⟩

/-- **An s8-family dot chain** on top of the gates `pre`: the accumulator's constant 0 at `pre.length`, then step `s`
of `S` (32 products each, `K = 32·S`, committed once at `T_1`) at `pre.length + 1 + s`, reading the accumulator before
it and its operands `a s` and `b s`, each declared s32. -/
def dotChain (pre : Dag ι) (fa fb : Int8Fmt) (sat : Bool) (prim : String) (S : ℕ) (a b : ℕ → List ℕ) : Dag ι :=
  pre ++ (plainGate (.const 0) [] 0 0 "" :: (List.range S).map fun s =>
    ⟨.mma fa fb 32 sat, (pre.length + s) :: (a s ++ b s), s32Min, s32Max, 0, prim, .dot (32 * S) (32 * S) [] s,
      decide (s + 1 = S)⟩)

/-- `int8Linear`'s gate `i`, `K = 32·S`: the token row's bf16 bits `x_k` (run inputs, `i < K`), its scale (`i = K`,
`Int8RowScale`), each `rint(RN32(x_k/s))` clamped to `±127.0` (`K < i ≤ 2K`), its s8 code (`F32ToS8Rzi`,
`2K < i ≤ 3K`), and the weight row's s8 codes `W_nk` (registered inputs, `3K < i ≤ 4K`). -/
def linearGate (S : ℕ) (fs fr : List ℤ → ℤ) (i : ℕ) : Gate ℕ :=
  if i < 32 * S then plainGate (.input .run i) [] (-2 ^ 15) (2 ^ 15 - 1) ""
  else if i = 32 * S then plainGate (word32 fs) (List.range (32 * S)) s32Min s32Max ""
  else if i ≤ 2 * (32 * S) then plainGate (word32 fr) [i - 32 * S - 1, 32 * S] s32Min s32Max ""
  else if i ≤ 3 * (32 * S) then
    plainGate (.budgeted .rounding fun v => f32ToS8Rzi (v.headD 0)) [i - 32 * S] (-128) 127 "F32ToS8Rzi_v1"
  else plainGate (.input .registered i) [] (-128) 127 ""

/-- **The int8 linear's coordinate** (`Int8Linear{K,N}` at output `n`, `K = 32·S`): `Int8RowQuant`'s codes, then
`ScaledMmS8Coordinate_v1`'s `Y_n = Σ_k X_k W_nk` in `S` satfinite steps of `BlackwellS8ImmaDot32_v1` from 0. The
row's scale and the quotient's rounding and clamp are `fs` and `fr`, any functions. -/
def int8Linear (S : ℕ) (fs fr : List ℤ → ℤ) : Dag ℕ :=
  dotChain ((List.range (4 * (32 * S) + 1)).map (linearGate S fs fr)) .s8 .s8 true "BlackwellS8ImmaDot32_v1" S
    (fun s => (List.range 32).map fun t => 2 * (32 * S) + 1 + 32 * s + t)
    (fun s => (List.range 32).map fun t => 3 * (32 * S) + 1 + 32 * s + t)

/-- `lutPv`'s gate `i`, for `n` keys padded to `K = 32·S`: the table `T`'s 256 entries (`i < 256`), A (`i = 256`), each
key's `d_j = m − S_j` (`j < n`), its `LutIndex(d_j, A)`, its `Exp2Lut127` entry `e_j` (a gather of the index's low 8
bits), then `P = (0, e_1 … e_{n−1})` as s8 codes (`S32LowByte`) zero-padded to `K`, and V'`[c]`'s s8 codes
(`Int8KvQuant`'s, run inputs) zero-padded to `K`. -/
def pvGate (S n : ℕ) (T : List ℤ) (i : ℕ) : Gate ℕ :=
  if i < 256 then plainGate (.const (T.getD i 0)) [] (T.getD i 0) (T.getD i 0) ""
  else if i < 257 + n then plainGate (.input .run i) [] s32Min s32Max ""
  else if i < 257 + 2 * n then
    plainGate (.budgeted .rounding fun v => lutIndex (v.headD 0) (v.getD 1 0)) [i - n, 256] 0 255 "LutIndex_v1"
  else if i < 257 + 3 * n then plainGate (.gather (some 8) 0) ((i - n) :: List.range 256) 0 127 "Exp2Lut127_v1"
  else if i < 257 + 3 * n + 32 * S then
    if 1 ≤ i - (257 + 3 * n) ∧ i - (257 + 3 * n) < n then
      plainGate (.budgeted .rounding fun v => s32LowByte (v.headD 0)) [i - n] 0 127 "S32LowByte_v1"
    else plainGate (.const 0) [] 0 0 ""
  else if i - (257 + 3 * n + 32 * S) < n then plainGate (.input .run i) [] (-128) 127 ""
  else plainGate (.const 0) [] 0 0 ""

/-- **The LUT attention's `P V'`** at one channel (`Int8LutAttentionRow`'s `S8Matmul_v1{D, S'}`, `S' = 32·S` padded
keys, `n` of them real): `Y_c = Σ_j P_j V'_jc` in `S` satfinite steps of `BlackwellS8ImmaDot32_v1` from 0, with the
scores' differences and A as s32 inputs (any words: `LutIndex` reads them only through its index). -/
def lutPv (S n : ℕ) (T : List ℤ) : Dag ℕ :=
  dotChain ((List.range (257 + 3 * n + 2 * (32 * S))).map (pvGate S n T)) .s8 .s8 true "BlackwellS8ImmaDot32_v1" S
    (fun s => (List.range 32).map fun t => 257 + 3 * n + 32 * s + t)
    (fun s => (List.range 32).map fun t => 257 + 3 * n + 32 * S + 32 * s + t)

end Accounting.Gates
