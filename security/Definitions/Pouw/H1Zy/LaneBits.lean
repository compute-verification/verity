module

public import Definitions.Pouw.H1Zy.Lane

/-!
# The s8 multiply on bits (trusted)

`GemmAccumulatorS8_v2{K, BlackwellS8ImmaDot32_v2}` (`gemm._steps`) at its lowering, each word a `Fin n → Bool`, bit `i`
the word's bit `i` LSB first: `K/32` steps from the zero accumulator (`gemmStepsBits`), each `integer.step_s8_bits` at
sm_120's s8 model (`immaS8Bits`): 32 Baugh–Wooley arrays (`bwBit`: `x i AND y j`, negated where exactly one of `i`, `j`
is 7), the accumulator sign-extended to 33 bits and `32 · _BW_CONSTANT mod 2^33` as constant bits, reduced to 33 bits by
`compress_columns` (stated by its contract, `compressCols`), then `.satfinite`'s clamp on those bits.
-/

@[expose] public section

namespace Pouw.H1Zy

/-- A bit's value. -/
def bval (b : Bool) : ℕ := if b then 1 else 0

/-- Baugh–Wooley's partial product `(i, j)` of `x` and `y`: `x i AND y j`, negated where exactly one of `i`, `j` is the
sign bit 7. -/
def bwBit (x y : Fin 8 → Bool) (i j : Fin 8) : Bool :=
  if (i.val = 7) ≠ (j.val = 7) then !(x i && y j) else (x i && y j)

/-- `_BW_CONSTANT` (`integer.py`): `-2^15 + 2^8`. -/
def bwConst : ℤ := -2 ^ 15 + 2 ^ 8

/-- The weighted sum of `step_s8_bits`' columns: 32 Baugh–Wooley arrays, the accumulator sign-extended to 33 bits, and
`32 · bwConst mod 2^33` as constant bits. -/
def stepColSum (acc : Fin 32 → Bool) (a b : Fin 32 → Fin 8 → Bool) : ℕ :=
  (∑ k, ∑ i : Fin 8, ∑ j : Fin 8, 2 ^ (i.val + j.val) * bval (bwBit (a k) (b k) i j)) +
    (∑ i : Fin 32, 2 ^ i.val * bval (acc i)) + 2 ^ 32 * bval (acc 31) + (32 * bwConst % 2 ^ 33).toNat

/-- `forms.compress_columns`' contract, not its gates: the low `W` bits of the columns' weighted sum. The lowering's
constant folding, Wallace passes and final ripple are not restated here. -/
def compressCols (W S : ℕ) : Fin W → Bool := fun i => S.testBit i.val

/-- **`BlackwellS8ImmaDot32_v2`'s lowering** (`integer.step_s8_bits` at `BLACKWELL_SM120_S8_M16N8K32`, `.satfinite`):
the columns compressed to 33 bits `s`, then clamped: `ovf = s 31 xor s 32`; bit `i < 31` is
`s i xor (ovf and (not s 32 xor s i))`, bit 31 is `s 32`. -/
def immaS8Bits (acc : Fin 32 → Bool) (a b : Fin 32 → Fin 8 → Bool) : Fin 32 → Bool :=
  let s := compressCols 33 (stepColSum acc a b)
  let ovf := xor (s 31) (s 32)
  fun i => if h : i.val < 31 then xor (s ⟨i.val, by omega⟩) (ovf && xor (!s 32) (s ⟨i.val, by omega⟩)) else s 32

/-- `GemmAccumulatorS8_v2`'s lowering after its first `n` steps: step `s` is `immaS8Bits` on codes `32 s .. 32 s + 31`,
from the zero accumulator (`gemm._steps`). -/
def gemmStepsBits {K : ℕ} (x w : Fin K → Fin 8 → Bool) : ℕ → Fin 32 → Bool
  | 0 => fun _ => false
  | n + 1 => immaS8Bits (gemmStepsBits x w n)
      (fun j => if h : 32 * n + j.val < K then x ⟨32 * n + j.val, h⟩ else fun _ => false)
      (fun j => if h : 32 * n + j.val < K then w ⟨32 * n + j.val, h⟩ else fun _ => false)

end Pouw.H1Zy
