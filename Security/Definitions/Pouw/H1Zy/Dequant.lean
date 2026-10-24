import Definitions.Core.Fp32
import Mathlib.Data.BitVec

/-!
# `PouwZyDequant_v1`'s word view (trusted)

`h1zy.dequant` (`verity/core/protocols/pouw/circuit/h1zy.py`), the served epilogue on encodings: the s32 z to the
nearest f32 (`S32ToF32Rn_v1`, `cvt.rn.f32.s32`), times the activation scale, times the weight scale (`F32Mul_v2` twice,
each one IEEE rounding to nearest even), to the nearest bf16 (`F2fpBf16_v1`, every NaN `0x7FFF`). The f32 and bf16
arithmetic is `Verity.Fp32`'s (`Definitions.Core.Fp32`); the conversion from s32 is `s32ToF32Rn`, the magnitude of the
two's-complement word as an exact value in units of `2^-149`, rounded once by `Verity.Fp32.roundRn`.
-/

namespace Pouw.H1Zy

open Verity.Fp32

/-- **`S32ToF32Rn_v1`** (`cast.s32_to_f32_rn_word`): the 32-bit two's-complement word `a` as the nearest f32 word, ties
to even; zero is `+0`. -/
def s32ToF32Rn (a : ℕ) : ℕ :=
  let a := a % 2 ^ 32
  if a < 2 ^ 31 then roundRn false (a <<< 149) 1 else roundRn true ((2 ^ 32 - a) <<< 149) 1

/-- **`PouwZyDequant_v1`** on encodings (`h1zy.dequant`): `F2fpBf16_v1(F32Mul_v2(F32Mul_v2(S32ToF32Rn_v1(z), sa), sw))`,
`z` an s32 word, `sa` and `sw` f32 encodings, the result a bf16 encoding. -/
def dequant (z sa sw : ℕ) : ℕ :=
  f32ToBf16RnBits (f32MulV2 (f32MulV2 (s32ToF32Rn z) (sa % 2 ^ 32)) (sw % 2 ^ 32)) BF16_NAN_F2FP

/-- `dequant` on words: `LeafChecks`' `dqW` (`Accounting.TS.Lanes.dqOf`). -/
def dequantW (z sa sw : BitVec 32) : BitVec 16 := BitVec.ofNat 16 (dequant z.toNat sa.toNat sw.toNat)

end Pouw.H1Zy
