/-!
# FP32 arithmetic: `verity.ml.fp32`

The FP32 and BF16 primitives of `packages/verity/src/verity/ml/fp32.py` as total functions on words, in
`Verity.TC.Spec`'s encoding: an FP32 word is a natural number below `2^32`, a BF16 word one below `2^16`, and the bit
operations are Python's. Each definition is named after its primitive and version (`F32Add` v2 is `f32AddV2`, `F32Fma`
v1 is `f32Fma`) and equals the primitive's evaluator on every word of its domain.

* **The `_v2` arithmetic** (`f32AddV2`, `f32MulV2`, `f32DivV2`, `f32FmaV2`, and the flush-to-zero `f32AddFtzV2`,
  `f32SubFtzV2`, `f32MulFtzV2`, `f32FmaFtzV2`, `f32FmaSubFtzV2`): IEEE 754 binary32 with one rounding to nearest, ties
  to even, subnormals kept unless the name says `Ftz`; every NaN result is the GPU's `F32_NAN_PTX`.
* **The `_v1` forms whose words don't depend on the host** (`f32Fma`, `f32FmaFtz`, `f32FmaSubFtz`, `f32Max`,
  `guardNegInfZero`, and the BF16 element-wise `bf16Add`, `bf16AddF2fp`, `bf16MulF32`, `bf16MulBf16`). The other `_v1`
  arithmetic returns the host's NaN payloads, which no definition here states.

Every finite word is an integer multiple of the least subnormal `2^-149` (`decode`), so the exact sum, product, quotient
or fused `a * b + c` of finite words is a rational `n / d` in those units, and `roundRn` (`fp32._round_f32`) takes it to
the nearest word. The Python computes add, sub, mul and div in numpy `float32` (the x86-64 host's IEEE arithmetic) and
the FMA in exact rational arithmetic; `Verity.SecurityProofs.Fp32.Vectors` holds both to these definitions.
-/

namespace Verity.Fp32

/-- `scalar.F32_NAN_PTX`: the GPU's word for every NaN result of the FP32 arithmetic. -/
def F32_NAN_PTX : Nat := 0x7FFFFFFF

/-- `fp32.F32_NAN`: the `_v1` FMAs' NaN. -/
def F32_NAN : Nat := 0x7FC00000

/-- `fp32.BF16_NAN_TORCH`: the aten conversion's canonical BF16 NaN. -/
def BF16_NAN_TORCH : Nat := 0x7FC0

/-- `fp32.BF16_NAN_F2FP`: SASS `F2FP.BF16.F32`'s canonical BF16 NaN. -/
def BF16_NAN_F2FP : Nat := 0x7FFF

/-- The zero of a sign. -/
def zeroWord (negative : Bool) : Nat := if negative then 0x80000000 else 0

/-- The infinity of a sign. -/
def infWord (negative : Bool) : Nat := zeroWord negative ||| 0x7F800000

/-- `fp32._decode`: a NaN, a signed infinity, or a finite word's sign and magnitude in units of `2^-149`. -/
inductive Decoded where
  | nan
  | inf (negative : Bool)
  | fin (negative : Bool) (magnitude : Nat)

/-- `fp32._decode` on words: the magnitude of a finite word is its significand (the mantissa, with the hidden bit when
the exponent field `e` is nonzero) times `2^(max(e, 1) - 1)` units. -/
def decode (u : Nat) : Decoded :=
  let e := (u >>> 23) &&& 0xFF
  let m := u &&& 0x7FFFFF
  let negative := (u >>> 31) &&& 1 == 1
  if e = 0xFF then (if m = 0 then .inf negative else .nan)
  else if e = 0 then .fin negative m
  else .fin negative ((m ||| 0x800000) <<< (e - 1))

/-- `fp32._round_f32` on `(-1)^negative * n / d` units of `2^-149`, `d > 0`: the nearest word, ties to even. The words
of the binade of `n / d` are `2^j` units apart, `j = max(0, ⌊log₂(n / d)⌋ - 23)`; the rounded significand `q` then
lies in `[2^23, 2^24]` (below `2^23` only when `j = 0`, a subnormal), and the word is `j * 2^23 + q`, which carries a
significand of `2^24` into the next binade and is `±inf` from `0x7F800000` up. A magnitude below half the least
subnormal is the zero of its sign. -/
def roundRn (negative : Bool) (n d : Nat) : Nat :=
  let j := (n / d).log2 - 23
  let unit := d <<< j
  let q := n / unit
  let r := n % unit
  let q := if 2 * r > unit ∨ (2 * r = unit ∧ q % 2 = 1) then q + 1 else q
  let word := (j <<< 23) + q
  zeroWord negative ||| (if word ≥ 0x7F800000 then 0x7F800000 else word)

/-- The rounded sum of the exact values `(-1)^s₁ n₁ / d` and `(-1)^s₂ n₂ / d` (units of `2^-149`). An exact zero sum is
`-0` only when both addends are zeros of negative sign (IEEE 754 §6.3); a cancellation of nonzero addends is `+0`. -/
def roundSum (s₁ : Bool) (n₁ : Nat) (s₂ : Bool) (n₂ : Nat) (d : Nat) : Nat :=
  if s₁ = s₂ then roundRn s₁ (n₁ + n₂) d
  else if n₁ > n₂ then roundRn s₁ (n₁ - n₂) d
  else if n₂ > n₁ then roundRn s₂ (n₂ - n₁) d
  else 0

/-- The exact product: a NaN when either factor is one or for `±inf * ±0`, else the signed infinity, or the product of
the magnitudes in units of `2^-298`. -/
def Decoded.mul : Decoded → Decoded → Decoded
  | .nan, _ | _, .nan => .nan
  | .inf sa, .inf sb => .inf (sa ^^ sb)
  | .inf sa, .fin sb mb => if mb = 0 then .nan else .inf (sa ^^ sb)
  | .fin sa ma, .inf sb => if ma = 0 then .nan else .inf (sa ^^ sb)
  | .fin sa ma, .fin sb mb => .fin (sa ^^ sb) (ma * mb)

/-- The rounded sum of two exact values whose finite magnitudes are in units of `2^-149 / d`: `nan` for a NaN addend or
for infinities of opposite signs, else the infinity, else `roundSum`. -/
def Decoded.addRn (nan d : Nat) : Decoded → Decoded → Nat
  | .nan, _ | _, .nan => nan
  | .inf sa, .inf sb => if sa = sb then infWord sa else nan
  | .inf sa, .fin _ _ => infWord sa
  | .fin _ _, .inf sb => infWord sb
  | .fin sa ma, .fin sb mb => roundSum sa ma sb mb d

/-- `fp32.f32_is_nan`. -/
def f32IsNan (u : Nat) : Bool := u &&& 0x7F800000 == 0x7F800000 && u &&& 0x7FFFFF != 0

/-- `fp32.f32_gpu_nan`: `u`, or `F32_NAN_PTX` when `u` is a NaN. -/
def f32GpuNan (u : Nat) : Nat := if f32IsNan u then F32_NAN_PTX else u

/-- `fp32.ftz_bits`: a subnormal word to the zero of its sign. -/
def ftzBits (u : Nat) : Nat :=
  let u := u &&& 0xFFFFFFFF
  if u &&& 0x7F800000 = 0 ∧ u &&& 0x7FFFFF ≠ 0 then u &&& 0x80000000 else u

/-! ## The `_v2` arithmetic -/

/-- `F32Add` v2, FADD: `RN(a + b)`; every NaN result `F32_NAN_PTX`. -/
def f32AddV2 (a b : Nat) : Nat := (decode a).addRn F32_NAN_PTX 1 (decode b)

/-- `F32Mul` v2, FMUL: `RN(a * b)`; every NaN result `F32_NAN_PTX`. -/
def f32MulV2 (a b : Nat) : Nat :=
  match (decode a).mul (decode b) with
  | .nan => F32_NAN_PTX
  | .inf s => infWord s
  | .fin s m => roundRn s m (1 <<< 149)

/-- `F32Div` v2, `div.rn.f32`: `RN(a / b)`; every NaN result `F32_NAN_PTX`. `±inf / ±inf` and `±0 / ±0` are NaNs, a
nonzero finite word over a zero is the infinity of the quotient's sign, and a finite word over an infinity its zero. -/
def f32DivV2 (a b : Nat) : Nat :=
  match decode a, decode b with
  | .nan, _ | _, .nan | .inf _, .inf _ => F32_NAN_PTX
  | .inf sa, .fin sb _ => infWord (sa ^^ sb)
  | .fin sa _, .inf sb => zeroWord (sa ^^ sb)
  | .fin sa ma, .fin sb mb =>
    if mb = 0 then (if ma = 0 then F32_NAN_PTX else infWord (sa ^^ sb))
    else roundRn (sa ^^ sb) (ma <<< 149) mb

/-- `fp32.f32_fma_bits`: `RN(a * b + c)`, one rounding; every NaN result `F32_NAN`. The product is exact, in units of
`2^-298`, and `c` is read in the same units. -/
def f32FmaBits (a b c : Nat) : Nat :=
  let c' : Decoded := match decode c with
    | .fin s m => .fin s (m <<< 149)
    | x => x
  ((decode a).mul (decode b)).addRn F32_NAN (1 <<< 149) c'

/-- `F32Fma` v2, FFMA: `RN(a * b + c)`, one rounding; every NaN result `F32_NAN_PTX`. -/
def f32FmaV2 (a b c : Nat) : Nat := f32GpuNan (f32FmaBits a b c)

/-- `F32AddFtz` v2, FADD.FTZ: `ftz(RN(ftz(a) + ftz(b)))`; every NaN result `F32_NAN_PTX`. -/
def f32AddFtzV2 (a b : Nat) : Nat := ftzBits (f32AddV2 (ftzBits a) (ftzBits b))

/-- `F32SubFtz` v2: `ftz(RN(ftz(a) - ftz(b)))`; every NaN result `F32_NAN_PTX`. IEEE 754's `x - y` is `x + (-y)`, so
the difference is the sum with the second operand's sign bit flipped. -/
def f32SubFtzV2 (a b : Nat) : Nat := ftzBits (f32AddV2 (ftzBits a) (ftzBits b ^^^ 0x80000000))

/-- `F32MulFtz` v2, FMUL.FTZ: `ftz(RN(ftz(a) * ftz(b)))`; every NaN result `F32_NAN_PTX`. -/
def f32MulFtzV2 (a b : Nat) : Nat := ftzBits (f32MulV2 (ftzBits a) (ftzBits b))

/-! ## The `_v1` forms -/

/-- `F32Fma` v1, FFMA: `RN(a * b + c)`, one rounding; every NaN result `F32_NAN`. -/
def f32Fma (a b c : Nat) : Nat := f32FmaBits a b c

/-- `F32FmaFtz` v1, FFMA.FTZ: `ftz(RN(ftz(a) * ftz(b) + ftz(c)))`; every NaN result `F32_NAN`. -/
def f32FmaFtz (a b c : Nat) : Nat := ftzBits (f32FmaBits (ftzBits a) (ftzBits b) (ftzBits c))

/-- `F32FmaSubFtz` v1, FFMA.FTZ with negated addend: `ftz(RN(ftz(a) * ftz(b) - ftz(c)))`; every NaN result `F32_NAN`. -/
def f32FmaSubFtz (a b c : Nat) : Nat := ftzBits (f32FmaBits (ftzBits a) (ftzBits b) (ftzBits c ^^^ 0x80000000))

/-- `F32FmaFtz` v2: `f32FmaFtz` with every NaN result `F32_NAN_PTX`. -/
def f32FmaFtzV2 (a b c : Nat) : Nat := f32GpuNan (f32FmaFtz a b c)

/-- `F32FmaSubFtz` v2: `f32FmaSubFtz` with every NaN result `F32_NAN_PTX`. -/
def f32FmaSubFtzV2 (a b c : Nat) : Nat := f32GpuNan (f32FmaSubFtz a b c)

/-- numpy `float32`'s `a > b`: false when either is a NaN, and `-0 > +0` is false. Words of one sign order as their
magnitude bits do, the infinity last. -/
def f32Gt (a b : Nat) : Bool :=
  let ma := a &&& 0x7FFFFFFF
  let mb := b &&& 0x7FFFFFFF
  if f32IsNan a || f32IsNan b then false
  else match (a >>> 31) &&& 1 == 1, (b >>> 31) &&& 1 == 1 with
    | false, false => ma > mb
    | false, true => ma ≠ 0 ∨ mb ≠ 0
    | true, false => false
    | true, true => ma < mb

/-- `fp32.f32_max_bits`: `a > b ? a : b` (FA2 `MaxOp`). -/
def f32MaxBits (a b : Nat) : Nat := if f32Gt a b then a else b

/-- `F32Max` v1: `max(x, y) = x > y ? x : y` on the flushed operands, as the fast-math build does. -/
def f32Max (a b : Nat) : Nat := f32MaxBits (ftzBits a) (ftzBits b)

/-- `GuardNegInfZero` v1, FA2 `Check_inf`: `m == -inf ? 0 : m`. -/
def guardNegInfZero (m : Nat) : Nat := if m &&& 0xFFFFFFFF = 0xFF800000 then 0 else m

/-- `fp32.bf16_to_f32_bits`: a BF16 word as the FP32 word of the same value. -/
def bf16ToF32Bits (b : Nat) : Nat := (b &&& 0xFFFF) <<< 16

/-- `fp32.f32_to_bf16_rn_bits`: round to nearest even on the low 16 bits (overflow to `±inf`), every NaN to `nan`. -/
def f32ToBf16RnBits (u nan : Nat) : Nat :=
  let u := u &&& 0xFFFFFFFF
  if u &&& 0x7F800000 = 0x7F800000 ∧ u &&& 0x7FFFFF ≠ 0 then nan
  else ((u + 0x7FFF + ((u >>> 16) &&& 1)) >>> 16) &&& 0xFFFF

/-- `Bf16Add` v1, `aten.add(bf16, bf16)`: `bf16_rn(f32(a) + f32(b))`; every NaN `0x7FC0`. -/
def bf16Add (a b : Nat) : Nat := f32ToBf16RnBits (f32AddV2 (bf16ToF32Bits a) (bf16ToF32Bits b)) BF16_NAN_TORCH

/-- `Bf16AddF2fp` v1, CUDA `_f16Vec += _f16Vec`: `bf16_rn(f32(a) + f32(b))`; every NaN `0x7FFF`. -/
def bf16AddF2fp (a b : Nat) : Nat := f32ToBf16RnBits (f32AddV2 (bf16ToF32Bits a) (bf16ToF32Bits b)) BF16_NAN_F2FP

/-- `Bf16MulF32` v1, `_f16Vec *= float`: `bf16_rn(f32(a) * s)`; every NaN `0x7FFF`. -/
def bf16MulF32 (a s : Nat) : Nat := f32ToBf16RnBits (f32MulV2 (bf16ToF32Bits a) s) BF16_NAN_F2FP

/-- `Bf16MulBf16` v1, `_f16Vec *= _f16Vec`: `bf16_rn(f32(a) * f32(b))`; every NaN `0x7FFF`. -/
def bf16MulBf16 (a b : Nat) : Nat := f32ToBf16RnBits (f32MulV2 (bf16ToF32Bits a) (bf16ToF32Bits b)) BF16_NAN_F2FP

end Verity.Fp32
