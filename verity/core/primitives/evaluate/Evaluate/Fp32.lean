import Evaluate.Scalar

/-!
# The FP32 and BF16 arithmetic primitives

Executable transcriptions of `verity.ml.circuits.ops.fp32` and of the FP32 word primitives the vLLM integration registers
(`verity_vllm.program.registry`: `prims`, `ref_prims`, `moe`, `dsa`, `topp_word_gates`), and of `mufu.DivFullScaleA_v1`,
the one MUFU primitive that reads no table. Words are natural numbers as in `Evaluate.Scalar`.

Where `Security/Definitions/Core/Fp32.lean` defines a primitive, the definition here is a copy of it, the certifier's
pattern; each doc names the definition it must equal, to be proved in `Security/Proofs`. Every finite word is a multiple of
the least subnormal `2^-149` (`decode`), so an exact sum, product, quotient or fused `a * b + c` is a rational `n / d` in
those units, which `roundRn` (`fp32._round_f32`) or `roundRm` (`moe._round_f32_toward_neg_inf`) takes to a word.

The NaN of a result is a parameter of each operation (`addWith`, `mulWith`, `divWith`, `fmaWith`): the `_v2` forms pass the
GPU's `F32_NAN_PTX`, the FMAs of `_v1` `F32_NAN`, and the `_v1` add, sub, mul and div the host's word (`hostNan`), which
Security does not state: the Python computes them in numpy `float32` on x86-64, whose SSE arithmetic returns a NaN operand
quieted, or `0xFFC00000` for an invalid operation.
-/

namespace Evaluate.Fp32

open Evaluate.Scalar (M32 f32IsNan F32_NAN_PTX)

/-- `fp32.F32_NAN`: the `_v1` FMAs' NaN. Must equal `Verity.Fp32.F32_NAN`. -/
def F32_NAN : Nat := 0x7FC00000

/-- `fp32.BF16_NAN_TORCH`. Must equal `Verity.Fp32.BF16_NAN_TORCH`. -/
def BF16_NAN_TORCH : Nat := 0x7FC0

/-- `fp32.BF16_NAN_F2FP`. Must equal `Verity.Fp32.BF16_NAN_F2FP`. -/
def BF16_NAN_F2FP : Nat := 0x7FFF

/-- The zero of a sign. Must equal `Verity.Fp32.zeroWord`. -/
def zeroWord (negative : Bool) : Nat := if negative then 0x80000000 else 0

/-- The infinity of a sign. Must equal `Verity.Fp32.infWord`. -/
def infWord (negative : Bool) : Nat := zeroWord negative ||| 0x7F800000

/-- `fp32._decode`: a NaN, a signed infinity, or a finite word's sign and magnitude in units of `2^-149`. Must equal
`Verity.Fp32.Decoded`. -/
inductive Decoded where
  | nan
  | inf (negative : Bool)
  | fin (negative : Bool) (magnitude : Nat)

/-- `fp32._decode` on the word's low 32 bits, as the Python masks it. Must equal `Verity.Fp32.decode` on words. -/
def decode (u : Nat) : Decoded :=
  let u := u &&& M32
  let e := (u >>> 23) &&& 0xFF
  let m := u &&& 0x7FFFFF
  let negative := (u >>> 31) &&& 1 == 1
  if e = 0xFF then (if m = 0 then .inf negative else .nan)
  else if e = 0 then .fin negative m
  else .fin negative ((m ||| 0x800000) <<< (e - 1))

/-- `fp32._round_f32` on `(-1)^negative * n / d` units of `2^-149`, `d > 0`: the nearest word, ties to even. The words of
the binade of `n / d` are `2^j` units apart, `j = max(0, ⌊log₂(n / d)⌋ - 23)`; a rounded significand of `2^24` carries into
the next binade, and from `0x7F800000` up the word is `±inf`. Must equal `Verity.Fp32.roundRn`. -/
def roundRn (negative : Bool) (n d : Nat) : Nat :=
  let j := (n / d).log2 - 23
  let unit := d <<< j
  let q := n / unit
  let r := n % unit
  let q := if 2 * r > unit ∨ (2 * r = unit ∧ q % 2 = 1) then q + 1 else q
  let word := (j <<< 23) + q
  zeroWord negative ||| (if word ≥ 0x7F800000 then 0x7F800000 else word)

/-- `moe._round_f32_toward_neg_inf` on `(-1)^negative * n / d` units of `2^-149`, `d > 0`, `n > 0`: the largest word at
most the value. The significand is truncated, and raised by one when the value is negative and inexact; past the range a
negative value is `-inf` and a positive one the largest finite word `0x7F7FFFFF`. Security: none yet. -/
def roundRm (negative : Bool) (n d : Nat) : Nat :=
  let j := (n / d).log2 - 23
  let unit := d <<< j
  let q := n / unit
  let q := if negative ∧ n % unit ≠ 0 then q + 1 else q
  let word := (j <<< 23) + q
  if word ≥ 0x7F800000 then (if negative then 0xFF800000 else 0x7F7FFFFF) else zeroWord negative ||| word

/-- The rounded sum of `(-1)^s₁ n₁ / d` and `(-1)^s₂ n₂ / d` units; an exact zero sum is `-0` only when both addends are
zeros of negative sign. Must equal `Verity.Fp32.roundSum`. -/
def roundSum (s₁ : Bool) (n₁ : Nat) (s₂ : Bool) (n₂ : Nat) (d : Nat) : Nat :=
  if s₁ = s₂ then roundRn s₁ (n₁ + n₂) d
  else if n₁ > n₂ then roundRn s₁ (n₁ - n₂) d
  else if n₂ > n₁ then roundRn s₂ (n₂ - n₁) d
  else 0

/-- The exact product, in units of `2^-298` when finite. Must equal `Verity.Fp32.Decoded.mul`. -/
def Decoded.mul : Decoded → Decoded → Decoded
  | .nan, _ | _, .nan => .nan
  | .inf sa, .inf sb => .inf (sa ^^ sb)
  | .inf sa, .fin sb mb => if mb = 0 then .nan else .inf (sa ^^ sb)
  | .fin sa ma, .inf sb => if ma = 0 then .nan else .inf (sa ^^ sb)
  | .fin sa ma, .fin sb mb => .fin (sa ^^ sb) (ma * mb)

/-- The rounded sum of two exact values whose finite magnitudes are in units of `2^-149 / d`, `nan` for a NaN.
Must equal `Verity.Fp32.Decoded.addRn`. -/
def Decoded.addRn (nan d : Nat) : Decoded → Decoded → Nat
  | .nan, _ | _, .nan => nan
  | .inf sa, .inf sb => if sa = sb then infWord sa else nan
  | .inf sa, .fin _ _ => infWord sa
  | .fin _ _, .inf sb => infWord sb
  | .fin sa ma, .fin sb mb => roundSum sa ma sb mb d

/-- `fp32.f32_gpu_nan`. Must equal `Verity.Fp32.f32GpuNan`. -/
def f32GpuNan (u : Nat) : Nat := if f32IsNan u then F32_NAN_PTX else u

/-- `fp32.ftz_bits`. Must equal `Verity.Fp32.ftzBits`. -/
def ftzBits (u : Nat) : Nat :=
  let u := u &&& 0xFFFFFFFF
  if u &&& 0x7F800000 = 0 ∧ u &&& 0x7FFFFF ≠ 0 then u &&& 0x80000000 else u

/-! ## IEEE add, mul and div, with the NaN as a parameter -/

/-- `RN(a + b)`, every NaN result `nan`. `addWith F32_NAN_PTX` must equal `Verity.Fp32.f32AddV2`. -/
def addWith (nan a b : Nat) : Nat := (decode a).addRn nan 1 (decode b)

/-- `RN(a * b)`, every NaN result `nan`. `mulWith F32_NAN_PTX` must equal `Verity.Fp32.f32MulV2`. -/
def mulWith (nan a b : Nat) : Nat :=
  match (decode a).mul (decode b) with
  | .nan => nan
  | .inf s => infWord s
  | .fin s m => roundRn s m (1 <<< 149)

/-- `RN(a / b)`, every NaN result `nan`. `divWith F32_NAN_PTX` must equal `Verity.Fp32.f32DivV2`. -/
def divWith (nan a b : Nat) : Nat :=
  match decode a, decode b with
  | .nan, _ | _, .nan | .inf _, .inf _ => nan
  | .inf sa, .fin sb _ => infWord (sa ^^ sb)
  | .fin sa _, .inf sb => zeroWord (sa ^^ sb)
  | .fin sa ma, .fin sb mb =>
    if mb = 0 then (if ma = 0 then nan else infWord (sa ^^ sb))
    else roundRn (sa ^^ sb) (ma <<< 149) mb

/-- `RN(a * b + c)`, one rounding, every NaN result `nan`. `fmaWith F32_NAN` must equal `Verity.Fp32.f32FmaBits`. -/
def fmaWith (nan a b c : Nat) : Nat :=
  let c' : Decoded := match decode c with
    | .fin s m => .fin s (m <<< 149)
    | x => x
  ((decode a).mul (decode b)).addRn nan (1 <<< 149) c'

/-- The NaN numpy `float32` returns on x86-64 for an operation whose operands, as SSE reads them, are `x` then `y`: `x`
quieted (bit 22 set) when it is a NaN, else `y` quieted when it is, else `0xFFC00000`, the invalid operation's default
NaN. numpy evaluates `a - b` and `a / b` as `x = a, y = b`, and `a + b` and `a * b` with the operands swapped, `x = b`.
Security: none (the host's word). -/
def hostNan (x y : Nat) : Nat :=
  let x := x &&& M32
  let y := y &&& M32
  if f32IsNan x then x ||| 0x400000 else if f32IsNan y then y ||| 0x400000 else 0xFFC00000

/-- `fp32.f32_add_bits`, numpy `float32` `a + b`. Security: none (the host's NaN); off NaNs `Verity.Fp32.f32AddV2`. -/
def f32AddBits (a b : Nat) : Nat := addWith (hostNan b a) a b

/-- `fp32.f32_sub_bits`, numpy `float32` `a - b`, which IEEE 754 defines as `a + (-b)`. Security: none (the host's NaN). -/
def f32SubBits (a b : Nat) : Nat := addWith (hostNan a b) a (b ^^^ 0x80000000)

/-- `fp32.f32_mul_bits`, numpy `float32` `a * b`. Security: none (the host's NaN); off NaNs `Verity.Fp32.f32MulV2`. -/
def f32MulBits (a b : Nat) : Nat := mulWith (hostNan b a) a b

/-- `fp32.f32_div_bits`, numpy `float32` `a / b`. Security: none (the host's NaN); off NaNs `Verity.Fp32.f32DivV2`. -/
def f32DivBits (a b : Nat) : Nat := divWith (hostNan a b) a b

/-- `fp32.f32_fma_bits`. Must equal `Verity.Fp32.f32FmaBits`. -/
def f32FmaBits (a b c : Nat) : Nat := fmaWith F32_NAN a b c

/-! ## The `_v1` arithmetic -/

/-- `F32Add_v1`, FADD: `RN(a + b)`, the host's NaN. Security: none (the host's NaN). -/
def f32Add (a b : Nat) : Nat := f32AddBits a b

/-- `F32Mul_v1`, FMUL: `RN(a * b)`, the host's NaN. Security: none (the host's NaN). -/
def f32Mul (a b : Nat) : Nat := f32MulBits a b

/-- `F32Div_v1`, `div.rn.f32`: `RN(a / b)`, the host's NaN. Security: none (the host's NaN). -/
def f32Div (a b : Nat) : Nat := f32DivBits a b

/-- `F32Fma_v1`, FFMA: `RN(a * b + c)`, one rounding; every NaN `F32_NAN`. Must equal `Verity.Fp32.f32Fma`. -/
def f32Fma (a b c : Nat) : Nat := f32FmaBits a b c

/-- `F32AddFtz_v1`: `ftz(RN(ftz(a) + ftz(b)))`, the host's NaN. Security: none (the host's NaN). -/
def f32AddFtz (a b : Nat) : Nat := ftzBits (f32AddBits (ftzBits a) (ftzBits b))

/-- `F32SubFtz_v1`: `ftz(RN(ftz(a) - ftz(b)))`, the host's NaN. Security: none (the host's NaN). -/
def f32SubFtz (a b : Nat) : Nat := ftzBits (f32SubBits (ftzBits a) (ftzBits b))

/-- `F32MulFtz_v1`: `ftz(RN(ftz(a) * ftz(b)))`, the host's NaN. Security: none (the host's NaN). -/
def f32MulFtz (a b : Nat) : Nat := ftzBits (f32MulBits (ftzBits a) (ftzBits b))

/-- `F32FmaFtz_v1`: `ftz(RN(ftz(a) * ftz(b) + ftz(c)))`; every NaN `F32_NAN`. Must equal `Verity.Fp32.f32FmaFtz`. -/
def f32FmaFtz (a b c : Nat) : Nat := ftzBits (f32FmaBits (ftzBits a) (ftzBits b) (ftzBits c))

/-- `F32FmaSubFtz_v1`: `ftz(RN(ftz(a) * ftz(b) - ftz(c)))`; every NaN `F32_NAN`. Must equal `Verity.Fp32.f32FmaSubFtz`. -/
def f32FmaSubFtz (a b c : Nat) : Nat := ftzBits (f32FmaBits (ftzBits a) (ftzBits b) (ftzBits c ^^^ 0x80000000))

/-- `ref_prims._canon_nan`: every NaN to `F32_NAN`. Security: none yet. -/
def canonNan (u : Nat) : Nat := let u := u &&& M32; if f32IsNan u then F32_NAN else u

/-- `F32Sub_v1` (`ref_prims`): `RN(a - b)`; every NaN `F32_NAN`. Security: none yet. -/
def f32Sub (a b : Nat) : Nat := canonNan (f32SubBits a b)

/-! ## The `_v2` arithmetic: every NaN result `F32_NAN_PTX` -/

/-- `F32Add_v2`. Must equal `Verity.Fp32.f32AddV2`. -/
def f32AddV2 (a b : Nat) : Nat := addWith F32_NAN_PTX a b

/-- `F32Mul_v2`. Must equal `Verity.Fp32.f32MulV2`. -/
def f32MulV2 (a b : Nat) : Nat := mulWith F32_NAN_PTX a b

/-- `F32Div_v2`. Must equal `Verity.Fp32.f32DivV2`. -/
def f32DivV2 (a b : Nat) : Nat := divWith F32_NAN_PTX a b

/-- `F32Fma_v2`. Must equal `Verity.Fp32.f32FmaV2`. -/
def f32FmaV2 (a b c : Nat) : Nat := f32GpuNan (f32FmaBits a b c)

/-- `F32Sub_v2` (the vLLM integration's `prims`): `RN(a - b)` as `a + (-b)`. Security: none yet. -/
def f32SubV2 (a b : Nat) : Nat := addWith F32_NAN_PTX a (b ^^^ 0x80000000)

/-- `F32AddFtz_v2`. Must equal `Verity.Fp32.f32AddFtzV2`. -/
def f32AddFtzV2 (a b : Nat) : Nat := ftzBits (f32AddV2 (ftzBits a) (ftzBits b))

/-- `F32SubFtz_v2`. Must equal `Verity.Fp32.f32SubFtzV2`. -/
def f32SubFtzV2 (a b : Nat) : Nat := ftzBits (f32AddV2 (ftzBits a) (ftzBits b ^^^ 0x80000000))

/-- `F32MulFtz_v2`. Must equal `Verity.Fp32.f32MulFtzV2`. -/
def f32MulFtzV2 (a b : Nat) : Nat := ftzBits (f32MulV2 (ftzBits a) (ftzBits b))

/-- `F32FmaFtz_v2`. Must equal `Verity.Fp32.f32FmaFtzV2`. -/
def f32FmaFtzV2 (a b c : Nat) : Nat := f32GpuNan (f32FmaFtz a b c)

/-- `F32FmaSubFtz_v2`. Must equal `Verity.Fp32.f32FmaSubFtzV2`. -/
def f32FmaSubFtzV2 (a b c : Nat) : Nat := f32GpuNan (f32FmaSubFtz a b c)

/-! ## FMA rounding toward `-inf`, square root, ceiling, integer conversion and `div.full`'s numerator scaling -/

/-- `F32FmaRm_v1` (`moe`), `fma.rm.f32`: `a * b + c` with one rounding toward `-inf`, the non-finite operands as
`F32Fma_v1` (every NaN `F32_NAN`); an exact zero sum is `+0` unless the product is a zero of negative sign and `c` is
`-0`. The product is exact in units of `2^-298`, and `c` is read in the same units. Security: none yet. -/
def f32FmaRm (a b c : Nat) : Nat :=
  match decode a, decode b, decode c with
  | .nan, _, _ | _, .nan, _ | _, _, .nan => F32_NAN
  | .fin sa ma, .fin sb mb, .fin sc mc =>
    let sp := sa ^^ sb
    let p : Int := if sp then -((ma * mb : Nat) : Int) else ((ma * mb : Nat) : Int)
    let q : Int := if sc then -((mc <<< 149 : Nat) : Int) else ((mc <<< 149 : Nat) : Int)
    let s := p + q
    if s = 0 then (if ma * mb = 0 ∧ mc = 0 ∧ sp ∧ sc then 0x80000000 else 0)
    else roundRm (s < 0) s.natAbs (1 <<< 149)
  | .fin _ _, .fin _ _, .inf sc => infWord sc
  | da, db, dc =>
    -- a product involving an infinity: `Decoded.mul` is a NaN for `inf * 0`, else the infinity of the product's sign
    match da.mul db, dc with
    | .inf s, .inf sc => if sc = s then infWord s else F32_NAN
    | .inf s, _ => infWord s
    | _, _ => F32_NAN

/-- `F32FmaRm_v2`: `F32FmaRm_v1` with every NaN result `F32_NAN_PTX`. Security: none yet. -/
def f32FmaRmV2 (a b c : Nat) : Nat := f32GpuNan (f32FmaRm a b c)

/-- `⌊√n⌋`, by Newton's iteration from `2^(⌊log₂ n⌋ / 2 + 1) > √n`, which decreases to `⌊√n⌋` and stops there; the
error at least halves each step, so `log₂ n + 4` steps suffice. -/
def isqrt (n : Nat) : Nat :=
  let rec go : Nat → Nat → Nat
    | 0, x => x
    | fuel + 1, x => let y := (x + n / x) / 2; if y < x then go fuel y else x
  if n = 0 then 0 else go (n.log2 + 4) (1 <<< (n.log2 / 2 + 1))

/-- numpy `float32` `sqrt`: IEEE `sqrt.rn.f32` with `F32_NAN` for every NaN (the caller canonicalizes). For a positive
finite word of `m` units the root is `√(m · 2^149)` units; when that is not an integer it lies strictly between
`s = ⌊√(m · 2^149)⌋` and `s + 1`, both inside one rounding interval (the result's spacing is at least `2^51` units), so it
rounds as `s + 1/2` does. -/
def sqrtRn (x : Nat) : Nat :=
  match decode x with
  | .nan | .inf true => F32_NAN
  | .inf false => 0x7F800000
  | .fin true m => if m = 0 then 0x80000000 else F32_NAN
  | .fin false m =>
    if m = 0 then 0
    else
      let n := m <<< 149
      let s := isqrt n
      roundRn false (2 * s + (if s * s < n then 1 else 0)) 2

/-- `F32Sqrt_v1` (`ref_prims`), `sqrt.rn.f32`: `-0` kept, every NaN `F32_NAN`. Security: none yet. -/
def f32Sqrt (x : Nat) : Nat :=
  let x := x &&& M32
  if x = 0x80000000 then x else canonNan (sqrtRn x)

/-- `F32CeilFtz_v1` (`dsa.f32_ceil_ftz`), `cvt.rpi.ftz.f32.f32`: a subnormal flushed to the zero of its sign, then rounded
toward `+inf` to an integer by the fraction mask of the `150 - e` low bits; a NaN gives `0x7FFFFFFF`. Security: none yet. -/
def f32CeilFtz (x : Nat) : Nat :=
  let x := x &&& M32
  let e := (x >>> 23) &&& 0xFF
  if e = 0xFF then (if x &&& 0x7FFFFF ≠ 0 then 0x7FFFFFFF else x)
  else if e = 0 then x &&& 0x80000000
  else if e ≥ 150 then x
  else
    let neg := x >>> 31 ≠ 0
    if e < 127 then (if neg then 0x80000000 else 0x3F800000)
    else
      let mask := (1 <<< (150 - e)) - 1
      if x &&& mask = 0 then x
      else if neg then x &&& (M32 ^^^ mask) else ((x ||| mask) + 1) &&& M32

/-- `I32ToF32_v1` (`topp_word_gates._i2f`), `cvt.rn.f32.s32`: the i32 as the nearest f32, ties to even. Security: none
yet. -/
def i32ToF32 (a : Nat) : Nat :=
  let a := a &&& M32
  if a &&& 0x80000000 ≠ 0 then roundRn true (((1 <<< 32) - a) <<< 149) 1 else roundRn false (a <<< 149) 1

/-- `F32Max_v1`'s numpy `float32` `a > b`: false when either is a NaN, and `-0 > +0` false. Must equal
`Verity.Fp32.f32Gt`. -/
def f32Gt (a b : Nat) : Bool :=
  let ma := a &&& 0x7FFFFFFF
  let mb := b &&& 0x7FFFFFFF
  if f32IsNan a || f32IsNan b then false
  else match (a >>> 31) &&& 1 == 1, (b >>> 31) &&& 1 == 1 with
    | false, false => ma > mb
    | false, true => ma ≠ 0 ∨ mb ≠ 0
    | true, false => false
    | true, true => ma < mb

/-- `fp32.f32_max_bits`. Must equal `Verity.Fp32.f32MaxBits`. -/
def f32MaxBits (a b : Nat) : Nat := if f32Gt a b then a else b

/-- `F32Max_v1`: `x > y ? x : y` on the flushed operands. Must equal `Verity.Fp32.f32Max`. -/
def f32Max (a b : Nat) : Nat := f32MaxBits (ftzBits a) (ftzBits b)

/-- `F32MaxRef_v1` (`ref_prims`): `F32_NAN` when either is a NaN, else `a` when `a >= b` (numpy, which for non-NaN words
is `¬ b > a`), else `b`. Security: none yet. -/
def f32MaxRef (a b : Nat) : Nat :=
  if f32IsNan a || f32IsNan b then F32_NAN
  else if !f32Gt (b &&& M32) (a &&& M32) then a &&& M32 else b &&& M32

/-- `F32GtStrict_v1` (`ref_prims`): numpy `float32` `a > b` as a bit. Security: none yet. -/
def f32GtStrict (a b : Nat) : Nat := if f32Gt (a &&& M32) (b &&& M32) then 1 else 0

/-- `GuardNegInfZero_v1`, FA2 `Check_inf`. Must equal `Verity.Fp32.guardNegInfZero`. -/
def guardNegInfZero (m : Nat) : Nat := if m &&& 0xFFFFFFFF = 0xFF800000 then 0 else m

/-- `DivFullScaleA_v1` (`mufu`), `div.full.f32`'s numerator scaling: numpy `a * 0.25` when `|b| > 2^126`, `a * 2^24` when
`|b| < 2^-126`, else `a`; numpy's comparisons are false for a NaN `b`. Security: none (the host's NaN). -/
def divFullScaleA (a b : Nat) : Nat :=
  let ab := b &&& 0x7FFFFFFF
  if f32Gt ab 0x7E800000 then f32MulBits a 0x3E800000
  else if f32Gt 0x00800000 ab then f32MulBits a 0x4B800000
  else a &&& M32

/-- `DivFullScaleA_v2` (the vLLM integration's `prims`): each scaling an FMUL whose every NaN is `F32_NAN_PTX`; an
unscaled `a` passes untouched. Security: none yet. -/
def divFullScaleAV2 (a b : Nat) : Nat :=
  let ab := b &&& 0x7FFFFFFF
  if f32Gt ab 0x7E800000 then f32GpuNan (f32MulBits a 0x3E800000)
  else if f32Gt 0x00800000 ab then f32GpuNan (f32MulBits a 0x4B800000)
  else a &&& M32

/-! ## BF16 element-wise: computed in FP32, rounded once to BF16 -/

/-- `fp32.bf16_to_f32_bits`. Must equal `Verity.Fp32.bf16ToF32Bits`. -/
def bf16ToF32Bits (b : Nat) : Nat := (b &&& 0xFFFF) <<< 16

/-- `fp32.f32_to_bf16_rn_bits` (`cast.f32_to_bf16_rn_word`). Must equal `Verity.Fp32.f32ToBf16RnBits`. -/
def f32ToBf16RnBits (u nan : Nat) : Nat :=
  let u := u &&& 0xFFFFFFFF
  if u &&& 0x7F800000 = 0x7F800000 ∧ u &&& 0x7FFFFF ≠ 0 then nan
  else ((u + 0x7FFF + ((u >>> 16) &&& 1)) >>> 16) &&& 0xFFFF

/-- `Bf16Add_v1`: `bf16_rn(f32(a) + f32(b))`; every NaN `0x7FC0`. Must equal `Verity.Fp32.bf16Add`. -/
def bf16Add (a b : Nat) : Nat := f32ToBf16RnBits (f32AddV2 (bf16ToF32Bits a) (bf16ToF32Bits b)) BF16_NAN_TORCH

/-- `Bf16AddF2fp_v1`: `bf16_rn(f32(a) + f32(b))`; every NaN `0x7FFF`. Must equal `Verity.Fp32.bf16AddF2fp`. -/
def bf16AddF2fp (a b : Nat) : Nat := f32ToBf16RnBits (f32AddV2 (bf16ToF32Bits a) (bf16ToF32Bits b)) BF16_NAN_F2FP

/-- `Bf16MulF32_v1`: `bf16_rn(f32(a) * s)`; every NaN `0x7FFF`. Must equal `Verity.Fp32.bf16MulF32`. -/
def bf16MulF32 (a s : Nat) : Nat := f32ToBf16RnBits (f32MulV2 (bf16ToF32Bits a) s) BF16_NAN_F2FP

/-- `Bf16MulBf16_v1`: `bf16_rn(f32(a) * f32(b))`; every NaN `0x7FFF`. Must equal `Verity.Fp32.bf16MulBf16`. -/
def bf16MulBf16 (a b : Nat) : Nat := f32ToBf16RnBits (f32MulV2 (bf16ToF32Bits a) (bf16ToF32Bits b)) BF16_NAN_F2FP

end Evaluate.Fp32
