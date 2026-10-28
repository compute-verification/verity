import Mathlib.Data.Int.Basic

/-!
# `PouwZyDequant_v1`, the served dequant, on words (trusted)

`h1zy.dequant` (`verity/core/protocols/pouw/circuit/h1zy.py`): y = RN_bf16(RN32(RN32(RN32(z) · s_a) · s_w)), z an s32
word, the scales f32 words, y a bf16 word, the activation scale first and no bias add (`int8_ops.scaled_mm`'s epilogue).
Its primitives' word evaluators, by integer arithmetic on the encodings:

* `S32ToF32Rn_v1` (`silicon.cast.s32_to_f32_rn_word`): the s32 word as the nearest f32, ties to even, zero `+0`;
* `F32Mul_v2` (`ops.fp32.F32MulV2`): IEEE 754 binary32 multiplication, round to nearest even with subnormals and no flush,
  every NaN result `0x7FFFFFFF` (`F32_NAN_PTX`);
* `F2fpBf16_v1` (`silicon.cast.f32_to_bf16_hw_word`): `(u + 0x7FFF + lsb) >> 16` on a non-NaN word, every NaN `0x7FFF`.

`PouwZyDequant_v2`, the circuit, is circuits' three Boolean pieces (`S32ToF32Rn_v2`, two `F32Mul_v4`, `F2fpBf16_v2`) whose
word view this is. `Proofs.Pouw.H1Zy.DequantVectors` holds it to the Python evaluators on vectors through each case.
-/

namespace Pouw.H1Zy

/-- The number of bits of `m`, `⌊log₂ m⌋ + 1` for `m > 0` and `0` for `0`, by fuel (`m < 2^64` here). -/
def bitLen (m : ℕ) : ℕ := go 64 m
where
  go : ℕ → ℕ → ℕ
    | 0, _ => 0
    | k + 1, m => if m = 0 then 0 else go k (m / 2) + 1

/-- `m / 2^d` rounded to nearest, ties to even. -/
def rneShift (m d : ℕ) : ℕ :=
  if d = 0 then m
  else
    let q := m / 2 ^ d
    let r := m % 2 ^ d
    if r > 2 ^ (d - 1) ∨ (r = 2 ^ (d - 1) ∧ q % 2 = 1) then q + 1 else q

/-- **The f32 nearest `(-1)^s · m · 2^E`**, ties to even, with subnormals and overflow to infinity. The quantum is
`2^qe`, `qe = max(bitLen m - 24 + E, -149)`; the rounded `q = m · 2^E / 2^qe` lies in `[2^23, 2^24]` for a normal result
and below `2^23` for a subnormal one, and either way the magnitude's encoding is `(qe + 149) · 2^23 + q` (a carry to
`2^24` is the next binade's `q = 2^23`), an infinity from `0x7F800000` on. -/
def roundF32 (s : Bool) (m : ℕ) (E : ℤ) : BitVec 32 :=
  let sign : ℕ := if s then 2 ^ 31 else 0
  if m = 0 then BitVec.ofNat 32 sign
  else
    let qe : ℤ := max ((bitLen m : ℤ) - 24 + E) (-149)
    let q : ℕ := if qe ≥ E then rneShift m (qe - E).toNat else m * 2 ^ (E - qe).toNat
    let mag : ℕ := (qe + 149).toNat * 2 ^ 23 + q
    BitVec.ofNat 32 (sign + min mag 0x7F800000)

/-- **`S32ToF32Rn_v1`**: the s32 word `z` as the nearest f32 word, ties to even; zero is `+0`. -/
def s32ToF32Rn (z : BitVec 32) : BitVec 32 := roundF32 (decide (z.toInt < 0)) z.toInt.natAbs 0

/-- An f32 word's sign, biased exponent and fraction. -/
def f32Sign (a : BitVec 32) : Bool := decide (a.toNat ≥ 2 ^ 31)
def f32Exp (a : BitVec 32) : ℕ := a.toNat / 2 ^ 23 % 2 ^ 8
def f32Frac (a : BitVec 32) : ℕ := a.toNat % 2 ^ 23

def f32IsNaN (a : BitVec 32) : Bool := f32Exp a = 255 && f32Frac a ≠ 0
def f32IsInf (a : BitVec 32) : Bool := f32Exp a = 255 && f32Frac a = 0
def f32IsZero (a : BitVec 32) : Bool := f32Exp a = 0 && f32Frac a = 0

/-- A finite f32 word's significand `m`, its value `m · 2^(f32Scale a)`. -/
def f32Sig (a : BitVec 32) : ℕ := if f32Exp a = 0 then f32Frac a else f32Frac a + 2 ^ 23
def f32Scale (a : BitVec 32) : ℤ := (max (f32Exp a) 1 : ℕ) - 150

/-- `F32_NAN_PTX`, the GPU's word for every NaN result of the FP32 arithmetic. -/
def f32NanPtx : BitVec 32 := 0x7FFFFFFF

/-- **`F32Mul_v2`**: IEEE 754 binary32 `a · b`, round to nearest even, no flush; every NaN result `0x7FFFFFFF`. -/
def f32Mul (a b : BitVec 32) : BitVec 32 :=
  let s := xor (f32Sign a) (f32Sign b)
  if f32IsNaN a || f32IsNaN b then f32NanPtx
  else if (f32IsInf a && f32IsZero b) || (f32IsZero a && f32IsInf b) then f32NanPtx
  else if f32IsInf a || f32IsInf b then BitVec.ofNat 32 ((if s then 2 ^ 31 else 0) + 0x7F800000)
  else roundF32 s (f32Sig a * f32Sig b) (f32Scale a + f32Scale b)

/-- **`F2fpBf16_v1`** (`cvt.rn.bf16.f32`): round to nearest even by `(u + 0x7FFF + lsb) >> 16`, a carry past the largest
finite bf16 giving an infinity; every NaN `0x7FFF`. -/
def f2fpBf16 (u : BitVec 32) : BitVec 16 :=
  if f32IsNaN u then 0x7FFF
  else BitVec.ofNat 16 ((u.toNat + 0x7FFF + u.toNat / 2 ^ 16 % 2) / 2 ^ 16)

/-- **`PouwZyDequant_v1`**: `F2fpBf16_v1(F32Mul_v2(F32Mul_v2(S32ToF32Rn_v1(z), sa), sw))`. -/
def zyDequant (z sa sw : BitVec 32) : BitVec 16 := f2fpBf16 (f32Mul (f32Mul (s32ToF32Rn z) sa) sw)

end Pouw.H1Zy
