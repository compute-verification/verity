/-!
# The scalar primitives: integer, bit, select and comparison operations, and the FP32 operations on the encoding

Executable transcriptions, function for function and branch for branch, of `verity.ml.circuits.ops.scalar` and of the word
primitives the vLLM integration registers (`verity_vllm.program.registry`: `dsa`, `topp_word_gates`, `ref_prims`). Words are
natural numbers: an f32 or an i32 (two's complement, bit 31 the sign) below `2^32`, an i64 below `2^64`, a bf16 below `2^16`, a
bit `0` or `1`. Each function takes its operands unmasked, as the Python does, and is the Python's function on every
non-negative integer; `Int` holds the Python's signed integers, and Python's `&` of a negative integer and `2^k - 1` is
`Int.emod` by `2^k`.

Each definition's doc names the definition of `Security/Definitions/Core/Scalar.lean` it must equal, to be proved in
`Security/Proofs` as the certifier's copies are; where Security defines none yet, the Python function is the definition.
-/

namespace Evaluate.Scalar

/-- `0xFFFFFFFF`, the Python's `M32`. -/
def M32 : Nat := 0xFFFFFFFF

/-- The Python's `v & (2^k - 1)` on a signed `v`: `v mod 2^k`. -/
def lowBits (v : Int) (k : Nat) : Nat := (v % ((2 : Int) ^ k)).toNat

/-- `cast.f32_is_nan`. Must equal `Verity.Scalar.f32IsNan`. -/
def f32IsNan (u : Nat) : Bool := (u &&& 0x7F800000 == 0x7F800000) && (u &&& 0x7FFFFF != 0)

/-- `cast.bf16_is_nan`. Security: none yet. -/
def bf16IsNan (w : Nat) : Bool := (w &&& 0x7F80 == 0x7F80) && (w &&& 0x7F != 0)

/-- `scalar._order(u, width)`. Must equal `Verity.Scalar.order`. -/
def order (u width : Nat) : Int :=
  let sign := 1 <<< (width - 1)
  let u := u &&& ((sign <<< 1) - 1)
  let m := u &&& (sign - 1)
  if u &&& sign ≠ 0 then -(m : Int) else m

/-- `scalar.F32_NAN_PTX`. Must equal `Verity.Scalar.F32_NAN_PTX`. -/
def F32_NAN_PTX : Nat := 0x7FFFFFFF

/-- `F32Fabs_v1`: `x & 0x7FFFFFFF`. Must equal `Verity.Scalar.f32Fabs`. -/
def f32Fabs (x : Nat) : Nat := x &&& 0x7FFFFFFF

/-- `F32Neg_v1`: `(a ^ 0x80000000) & 0xFFFFFFFF`. Must equal `Verity.Scalar.f32Neg`. -/
def f32Neg (a : Nat) : Nat := (a ^^^ 0x80000000) &&& 0xFFFFFFFF

/-- `scalar._fmaxmin(a, b, want_max)`. Security: `Verity.Scalar.f32Fmaxf` is its `want_max = true` case. -/
def fmaxmin (a b : Nat) (wantMax : Bool) : Nat :=
  let na := f32IsNan a
  let nb := f32IsNan b
  if na && nb then F32_NAN_PTX
  else if na then b
  else if nb then a
  else
    let ka := order a 32
    let kb := order b 32
    if ka = kb then
      if a &&& 0x7FFFFFFF = 0 ∧ b &&& 0x7FFFFFFF = 0 then
        let pos := if a = 0 ∨ b = 0 then 0 else 0x80000000
        let neg := if a = 0x80000000 ∨ b = 0x80000000 then 0x80000000 else 0
        if wantMax then pos else neg
      else a
    else if wantMax then (if ka > kb then a else b)
    else if ka < kb then a else b

/-- `F32Fmaxf_v1`: `_fmaxmin(a, b, True)`. Must equal `Verity.Scalar.f32Fmaxf`. -/
def f32Fmaxf (a b : Nat) : Nat := fmaxmin a b true

/-- `F32Fminf_v1`: `_fmaxmin(a, b, False)`. Security: none yet. -/
def f32Fminf (a b : Nat) : Nat := fmaxmin a b false

/-- `F32Sat_v1` (`cvt.sat.f32.f32`): NaN and the negative encodings to `+0`, from `1.0` up `1.0`. Security: none yet. -/
def f32Sat (a : Nat) : Nat :=
  let u := a &&& 0xFFFFFFFF
  if f32IsNan u || u >>> 31 != 0 then 0
  else if u ≥ 0x3F800000 then 0x3F800000 else u

/-- `F32BitsShl23_v1`: `((a & 0xFFFFFFFF) << 23) & 0xFFFFFFFF`. Security: none yet. -/
def f32BitsShl23 (a : Nat) : Nat := ((a &&& 0xFFFFFFFF) <<< 23) &&& 0xFFFFFFFF

/-- `F32IsFinite_v1`: `0` when the exponent field is all ones. Security: none yet. -/
def f32IsFinite (a : Nat) : Nat := if a &&& 0x7F800000 = 0x7F800000 then 0 else 1

/-- `F32Eq_v1`: IEEE `a == b` (`+0 == -0`, NaN unequal to everything). Security: none yet. -/
def f32Eq (a b : Nat) : Nat :=
  if f32IsNan (a &&& 0xFFFFFFFF) || f32IsNan (b &&& 0xFFFFFFFF) then 0
  else if order a 32 = order b 32 then 1 else 0

/-- `Bf16GtStrict_v1`: `a > b` on bf16 values (NaN compares false, `+0 == -0`). Security: none yet. -/
def bf16GtStrict (a b : Nat) : Nat :=
  if bf16IsNan (a &&& 0xFFFF) || bf16IsNan (b &&& 0xFFFF) then 0
  else if order a 16 > order b 16 then 1 else 0

/-- `I32Le_v1`: signed 32-bit `a <= b`. Must equal `Verity.Scalar.i32Le`. -/
def i32Le (a b : Nat) : Nat :=
  let sa : Int := if a &&& 0x80000000 ≠ 0 then (a : Int) - 2 ^ 32 else a
  let sb : Int := if b &&& 0x80000000 ≠ 0 then (b : Int) - 2 ^ 32 else b
  if sa ≤ sb then 1 else 0

/-- `I32Eq_v1`: `(a & 0xFFFFFFFF) == (b & 0xFFFFFFFF)`. Security: none yet. -/
def i32Eq (a b : Nat) : Nat := if a &&& 0xFFFFFFFF = b &&& 0xFFFFFFFF then 1 else 0

/-- `I32Add_v1`: `(a + b) & 0xFFFFFFFF`. Must equal `Verity.Scalar.i32Add`. -/
def i32Add (a b : Nat) : Nat := (a + b) &&& 0xFFFFFFFF

/-- `I32Sub_v1` (`dsa.I32Sub`, `sub.s32`): `(a - b) & M32`. Security: none yet. -/
def i32Sub (a b : Nat) : Nat := lowBits ((a : Int) - b) 32

/-- `I64ToI32_v1` (`ref_prims.I64ToI32`): the two's-complement value `v` of the low 64 bits, as a 32-bit word when
`-2^31 <= v < 2^31`, else `0x80000000`. Security: none yet. -/
def i64ToI32 (t : Nat) : Nat :=
  let t := t &&& (2 ^ 64 - 1)
  let v : Int := if t >>> 63 ≠ 0 then (t : Int) - 2 ^ 64 else t
  if -(2 : Int) ^ 31 ≤ v ∧ v < (2 : Int) ^ 31 then lowBits v 32 else 0x80000000

/-- `U32And_v1` (`dsa.U32And`, `and.b32`): `a & b & M32`. Security: none yet. -/
def u32And (a b : Nat) : Nat := a &&& b &&& M32

/-- `U32Xor_v1` (`topp_word_gates._uxor`): `(a ^ b) & M32`. Security: none yet. -/
def u32Xor (a b : Nat) : Nat := (a ^^^ b) &&& M32

/-- `BitAnd_v1`: `a & b & 1`. Must equal `Verity.Scalar.bitAnd`. -/
def bitAnd (a b : Nat) : Nat := a &&& b &&& 1

/-- `BitOr_v1`: `(a | b) & 1`. Security: none yet. -/
def bitOr (a b : Nat) : Nat := (a ||| b) &&& 1

/-- `BitXor_v1` (`topp_word_gates._xor`): `(a ^ b) & 1`. Security: none yet. -/
def bitXor (a b : Nat) : Nat := (a ^^^ b) &&& 1

/-- `BitNot_v1`: `(a ^ 1) & 1`. Security: none yet. -/
def bitNot (a : Nat) : Nat := (a ^^^ 1) &&& 1

/-- `SelectBit_v1` (`topp_word_gates._selb`): `(a if c & 1 else b) & 1`. Security: none yet. -/
def selectBit (c a b : Nat) : Nat := (if c &&& 1 ≠ 0 then a else b) &&& 1

/-- `SelectI32_v1`: `a if c & 1 else b`. Must equal `Verity.Scalar.selectI32`. -/
def selectI32 (c a b : Nat) : Nat := if c &&& 1 ≠ 0 then a else b

/-- `SelectF32_v1`: `a if c & 1 else b`. Security: none yet. -/
def selectF32 (c a b : Nat) : Nat := if c &&& 1 ≠ 0 then a else b

/-- `SelectBf16_v1`: `a if c & 1 else b`. Must equal `Verity.Scalar.selectBf16`. -/
def selectBf16 (c a b : Nat) : Nat := if c &&& 1 ≠ 0 then a else b

end Evaluate.Scalar
