import Flock.Program
import Evaluate.Scalar
import Evaluate.Casts
import Evaluate.TC.Dots

/-!
# The primitives the evaluator implements

Each gate's value is the Python evaluator's, transcribed on `Nat`, its operands taken unmasked as the Python takes them:

- E0, the Boolean profile (`verity/ml/circuits/PROTOCOL.md` §9, `verity.ml.circuits.boolean`): `And<n>_v1` and `Xor<n>_v1`
  for every n ≥ 2, `Not_v1`; and the input primitives `Input<w>_v1` (§5.2), whose gates take the values bound per run.
- E1: the constant family `Const<w>[0x<hex>]_v1` (`verity.ml.circuits.ops.prims.const`, of which the constant bits are the
  1-bit members), and the scalar primitives of `Evaluate.Scalar`.
- E2: the FP32 and BF16 arithmetic of `Evaluate.Fp32` and the casts of `Evaluate.Casts`.
- E3: the tensor-core dot steps of `Evaluate.TC.Dots`, an array operand's words in leaf order.

A primitive is recognized by its id and must declare the signature the registry holds for it, parameter names included
(`codec.decode_program`'s check). Every other primitive is refused by name, never guessed: `RegisteredInput<w>_v1`
(registered inputs and their tables are a later step) and every primitive not listed here.
-/

namespace Evaluate

open Flock.Program

/-- A primitive the evaluator implements: an input of a width, or a gate's value on its operands in operand order. -/
inductive Prim where
  | input (w : Nat)
  | gate (apply : Array Nat → Nat)

/-- The gate's value on its operands; an input gate has none (its value is bound, not computed). -/
def Prim.apply : Prim → Array Nat → Nat
  | .gate f, xs => f xs
  | .input _, _ => 0

/-- `boolean._and`: `int(all(b & 1 for b in x))`. Must equal `Verity.Boolean.Op.apply` of `and` on bits, folded. -/
def andBits (xs : Array Nat) : Nat := if xs.all (fun b => b &&& 1 == 1) then 1 else 0

/-- `boolean._xor`: the XOR of the operands, masked to its low bit. Must equal `Verity.Boolean.xorSum` on bits. -/
def xorBits (xs : Array Nat) : Nat := xs.foldl (· ^^^ ·) 0 &&& 1

/-- `boolean.NOT`: `(a ^ 1) & 1`. Must equal `Verity.Boolean.Op.apply` of `not` on bits. -/
def notBit (a : Nat) : Nat := (a ^^^ 1) &&& 1

/-- `n` of `boolean._N_ARY_ID`, `^(And|Xor)([2-9]|[1-9][0-9]+)_v1$`: decimal without a leading zero, at least 2. -/
private def arity? (pre id : String) : Option Nat :=
  if !(id.startsWith pre && id.endsWith "_v1") then none else
  let n := ((id.drop pre.length).dropEnd 3).toString
  if n.length > 0 && n.all Char.isDigit && n.front != '0' then n.toNat?.filter (· ≥ 2) else none

private def hexDigit? (c : Char) : Option Nat :=
  if c.isDigit then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else if 'A' ≤ c ∧ c ≤ 'F' then some (c.toNat - 'A'.toNat + 10)
  else none

/-- `(w, bits)` of the constant family's id, `^Const(\d+)\[0x([0-9a-fA-F]+)\]_v1$` (`prims._register_lazy_families`):
`int(m[1])` and `int(m[2], 16)`, leading zeros and either case allowed. -/
def constId? (id : String) : Option (Nat × Nat) := do
  guard (id.startsWith "Const" && id.endsWith "]_v1")
  let inner := ((id.drop 5).dropEnd 4).toString
  let [w, hex] := inner.splitOn "[0x" | none
  guard (w.length > 0 && w.all Char.isDigit && hex.length > 0)
  let bits ← hex.toList.foldlM (fun acc c => (acc * 16 + ·) <$> hexDigit? c) 0
  return (← w.toNat?, bits)

private def bit := Ty.value 1
private def f32 := Ty.value 32
private def bf16 := Ty.value 16

private def unary (f : Nat → Nat) : Array Nat → Nat := fun xs => f xs[0]!
private def binary (f : Nat → Nat → Nat) : Array Nat → Nat := fun xs => f xs[0]! xs[1]!
private def ternary (f : Nat → Nat → Nat → Nat) : Array Nat → Nat := fun xs => f xs[0]! xs[1]! xs[2]!

/-- The scalar primitives by id, with the signatures the registry holds for them (`verity.ml.circuits.ops.scalar`, and the
vLLM integration's `dsa.U32And` / `I32Sub`, `topp_word_gates.BitXor` / `U32Xor` / `SelectBit`, `ref_prims.I64ToI32`). -/
def scalars : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open Scalar in
  [("F32Fabs_v1", #[("x", f32)], f32, unary f32Fabs),
   ("F32Neg_v1", #[("a", f32)], f32, unary f32Neg),
   ("F32Fmaxf_v1", #[("a", f32), ("b", f32)], f32, binary f32Fmaxf),
   ("F32Fminf_v1", #[("a", f32), ("b", f32)], f32, binary f32Fminf),
   ("F32Sat_v1", #[("a", f32)], f32, unary f32Sat),
   ("F32BitsShl23_v1", #[("a", f32)], f32, unary f32BitsShl23),
   ("F32IsFinite_v1", #[("a", f32)], bit, unary f32IsFinite),
   ("F32Eq_v1", #[("a", f32), ("b", f32)], bit, binary f32Eq),
   ("Bf16GtStrict_v1", #[("a", bf16), ("b", bf16)], bit, binary bf16GtStrict),
   ("I32Le_v1", #[("a", f32), ("b", f32)], bit, binary i32Le),
   ("I32Eq_v1", #[("a", f32), ("b", f32)], bit, binary i32Eq),
   ("I32Add_v1", #[("a", f32), ("b", f32)], f32, binary i32Add),
   ("I32Sub_v1", #[("a", f32), ("b", f32)], f32, binary i32Sub),
   ("I64ToI32_v1", #[("t", .value 64)], f32, unary i64ToI32),
   ("U32And_v1", #[("a", f32), ("b", f32)], f32, binary u32And),
   ("U32Xor_v1", #[("a", f32), ("b", f32)], f32, binary u32Xor),
   ("BitAnd_v1", #[("a", bit), ("b", bit)], bit, binary bitAnd),
   ("BitOr_v1", #[("a", bit), ("b", bit)], bit, binary bitOr),
   ("BitXor_v1", #[("a", bit), ("b", bit)], bit, binary bitXor),
   ("BitNot_v1", #[("a", bit)], bit, unary bitNot),
   ("SelectBit_v1", #[("c", bit), ("a", bit), ("b", bit)], bit, ternary selectBit),
   ("SelectI32_v1", #[("c", bit), ("a", f32), ("b", f32)], f32, ternary selectI32),
   ("SelectF32_v1", #[("c", bit), ("a", f32), ("b", f32)], f32, ternary selectF32),
   ("SelectBf16_v1", #[("c", bit), ("a", bf16), ("b", bf16)], bf16, ternary selectBf16)]

private def e4m3 := Ty.value 8
private def e2m1 := Ty.value 4

/-- The FP32 and BF16 arithmetic and the casts by id, with the signatures the registry holds for them
(`verity.ml.circuits.ops.fp32`, `prims` and `mufu`, and the vLLM integration's `prims`, `ref_prims`, `moe`, `dsa` and
`topp_word_gates`). -/
def floats : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open Fp32 Casts in
  let ab := #[("a", f32), ("b", f32)]
  let abc := #[("a", f32), ("b", f32), ("c", f32)]
  let bf := #[("a", bf16), ("b", bf16)]
  [("F32Add_v1", ab, f32, binary f32Add), ("F32Add_v2", ab, f32, binary f32AddV2),
   ("F32Sub_v1", ab, f32, binary f32Sub), ("F32Sub_v2", ab, f32, binary f32SubV2),
   ("F32Mul_v1", ab, f32, binary f32Mul), ("F32Mul_v2", ab, f32, binary f32MulV2),
   ("F32Div_v1", ab, f32, binary f32Div), ("F32Div_v2", ab, f32, binary f32DivV2),
   ("F32Fma_v1", abc, f32, ternary f32Fma), ("F32Fma_v2", abc, f32, ternary f32FmaV2),
   ("F32AddFtz_v1", ab, f32, binary f32AddFtz), ("F32AddFtz_v2", ab, f32, binary f32AddFtzV2),
   ("F32SubFtz_v1", ab, f32, binary f32SubFtz), ("F32SubFtz_v2", ab, f32, binary f32SubFtzV2),
   ("F32MulFtz_v1", ab, f32, binary f32MulFtz), ("F32MulFtz_v2", ab, f32, binary f32MulFtzV2),
   ("F32FmaFtz_v1", abc, f32, ternary f32FmaFtz), ("F32FmaFtz_v2", abc, f32, ternary f32FmaFtzV2),
   ("F32FmaSubFtz_v1", abc, f32, ternary f32FmaSubFtz), ("F32FmaSubFtz_v2", abc, f32, ternary f32FmaSubFtzV2),
   ("F32FmaRm_v1", abc, f32, ternary f32FmaRm), ("F32FmaRm_v2", abc, f32, ternary f32FmaRmV2),
   ("F32Sqrt_v1", #[("x", f32)], f32, unary f32Sqrt),
   ("F32CeilFtz_v1", #[("x", f32)], f32, unary f32CeilFtz),
   ("I32ToF32_v1", #[("a", f32)], f32, unary i32ToF32),
   ("F32Max_v1", ab, f32, binary f32Max),
   ("F32MaxRef_v1", ab, f32, binary f32MaxRef),
   ("F32GtStrict_v1", ab, bit, binary f32GtStrict),
   ("GuardNegInfZero_v1", #[("m", f32)], f32, unary guardNegInfZero),
   ("DivFullScaleA_v1", ab, f32, binary divFullScaleA), ("DivFullScaleA_v2", ab, f32, binary divFullScaleAV2),
   ("Bf16Add_v1", bf, bf16, binary bf16Add), ("Bf16AddF2fp_v1", bf, bf16, binary bf16AddF2fp),
   ("Bf16MulF32_v1", #[("a", bf16), ("s", f32)], bf16, binary bf16MulF32),
   ("Bf16MulBf16_v1", bf, bf16, binary bf16MulBf16),
   ("Bf16ToF32_v1", #[("a", bf16)], f32, unary bf16ToF32),
   ("F32ToBf16Rn_v1", #[("a", f32)], bf16, unary f32ToBf16Rn),
   ("F2fpBf16_v1", #[("a", f32)], bf16, unary f2fpBf16),
   ("E4m3ToF32_v1", #[("x", e4m3)], f32, unary e4m3ToF32),
   ("F32ToE4m3Sat_v1", #[("x", f32)], e4m3, unary f32ToE4m3Sat),
   ("F32ToE2m1Sat_v1", #[("x", f32)], e2m1, unary f32ToE2m1Sat)]

/-- `n` operand words from index `i`: one array operand's leaves. -/
private def seg (xs : Array Nat) (i n : Nat) : List Nat := (xs.extract i (i + n)).toList

/-- The tensor-core dot steps by id (`verity.ml.circuits.ops.prims`), with the signatures the registry holds for them. -/
def dots : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open TC in
  let bf := #[("acc", f32), ("a", .array 16 bf16), ("b", .array 16 bf16)]
  let fp8 := #[("acc", f32), ("a", .array 32 e4m3), ("b", .array 32 e4m3)]
  let fp4 (n : Nat) := #[("acc", f32), ("a", .array 64 e2m1), ("b", .array 64 e2m1), ("sa", .array n e4m3),
    ("sb", .array n e4m3)]
  [("AmpereBF16TcDot16_v2", bf, f32, fun xs => ampereBf16Dot16 xs[0]! (seg xs 1 16) (seg xs 17 16)),
   ("HopperBF16WgmmaDot16_v1", bf, f32, fun xs => hopperBf16Dot16 xs[0]! (seg xs 1 16) (seg xs 17 16)),
   ("HopperE4m3QgmmaDot32_v1", fp8, f32, fun xs => hopperE4m3Dot32 xs[0]! (seg xs 1 32) (seg xs 33 32)),
   ("BlackwellE4m3QmmaDot32_v1", fp8, f32, fun xs => blackwellE4m3Dot32 xs[0]! (seg xs 1 32) (seg xs 33 32)),
   ("BlackwellE5m2QmmaDot32_v1", fp8, f32, fun xs => blackwellE5m2Dot32 xs[0]! (seg xs 1 32) (seg xs 33 32)),
   ("BlackwellNvf4OmmaDot64_v1", fp4 4, f32,
    fun xs => blackwellNvf4Dot64 xs[0]! (seg xs 1 64) (seg xs 65 64) (seg xs 129 4) (seg xs 133 4)),
   ("BlackwellMxf4OmmaDot64_v1", fp4 2, f32,
    fun xs => blackwellMxf4Dot64 xs[0]! (seg xs 1 64) (seg xs 65 64) (seg xs 129 2) (seg xs 131 2))]

/-- The primitive an id denotes, with the signature the registry holds for it, or none. -/
def byId? (id : String) : Option (Prim × Array (String × Ty) × Ty) :=
  match arity? "And" id, arity? "Xor" id, constId? id with
  | some n, _, _ => some (.gate andBits, #[("x", .array n bit)], bit)
  | _, some n, _ => some (.gate xorBits, #[("x", .array n bit)], bit)
  | none, none, some (w, b) => some (.gate fun _ => b, #[], .value w)
  | none, none, none =>
    if id == "Not_v1" then some (.gate (unary notBit), #[("a", bit)], bit)
    else ((scalars ++ floats ++ dots).lookup id).map fun (params, ret, f) => (.gate f, params, ret)

private def sameParams (a b : Array (String × Ty)) : Bool :=
  a.size == b.size && (a.zip b).all fun (x, y) => x.1 == y.1 && x.2 == y.2

/-- The implementation of a primitive Definition of the program, or its refusal by name. -/
def ofDefn (d : Defn) : Flock.V Prim := do
  match inputId? d.key with
  | some (b, w) =>
    if b == REGISTRATION then
      throw s!"primitive {d.key}: a registered input, which this evaluator refuses (it binds run inputs only)"
    return .input w
  | none =>
    let some (p, params, ret) := byId? d.key
      | throw s!"primitive {d.key}: not implemented by this evaluator (E0 to E3: the Boolean profile, the constants, the scalar primitives, the FP32 and BF16 arithmetic and casts that read no table, and the tensor-core dot steps)"
    if !(sameParams d.params params && d.ret == ret) then throw s!"primitive {d.key} signature differs from the registry"
    return p

end Evaluate
