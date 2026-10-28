import Flock.Program
import Evaluate.Scalar
import Evaluate.Casts
import Evaluate.TC.Dots
import Evaluate.Words
import Evaluate.Libdevice
import Evaluate.Mufu
import Evaluate.Decimal
import Evaluate.Act

/-!
# The primitives the evaluator implements

Each gate's value is the Python evaluator's, transcribed on `Nat`, its operands taken unmasked as the Python takes them:

- E0, the Boolean profile (`verity/ml/circuits/PROTOCOL.md` §9, `verity.ml.circuits.boolean`): `And<n>_v1` and `Xor<n>_v1`
  for every n ≥ 2, `Not_v1`; and the input primitives `Input<w>_v1` and `RegisteredInput<w>_v1` (§5.2), whose gates take the
  values bound per run (a registered table's are read where a gate reads them, `Evaluate.Walk`).
- E1: the constant family `Const<w>[0x<hex>]_v1` (`verity.ml.circuits.ops.prims.const`, of which the constant bits are the
  1-bit members), and the scalar primitives of `Evaluate.Scalar`.
- E2: the FP32 and BF16 arithmetic of `Evaluate.Fp32` and the casts of `Evaluate.Casts`.
- E3: the tensor-core dot steps of `Evaluate.TC.Dots`, an array operand's words in leaf order.
- E4: PoUW's and NCP's word primitives (`Evaluate.Words`), the sampler's (`Evaluate.Libdevice`), the MUFU primitives
  (`Evaluate.Mufu`), the reference kinds' correctly rounded exp, erf and tanh (`Evaluate.Decimal`), the activations and
  RoPE's outputs (`Evaluate.Act`), and the lazy families `TableRead{IB}x{EW}_v1`, `GatherBf16x{V}_v1` and `GatherF32x{V}_v1`,
  which read their operands by index.

A primitive is recognized by its id and must declare the signature the registry holds for it, parameter names included
(`codec.decode_program`'s check). Every other primitive is refused by name, never guessed.
-/

namespace Evaluate

open Flock.Program

/-- A primitive the evaluator implements: an input of a width, a gate's value on its operands in operand order, or a gate's
value from the operands it reads, by index, of an operand list too wide to gather (a lazy family, `evaluate_lazy`). -/
inductive Prim where
  | input (w : Nat)
  | gate (apply : Array Nat → Nat)
  | lazy (apply : (Nat → Flock.V Nat) → Flock.V Nat)

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
  else none

/-- `(w, bits)` of the constant family's id in the one spelling `prims.const(w, bits)` gives it and the decoder takes
(`verity/ml/circuits/PROTOCOL.md` §9, `prims.CONST_ID`): `Const<w>[0x<h>]_v1` with `w ≥ 1` in decimal without a leading
zero, `bits < 2^w`, and `h` the lowercase hex of `bits` zero-padded to `w / 4` digits, at least one
(`f"{bits:0{w // 4}x}"`). -/
def constId? (id : String) : Option (Nat × Nat) := do
  guard (id.startsWith "Const" && id.endsWith "]_v1")
  let inner := ((id.drop 5).dropEnd 4).toString
  let [w, hex] := inner.splitOn "[0x" | none
  guard (w.length > 0 && w.all Char.isDigit && w.front != '0' && hex.length > 0)
  let w ← w.toNat?
  let bits ← hex.toList.foldlM (fun acc c => (acc * 16 + ·) <$> hexDigit? c) 0
  let ds := Nat.toDigits 16 bits
  guard (bits >>> w == 0 && hex.length == max (w / 4) ds.length
    && hex.toList == List.replicate (hex.length - ds.length) '0' ++ ds)
  return (w, bits)

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

private def quaternary (f : Nat → Nat → Nat → Nat → Nat) : Array Nat → Nat := fun xs => f xs[0]! xs[1]! xs[2]! xs[3]!

/-- PoUW's and NCP's word primitives by id (`verity.core.protocols.pouw.circuit.words`, the vLLM integration's
`pouw_rows`), with the signatures the registry holds for them. -/
def words : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open Words in
  let u32 := Ty.value 32
  let u8 := Ty.value 8
  let i7 := Ty.value 7
  let sq := Ty.value 576
  let a16 := Ty.array 16 u8
  let win := #[("a", .array 128 u8), ("b", .array 128 u8)]
  let sha (name : String) (f : Nat → Nat) :=
    [(s!"PouwSha512{name}Hi_v1", #[("hi", u32), ("lo", u32)], u32, binary (shaHi f)),
     (s!"PouwSha512{name}Lo_v1", #[("hi", u32), ("lo", u32)], u32, binary (shaLo f))]
  [("PouwXor32_v1", #[("a", u32), ("b", u32)], u32, binary xor32),
   ("PouwChi32_v1", #[("a", u32), ("b", u32), ("c", u32)], u32, ternary chi32),
   ("PouwRotl64Lo_v1", #[("lo", u32), ("hi", u32), ("c", u32)], u32, ternary rotl64Lo),
   ("PouwRotl64Hi_v1", #[("lo", u32), ("hi", u32), ("c", u32)], u32, ternary rotl64Hi),
   ("PouwAdd64Lo_v1", #[("alo", u32), ("blo", u32)], u32, binary add64Lo),
   ("PouwAdd64Hi_v1", #[("ahi", u32), ("alo", u32), ("bhi", u32), ("blo", u32)], u32, quaternary add64Hi)] ++
  sha "BigSigma0" bigSigma0 ++ sha "BigSigma1" bigSigma1 ++ sha "SmallSigma0" smallSigma0 ++ sha "SmallSigma1" smallSigma1 ++
  [("PouwCh32_v1", #[("e", u32), ("f", u32), ("g", u32)], u32, ternary ch32),
   ("PouwMaj32_v1", #[("a", u32), ("b", u32), ("c", u32)], u32, ternary maj32),
   ("PouwPackBE4_v1", #[("b0", i7), ("b1", i7), ("b2", i7), ("b3", i7)], u32, quaternary packBE4),
   ("PouwNoiseByte_v1", #[("w", u32), ("c", u32)], u8, binary noiseByte),
   ("PouwFormX_v1", #[("a", i7), ("e", u8)], u8, binary formX),
   ("PouwFormX3_v1", #[("a", i7), ("e1", u8), ("e2", u8)], u8, ternary formX3),
   ("PouwAdd8_v1", #[("b", i7), ("f", u8)], u8, binary add8),
   ("PouwSub8_v1", #[("b", i7), ("f", u8)], u8, binary sub8),
   ("PouwAddSub8_v1", #[("b", i7), ("f", u8), ("g", u8)], u8, ternary addSub8),
   ("PouwNegBetaSum16_v1", #[("acc", u32), ("y", a16)], u32, fun xs => negBetaSum16 xs[0]! (seg xs 1 16)),
   ("PouwDot16U8S8_v1", #[("acc", u32), ("x", a16), ("y", a16)], u32, fun xs => dot16U8S8 xs[0]! (seg xs 1 16) (seg xs 17 16)),
   ("PouwI32ToF32Rn_v1", #[("z", u32)], f32, unary i32ToF32Rn),
   ("PouwF32SqrtRn_v1", #[("a", f32)], f32, unary f32SqrtRn),
   ("PouwETilde_v1", #[("e", e2m1), ("ae", u8)], bf16, binary eTilde),
   ("PouwBf16Neg_v1", #[("a", bf16)], bf16, unary bf16NegWord),
   ("PouwSqAcc_v1", #[("acc", sq), ("x", f32)], sq, binary sqAccWord),
   ("PouwSqCeil_v1", #[("t", sq), ("n", u32)], .value 64, binary sqCeilWord),
   ("PouwSqGe_v1", #[("x", f32), ("th", .value 64)], bit, binary sqGeWord),
   ("PouwBlackwellE4m3Window_v1", win, .value 480, fun xs => window TC.blackwellE4m3Dot32 (seg xs 0 128) (seg xs 128 128)),
   ("PouwHopperE4m3Window_v1", win, .value 480, fun xs => window TC.hopperE4m3Dot32 (seg xs 0 128) (seg xs 128 128)),
   ("PouwWindowWord_v1", #[("x", .value 480), ("m", .value 4)], f32, binary windowWord),
   ("PouwBlackwellE4m3SaltDead_v1", #[("acc", f32), ("cb", u8)], bit, binary (endsAgree TC.blackwellE4m3Dot32)),
   ("PouwHopperE4m3SaltDead_v1", #[("acc", f32), ("cb", u8)], bit, binary (endsAgree TC.hopperE4m3Dot32)),
   ("NcpAbsMax_v1", #[("m", f32), ("x", f32)], f32, binary ncpAbsMax),
   ("NcpRneClampI7_v1", #[("q", f32)], i7, unary ncpRneClampI7),
   ("NcpDot16I7_v1", #[("acc", u32), ("a", .array 16 i7), ("b", .array 16 i7)], u32,
    fun xs => ncpDot16I7 xs[0]! (seg xs 1 16) (seg xs 17 16))]

/-- The sampler's primitives by id (the vLLM integration's `sampling` and `topp_word_gates`), with the signatures the
registry holds for them. -/
def sampler : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open Libdevice in
  let lane := #[("key", f32), ("v", f32)]
  [("NvLogf_v1", #[("a", f32)], f32, unary nvLogf), ("NvLogf_v2", #[("a", f32)], f32, unary nvLogfGpu),
   ("GumbelStreamKey_v1", #[("seed", .value 64), ("pos", .value 64)], f32, binary streamKey),
   ("GumbelNoiseLane_v1", lane, f32, binary noiseLane), ("GumbelNoiseLane_v2", lane, f32, binary noiseLaneGpu),
   ("TopPNumkeep_v1", #[("numdup", f32), ("q", f32)], f32, binary toppNumkeep)]

/-- The MUFU primitives by id (`verity.ml.circuits.ops.mufu`, the vLLM integration's `hopper.Fa3InvSum`), with the
signatures the registry holds for them. -/
def mufu : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open Mufu in
  [("MufuEx2Ftz_v1", #[("x", f32)], f32, unary mufuEx2Ftz), ("MufuRcpFtz_v1", #[("x", f32)], f32, unary mufuRcpFtz),
   ("Fa2InvSum_v1", #[("sum", f32)], f32, unary fa2InvSum), ("Fa3InvSum_v1", #[("sum", f32)], f32, unary fa3InvSum),
   ("MufuSqrtFtz_v1", #[("x", f32)], f32, unary mufuSqrtFtz), ("DivFullRcp_v1", #[("b", f32)], f32, unary divFullRcp),
   ("RsqrtApprox_v1", #[("x", f32)], f32, unary rsqrtApprox), ("MufuTanh_v1", #[("x", f32)], f32, unary mufuTanh)]

/-- The correctly rounded transcendentals of the reference kinds by id (the vLLM integration's `ref_prims`), with the
signatures the registry holds for them. -/
def decimals : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open Decimal in
  [("F32ExpRn_v1", #[("x", f32)], f32, unary f32ExpRn), ("F32ErfRn_v1", #[("x", f32)], f32, unary f32ErfRn),
   ("F32TanhRn_v1", #[("x", f32)], f32, unary f32TanhRn)]

/-- The activations and RoPE's outputs by id (the vLLM integration's `prims`, `act` and `dense`), with the signatures the
registry holds for them. -/
def activations : List (String × Array (String × Ty) × Ty × (Array Nat → Nat)) :=
  open Act in
  let gu := #[("g", bf16), ("u", bf16)]
  let xycs := #[("x", bf16), ("y", bf16), ("c", bf16), ("s", bf16)]
  [("SiluMulBf16_v1", gu, bf16, binary siluMulBf16), ("SiluMulBf16_v2", gu, bf16, binary siluMulBf16V2),
   ("RopeOut_v1", xycs, bf16, quaternary ropeOut), ("RopeOutAdd_v1", xycs, bf16, quaternary ropeOutAdd),
   ("TanhF32Rn_v1", #[("x", f32)], f32, unary tanhF32), ("GeluTanhMulBf16_v1", gu, bf16, binary geluTanhMulBf16),
   ("GeluErfBf16_v1", #[("x", bf16)], bf16, unary geluErfBf16)]

/-- A number in canonical decimal, as `str(n)` writes it: the decoder takes a family member only by the id its maker gives it. -/
private def digits? (s : String) : Option Nat := s.toNat?.filter (toString · == s)

/-- The lazy families by id, `^TableRead(\d+)x(\d+)_v1$` (`tables.table_read`, which refuses an index width and entry size
that do not fill rows of 64 words), `^GatherBf16x(\d+)_v1$` (`prims.gather_bf16`) and `^GatherF32x(\d+)_v1$`
(`ref_prims.gather_f32`), each number in canonical decimal, with the signatures their makers give them. -/
def lazyFamily? (id : String) : Option (Prim × Array (String × Ty) × Ty) := do
  guard (id.endsWith "_v1")
  let s := (id.dropEnd 3).toString
  if s.startsWith "TableRead" then
    let [a, b] := (s.drop 9).toString.splitOn "x" | none
    let ib ← digits? a
    let ew ← digits? b
    guard (0 < ib && ib ≤ 32 && ew ≥ 1 && 64 % ew == 0 && (ew <<< ib) % 64 == 0)
    return (.lazy (Act.tableRead ib ew), #[("index", .value (if ib ≤ 16 then 16 else 32)),
      ("table", .array ((ew <<< ib) / 64) (.array 64 bf16))], .value (16 * ew))
  else if s.startsWith "GatherBf16x" then
    let v ← digits? (s.drop 11).toString
    return (.lazy (Act.gather v 0x7FFF), #[("token", f32), ("column", .array v bf16)], bf16)
  else if s.startsWith "GatherF32x" then
    let v ← digits? (s.drop 10).toString
    return (.lazy (Act.gather v Fp32.F32_NAN), #[("token", f32), ("column", .array v f32)], f32)
  else none

/-- The primitive an id denotes, with the signature the registry holds for it, or none. -/
def byId? (id : String) : Option (Prim × Array (String × Ty) × Ty) :=
  match arity? "And" id, arity? "Xor" id, constId? id with
  | some n, _, _ => some (.gate andBits, #[("x", .array n bit)], bit)
  | _, some n, _ => some (.gate xorBits, #[("x", .array n bit)], bit)
  | none, none, some (w, b) => some (.gate fun _ => b, #[], .value w)
  | none, none, none =>
    if id == "Not_v1" then some (.gate (unary notBit), #[("a", bit)], bit)
    else match ((scalars ++ floats ++ dots ++ words ++ sampler ++ mufu ++ decimals ++ activations).lookup id) with
      | some (params, ret, f) => some (.gate f, params, ret)
      | none => lazyFamily? id

private def sameParams (a b : Array (String × Ty)) : Bool :=
  a.size == b.size && (a.zip b).all fun (x, y) => x.1 == y.1 && x.2 == y.2

/-- The implementation of a primitive Definition of the program, or its refusal by name. -/
def ofDefn (d : Defn) : Flock.V Prim := do
  match inputId? d.key with
  | some (_, w) => return .input w
  | none =>
    let some (p, params, ret) := byId? d.key
      | throw s!"primitive {d.key}: not implemented by this evaluator (neither a primitive of the registry nor a member of a family it implements)"
    if !(sameParams d.params params && d.ret == ret) then throw s!"primitive {d.key} signature differs from the registry"
    return p

end Evaluate
