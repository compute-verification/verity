import Definitions.Accounting.Gates.Model

/-!
# The registry's primitives, each in one class (trusted)

Every primitive of the registry (`verity/ml/circuits/ops/prims.py`, 194 at `615eabc13a`, with the s8 family's two steps
`BlackwellS8ImmaDot32_v1` and its u8·s8 sibling `BlackwellU8S8ImmaDot32_v1`) in exactly one class of the model
(`Model.lean`), as the 19:15Z ruling closes the free set (note:20261017T1859Z-report-property-redteam):

* **free:** the s8 family's steps (`family`), whose window property/v1 proves from derived ranges; `I32Add_v1` and
  `I32Sub_v1` (`affine`), proven not to wrap; the constants (`const`). A product with a constant operand is an affine
  gate of the census, whatever its primitive.
* **budgeted**, one unit an output value: `product` (`I32Mul_v1` of two runtime values, and the integer dots outside the
  family: `NcpDot16I7_v1`, `PouwDot16U8S8_v1`, `PouwSqAcc_v1`); `compareSelect` (comparisons, selections, min, max and
  the salt-dead tests); `gather` (`PouwWindowWord_v1`, and the kernel-level gathers and table reads, by family); and
  `rounding`, every other primitive: FP arithmetic and casts (`Bf16ToF32_v1`, `E4m3ToF32_v1` and `F32Neg_v1` among
  them, which census/v1's 18:21Z draft had free), the FP and FP8 dots, integer division, remainder and narrowing, bit
  operations, and PoUW's integer helpers.
* **refused:** any id neither list names (`classOf id = none`).

`classOf` is what property/v1's closed table is to agree with, id for id.
-/

namespace Accounting.Gates

/-- A primitive's class. -/
inductive Class
  | input (b : Binding)
  | const
  | affine
  | family
  | budgeted (cls : Budget)
  deriving DecidableEq, Repr

/-- The registry's primitives, each with its class. -/
def registry : List (String × Class) := [
  ("AmpereBF16TcDot16_v2", .budgeted .rounding),
  ("And2_v1", .budgeted .rounding),
  ("Bf16AddF2fp_v1", .budgeted .rounding),
  ("Bf16Add_v1", .budgeted .rounding),
  ("Bf16GtStrict_v1", .budgeted .compareSelect),
  ("Bf16MulBf16_v1", .budgeted .rounding),
  ("Bf16MulF32_v1", .budgeted .rounding),
  ("Bf16ToF32_v1", .budgeted .rounding),
  ("BitAnd_v1", .budgeted .rounding),
  ("BitNot_v1", .budgeted .rounding),
  ("BitOr_v1", .budgeted .rounding),
  ("BitXor_v1", .budgeted .rounding),
  ("BlackwellE4m3QmmaDot32_v1", .budgeted .rounding),
  ("BlackwellE5m2QmmaDot32_v1", .budgeted .rounding),
  ("BlackwellMxf4OmmaDot64_v1", .budgeted .rounding),
  ("BlackwellNvf4OmmaDot64_v1", .budgeted .rounding),
  ("BlackwellS8ImmaDot32_v1", .family),
  ("BlackwellU8S8ImmaDot32_v1", .family),
  ("Const16[0x0000]_v1", .const),
  ("Const16[0x8000]_v1", .const),
  ("Const1[0x0]_v1", .const),
  ("Const1[0x1]_v1", .const),
  ("Const32[0x00000000]_v1", .const),
  ("Const32[0x00000001]_v1", .const),
  ("Const32[0x1e3ce508]_v1", .const),
  ("Const32[0x2edbe6ff]_v1", .const),
  ("Const32[0x32a57060]_v1", .const),
  ("Const32[0x36924925]_v1", .const),
  ("Const32[0x3bbb989d]_v1", .const),
  ("Const32[0x3c80f082]_v1", .const),
  ("Const32[0x3c820821]_v1", .const),
  ("Const32[0x3e085941]_v1", .const),
  ("Const32[0x3f000000]_v1", .const),
  ("Const32[0x3f199999]_v1", .const),
  ("Const32[0x3f317218]_v1", .const),
  ("Const32[0x3f800000]_v1", .const),
  ("Const32[0x3fb8aa3b]_v1", .const),
  ("Const32[0x40000000]_v1", .const),
  ("Const32[0x4038aa3b]_v1", .const),
  ("Const32[0x40c00000]_v1", .const),
  ("Const32[0x41102cb3]_v1", .const),
  ("Const32[0x41a00000]_v1", .const),
  ("Const32[0x437c0000]_v1", .const),
  ("Const32[0x43e00000]_v1", .const),
  ("Const32[0x4b400001]_v1", .const),
  ("Const32[0x7fffffff]_v1", .const),
  ("Const32[0x80000000]_v1", .const),
  ("Const32[0xbd563cae]_v1", .const),
  ("Const32[0xbeaaa9ed]_v1", .const),
  ("Const32[0xbfb8aa3b]_v1", .const),
  ("Const32[0xc0000000]_v1", .const),
  ("Const32[0xc2fc0000]_v1", .const),
  ("Const32[0xc3e00000]_v1", .const),
  ("Const32[0xc61c4000]_v1", .const),
  ("Const32[0xcb40007f]_v1", .const),
  ("Const32[0xf149f2ca]_v1", .const),
  ("Const32[0xff800000]_v1", .const),
  ("Const32[0xffffffff]_v1", .const),
  ("DivFullRcp_v1", .budgeted .rounding),
  ("DivFullScaleA_v1", .budgeted .rounding),
  ("DivFullScaleA_v2", .budgeted .rounding),
  ("E2m1ToBf16_v1", .budgeted .rounding),
  ("E4m3ToF32_v1", .budgeted .rounding),
  ("E8m0DequantBf16_v1", .budgeted .rounding),
  ("E8m0ShlBf16_v1", .budgeted .rounding),
  ("E8m0ToBf16_v1", .budgeted .rounding),
  ("F2fpBf16_v1", .budgeted .rounding),
  ("F32AddFtz_v1", .budgeted .rounding),
  ("F32AddFtz_v2", .budgeted .rounding),
  ("F32Add_v1", .budgeted .rounding),
  ("F32Add_v2", .budgeted .rounding),
  ("F32BitsShl23_v1", .budgeted .rounding),
  ("F32CeilFtz_v1", .budgeted .rounding),
  ("F32Div_v1", .budgeted .rounding),
  ("F32Div_v2", .budgeted .rounding),
  ("F32Eq_v1", .budgeted .compareSelect),
  ("F32ErfRn_v1", .budgeted .rounding),
  ("F32ExpRn_v1", .budgeted .rounding),
  ("F32Fabs_v1", .budgeted .rounding),
  ("F32FmaFtz_v1", .budgeted .rounding),
  ("F32FmaFtz_v2", .budgeted .rounding),
  ("F32FmaRm_v1", .budgeted .rounding),
  ("F32FmaRm_v2", .budgeted .rounding),
  ("F32FmaSubFtz_v1", .budgeted .rounding),
  ("F32FmaSubFtz_v2", .budgeted .rounding),
  ("F32Fma_v1", .budgeted .rounding),
  ("F32Fma_v2", .budgeted .rounding),
  ("F32Fmaxf_v1", .budgeted .compareSelect),
  ("F32Fminf_v1", .budgeted .compareSelect),
  ("F32Geu_v1", .budgeted .compareSelect),
  ("F32GtStrict_v1", .budgeted .compareSelect),
  ("F32IsFinite_v1", .budgeted .compareSelect),
  ("F32MaxRef_v1", .budgeted .compareSelect),
  ("F32Max_v1", .budgeted .compareSelect),
  ("F32MulFtz_v1", .budgeted .rounding),
  ("F32MulFtz_v2", .budgeted .rounding),
  ("F32Mul_v1", .budgeted .rounding),
  ("F32Mul_v2", .budgeted .rounding),
  ("F32Neg_v1", .budgeted .rounding),
  ("F32Sat_v1", .budgeted .rounding),
  ("F32Sqrt_v1", .budgeted .rounding),
  ("F32SubFtz_v1", .budgeted .rounding),
  ("F32SubFtz_v2", .budgeted .rounding),
  ("F32Sub_v1", .budgeted .rounding),
  ("F32Sub_v2", .budgeted .rounding),
  ("F32TanhRn_v1", .budgeted .rounding),
  ("F32ToBf16Rn_v1", .budgeted .rounding),
  ("F32ToE2m1Sat_v1", .budgeted .rounding),
  ("F32ToE4m3Sat_v1", .budgeted .rounding),
  ("F32ToU32RzByte_v1", .budgeted .rounding),
  ("Fa2InvSum_v1", .budgeted .rounding),
  ("Fa3InvSum_v1", .budgeted .rounding),
  ("GeluErfBf16_v1", .budgeted .rounding),
  ("GeluTanhMulBf16_v1", .budgeted .rounding),
  ("GuardNegInfZero_v1", .budgeted .compareSelect),
  ("GumbelNoiseLane_v1", .budgeted .rounding),
  ("GumbelNoiseLane_v2", .budgeted .rounding),
  ("GumbelStreamKey_v1", .budgeted .rounding),
  ("HopperBF16WgmmaDot16_v1", .budgeted .rounding),
  ("HopperE4m3MmaV2Dot32_v1", .budgeted .rounding),
  ("HopperE4m3QgmmaDot32_v1", .budgeted .rounding),
  ("HopperTf32WgmmaDot8_v1", .budgeted .rounding),
  ("I32Add_v1", .affine),
  ("I32Div_v1", .budgeted .rounding),
  ("I32Eq_v1", .budgeted .compareSelect),
  ("I32Le_v1", .budgeted .compareSelect),
  ("I32Mul_v1", .budgeted .product),
  ("I32Rem_v1", .budgeted .rounding),
  ("I32Sub_v1", .affine),
  ("I32ToF32_v1", .budgeted .rounding),
  ("I64ToI32_v1", .budgeted .rounding),
  ("MufuEx2Ftz_v1", .budgeted .rounding),
  ("MufuLg2Mantissa_v1", .budgeted .rounding),
  ("MufuRcpFtz_v1", .budgeted .rounding),
  ("MufuSqrtFtz_v1", .budgeted .rounding),
  ("MufuTanh_v1", .budgeted .rounding),
  ("NcpAbsMax_v1", .budgeted .compareSelect),
  ("NcpDot16I7_v1", .budgeted .product),
  ("NcpRneClampI7_v1", .budgeted .rounding),
  ("Not_v1", .budgeted .rounding),
  ("NvLogf_v1", .budgeted .rounding),
  ("NvLogf_v2", .budgeted .rounding),
  ("PouwAdd64Hi_v1", .budgeted .rounding),
  ("PouwAdd64Lo_v1", .budgeted .rounding),
  ("PouwAdd8_v1", .budgeted .rounding),
  ("PouwAddSub8_v1", .budgeted .rounding),
  ("PouwBf16Neg_v1", .budgeted .rounding),
  ("PouwBlackwellE4m3SaltDead_v1", .budgeted .compareSelect),
  ("PouwBlackwellE4m3Window_v1", .budgeted .rounding),
  ("PouwCh32_v1", .budgeted .rounding),
  ("PouwChi32_v1", .budgeted .rounding),
  ("PouwDot16U8S8_v1", .budgeted .product),
  ("PouwETilde_v1", .budgeted .rounding),
  ("PouwF32SqrtRn_v1", .budgeted .rounding),
  ("PouwFormX3_v1", .budgeted .rounding),
  ("PouwFormX_v1", .budgeted .rounding),
  ("PouwHopperE4m3SaltDead_v1", .budgeted .compareSelect),
  ("PouwHopperE4m3Window_v1", .budgeted .rounding),
  ("PouwI32ToF32Rn_v1", .budgeted .rounding),
  ("PouwMaj32_v1", .budgeted .rounding),
  ("PouwNegBetaSum16_v1", .budgeted .rounding),
  ("PouwNoiseByte_v1", .budgeted .rounding),
  ("PouwPackBE4_v1", .budgeted .rounding),
  ("PouwRotl64Hi_v1", .budgeted .rounding),
  ("PouwRotl64Lo_v1", .budgeted .rounding),
  ("PouwSha512BigSigma0Hi_v1", .budgeted .rounding),
  ("PouwSha512BigSigma0Lo_v1", .budgeted .rounding),
  ("PouwSha512BigSigma1Hi_v1", .budgeted .rounding),
  ("PouwSha512BigSigma1Lo_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma0Hi_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma0Lo_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma1Hi_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma1Lo_v1", .budgeted .rounding),
  ("PouwSqAcc_v1", .budgeted .product),
  ("PouwSqCeil_v1", .budgeted .rounding),
  ("PouwSqGe_v1", .budgeted .compareSelect),
  ("PouwSub8_v1", .budgeted .rounding),
  ("PouwWindowWord_v1", .budgeted .gather),
  ("PouwXor32_v1", .budgeted .rounding),
  ("RopeOutAdd_v1", .budgeted .rounding),
  ("RopeOut_v1", .budgeted .rounding),
  ("RsqrtApprox_v1", .budgeted .rounding),
  ("SelectBf16_v1", .budgeted .compareSelect),
  ("SelectBit_v1", .budgeted .compareSelect),
  ("SelectF32_v1", .budgeted .compareSelect),
  ("SelectI32_v1", .budgeted .compareSelect),
  ("SiluMulBf16_v1", .budgeted .rounding),
  ("SiluMulBf16_v2", .budgeted .rounding),
  ("TanhF32Rn_v1", .budgeted .rounding),
  ("TopPNumkeep_v1", .budgeted .rounding),
  ("U32And_v1", .budgeted .rounding),
  ("U32Ge_v1", .budgeted .compareSelect),
  ("U32Xor_v1", .budgeted .rounding),
  ("Ue8m0CeilByte_v1", .budgeted .rounding),
  ("Ue8m0InvScale_v1", .budgeted .rounding),
  ("Xor2_v1", .budgeted .rounding)]

/-- The kernel-level families, by their id's prefix. -/
def families : List (String × Class) := [
  ("Input", .input .run),
  ("RegisteredInput", .input .registered),
  ("GatherBf16x", .budgeted .gather),
  ("GatherF32x", .budgeted .gather),
  ("TableRead", .budgeted .gather),
  ("BitAtx", .budgeted .gather),
  ("TopPMaskWordx", .budgeted .compareSelect)]

/-- A primitive id's class: the registry's entry, else its family's, else none (refused). -/
def classOf (id : String) : Option Class :=
  (registry.lookup id).orElse fun _ => (families.find? fun e => id.startsWith e.1).map Prod.snd

/-- The class of a gate's operation. -/
def Op.cls {ι : Type} : Op ι → Class
  | .input b _ => .input b
  | .const _ => .const
  | .affine .. => .affine
  | .mma .. => .family
  | .gather => .budgeted .gather
  | .budgeted cls _ => .budgeted cls

end Accounting.Gates
