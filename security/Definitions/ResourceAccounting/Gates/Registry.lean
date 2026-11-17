module

public import Definitions.ResourceAccounting.Gates.Model

/-! Every primitive of the registry in exactly one class of the model; `classOf` is what property/v1's closed table is to agree with, id for id. -/

@[expose] public section

namespace ResourceAccounting.Gates

/-- A primitive's class. -/
inductive Class
  | input (b : Binding)
  | const
  | affine
  /-- an s8-family step: `IntOp.mma a b k satfinite` -/
  | family (a b : Int8Fmt) (k : ℕ) (satfinite : Bool)
  /-- a gather: `IntOp.gather low fill`, its column's word at its index or `fill` outside it (or a table read's word at
  its index's low bits), never any other function -/
  | gather (low : Option ℕ) (fill : ℤ)
  | budgeted (cls : Budget)
  deriving DecidableEq, Repr

/-- The registry's primitives, each with its class. -/
def registry : List (String × Class) := [
  ("AmpereBF16TcDot16_v2", .budgeted .rounding),
  ("And2_v1", .budgeted .compareSelect),
  ("Bf16AddF2fp_v1", .budgeted .rounding),
  ("Bf16Add_v1", .budgeted .rounding),
  ("Bf16GtStrict_v1", .budgeted .compareSelect),
  ("Bf16MulBf16_v1", .budgeted .rounding),
  ("Bf16MulF32_v1", .budgeted .rounding),
  ("Bf16ToF32_v1", .affine),
  ("BitAnd_v1", .budgeted .compareSelect),
  ("BitNot_v1", .budgeted .compareSelect),
  ("BitOr_v1", .budgeted .compareSelect),
  ("BitXor_v1", .budgeted .compareSelect),
  ("BlackwellE4m3QmmaDot32_v1", .budgeted .rounding),
  ("BlackwellE5m2QmmaDot32_v1", .budgeted .rounding),
  ("BlackwellMxf4OmmaDot64_v1", .budgeted .rounding),
  ("BlackwellNvf4OmmaDot64_v1", .budgeted .rounding),
  ("BlackwellS8ImmaDot32_v1", .family .s8 .s8 32 true),
  ("BlackwellS8ImmaDot32_v2", .family .s8 .s8 32 true),
  ("BlackwellU8S8ImmaDot32_v1", .family .u8 .s8 32 false),
  ("BlackwellU8S8ImmaDot32_v2", .family .u8 .s8 32 false),
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
  ("F32Sat_v1", .budgeted .compareSelect),
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
  ("I32Mul_v1", .budgeted .rounding),
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
  ("NcpDot16I7_v1", .budgeted .rounding),
  ("NcpRneClampI7_v1", .budgeted .rounding),
  ("Not_v1", .budgeted .compareSelect),
  ("NvLogf_v1", .budgeted .rounding),
  ("NvLogf_v2", .budgeted .rounding),
  ("PouwAdd64Hi_v1", .budgeted .rounding),
  ("PouwAdd64Lo_v1", .budgeted .rounding),
  ("PouwAdd64_v1", .budgeted .rounding),
  ("PouwAdd64_v2", .budgeted .rounding),
  ("PouwAdd8_v1", .budgeted .rounding),
  ("PouwAddSub8_v1", .budgeted .rounding),
  ("PouwBf16Neg_v1", .budgeted .rounding),
  ("PouwBlackwellE4m3SaltDead_v1", .budgeted .compareSelect),
  ("PouwBlackwellE4m3Window_v1", .budgeted .rounding),
  ("PouwCallSeeds_v1", .budgeted .rounding),
  ("PouwCallSeeds_v2", .budgeted .rounding),
  ("PouwCh32_v1", .budgeted .rounding),
  ("PouwChi32_v1", .budgeted .rounding),
  ("PouwDigitHi_v1", .budgeted .rounding),
  ("PouwDigitHi_v2", .budgeted .rounding),
  ("PouwDigitLo_v1", .budgeted .rounding),
  ("PouwDigitLo_v2", .budgeted .rounding),
  ("PouwDigitZ_v1", .budgeted .compareSelect),
  ("PouwDigitZ_v2", .budgeted .compareSelect),
  ("PouwDot16U8S8_v1", .family .u8 .s8 16 false),
  ("PouwDuoAbsorb_v1", .budgeted .fold),
  ("PouwETilde_v1", .budgeted .rounding),
  ("PouwF32SqrtRn_v1", .budgeted .rounding),
  ("PouwFormX3_v1", .budgeted .rounding),
  ("PouwFormX_v1", .budgeted .rounding),
  ("PouwFx1Absorb_v1", .budgeted .fold),
  ("PouwFx1Absorb_v2", .budgeted .fold),
  ("PouwHopperE4m3SaltDead_v1", .budgeted .compareSelect),
  ("PouwHopperE4m3Window_v1", .budgeted .rounding),
  ("PouwI32ToF32Rn_v1", .budgeted .rounding),
  ("PouwI8AddBit_v1", .budgeted .rounding),
  ("PouwI8AddBit_v2", .budgeted .rounding),
  ("PouwI8Add_v1", .budgeted .rounding),
  ("PouwI8Add_v2", .budgeted .rounding),
  ("PouwI8Neg_v1", .budgeted .rounding),
  ("PouwI8Neg_v2", .budgeted .rounding),
  ("PouwLow8_v1", .budgeted .rounding),
  ("PouwLow8_v2", .budgeted .rounding),
  ("PouwMaj32_v1", .budgeted .rounding),
  ("PouwMask32_v1", .budgeted .compareSelect),
  ("PouwMask32_v2", .budgeted .compareSelect),
  ("PouwNegBetaSum16_v1", .budgeted .rounding),
  ("PouwNoiseBlock_v1", .budgeted .rounding),
  ("PouwNoiseBlock_v2", .budgeted .rounding),
  ("PouwNoiseByte_v1", .budgeted .rounding),
  ("PouwOneStream2_v1", .budgeted .rounding),
  ("PouwOneStream2_v2", .budgeted .rounding),
  ("PouwOneStream3_v1", .budgeted .rounding),
  ("PouwOneStream3_v2", .budgeted .rounding),
  ("PouwOrBit_v1", .budgeted .compareSelect),
  ("PouwOrBit_v2", .budgeted .compareSelect),
  ("PouwPackBE4_v1", .budgeted .rounding),
  ("PouwPeelSeg_v1", .budgeted .rounding),
  ("PouwPeelSeg_v2", .budgeted .rounding),
  ("PouwResAdd_v1", .budgeted .rounding),
  ("PouwResAdd_v2", .budgeted .rounding),
  ("PouwRotl64Hi_v1", .budgeted .rounding),
  ("PouwRotl64Lo_v1", .budgeted .rounding),
  ("PouwSegDot0_v1", .family .s8 .s8 32 false),
  ("PouwSegDot0_v2", .family .s8 .s8 32 false),
  ("PouwSegDot_v1", .family .s8 .s8 32 false),
  ("PouwSegDot_v2", .family .s8 .s8 32 false),
  ("PouwSel32_v1", .budgeted .compareSelect),
  ("PouwSha512BigSigma0Hi_v1", .budgeted .rounding),
  ("PouwSha512BigSigma0Lo_v1", .budgeted .rounding),
  ("PouwSha512BigSigma1Hi_v1", .budgeted .rounding),
  ("PouwSha512BigSigma1Lo_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma0Hi_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma0Lo_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma1Hi_v1", .budgeted .rounding),
  ("PouwSha512SmallSigma1Lo_v1", .budgeted .rounding),
  ("PouwSqAcc_v1", .budgeted .rounding),
  ("PouwSqCeil_v1", .budgeted .rounding),
  ("PouwSqGe_v1", .budgeted .compareSelect),
  ("PouwSub8_v1", .budgeted .rounding),
  ("PouwSx8_v1", .budgeted .rounding),
  ("PouwSx8_v2", .budgeted .rounding),
  ("PouwU129_v1", .budgeted .rounding),
  ("PouwU129_v2", .budgeted .rounding),
  ("PouwUnitSeeds_v1", .budgeted .rounding),
  ("PouwUnitSeeds_v2", .budgeted .rounding),
  ("PouwWinOk_v1", .budgeted .compareSelect),
  ("PouwWindowWord_v1", .budgeted .rounding),
  ("PouwWrapBits_v1", .budgeted .compareSelect),
  ("PouwWrapBits_v2", .budgeted .compareSelect),
  ("PouwWrapSeg_v1", .budgeted .rounding),
  ("PouwWrapSeg_v2", .budgeted .rounding),
  ("PouwXWrap_v1", .budgeted .compareSelect),
  ("PouwXWrap_v2", .budgeted .compareSelect),
  ("PouwXinFinal_v1", .budgeted .rounding),
  ("PouwXinFinal_v2", .budgeted .rounding),
  ("PouwXor32_v1", .budgeted .rounding),
  ("PouwXor32_v2", .budgeted .rounding),
  ("RopeOutAdd_v1", .budgeted .rounding),
  ("RopeOut_v1", .budgeted .rounding),
  ("RsqrtApprox_v1", .budgeted .rounding),
  ("S32ByteCarry_v1", .budgeted .rounding),
  ("S32ByteCarry_v2", .budgeted .rounding),
  ("S32ClampRest_v1", .budgeted .rounding),
  ("S32ClampRest_v2", .budgeted .rounding),
  ("S32ClampS8_v1", .budgeted .compareSelect),
  ("S32ClampS8_v2", .budgeted .compareSelect),
  ("S32LowByte_v1", .budgeted .rounding),
  ("S32LowByte_v2", .budgeted .rounding),
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
  ("Xor2_v1", .budgeted .compareSelect)]

/-- The kernel-level families, by their id's prefix (`numbered`: the prefix, a numeral and `_v1`), each gather with its
out-of-range constant read signed at its width: bf16's NaN `0x7FFF` and f32's `0x7FC00000` (`Evaluate.Act.gather`),
i32's `−1`, e4m3's NaN `0x7F`, e2m1's zero code and e8m0's NaN `0xFF` (`fp8_moe_experts._gather`). -/
def families : List (String × Class) := [
  ("Input", .input .run),
  ("RegisteredInput", .input .registered),
  ("GatherBf16x", .gather none 0x7FFF),
  ("GatherF32x", .gather none 0x7FC00000),
  ("GatherI32x", .gather none (-1)),
  ("GatherE4m3x", .gather none 0x7F),
  ("GatherE2m1x", .gather none 0),
  ("GatherE8m0x", .gather none (-1)),
  ("BitAtx", .budgeted .rounding),
  ("TopPMaskWordx", .budgeted .compareSelect)]

/-- The value of a positive decimal numeral with no leading `0`, else none. -/
def numeral : List Char → Option ℕ
  | [] => none
  | c :: cs =>
      if c != '0' && (c :: cs).all Char.isDigit then
        some ((c :: cs).foldl (fun n d => 10 * n + (d.toNat - '0'.toNat)) 0)
      else none

/-- The numeral `n` of an id `{pre}{n}_v1` (`numeral`), else none. -/
def numbered (pre id : List Char) : Option ℕ :=
  if pre.isPrefixOf id && "_v1".toList.isSuffixOf id then
    numeral ((id.drop pre.length).take (id.length - pre.length - 3))
  else none

/-- A table read's index bits, the `n` of `TableRead{n}x{w}_v1` (`n` and `w` numerals): it reads its table at its
index's low `n` bits. -/
def tableReadBits (id : String) : Option ℕ :=
  let n := (id.toList.drop 9).takeWhile Char.isDigit
  (numbered ("TableRead".toList ++ n ++ ['x']) id.toList).bind fun _ => numeral n

/-- A primitive id's class: the registry's entry, else a table read's (fill 0, never read: its table has `2^n`
entries), else its family's, else none (refused). -/
def classOf (id : String) : Option Class :=
  (registry.lookup id).orElse fun _ => ((tableReadBits id).map fun n => .gather (some n) 0).orElse fun _ =>
    (families.find? fun e => (numbered e.1.toList id.toList).isSome).map Prod.snd

/-- `Bf16ToF32_v1`'s operation: `×2^16`, from the bf16 word's s16 reading to the f32 word's s32 reading. -/
def bf16ToF32Op {ι : Type} : IntOp ι := .affine 0 [2 ^ 16]

def IntOp.cls {ι : Type} : IntOp ι → Class
  | .input b _ => .input b
  | .const _ => .const
  | .affine .. => .affine
  | .mma a b k sat => .family a b k sat
  | .gather low d => .gather low d
  | .budgeted cls _ => .budgeted cls

/-- A class as its primitive's id lists it: a coin input's is a run input's (`Input{n}_v1`), its binding its port's. -/
def Class.listed : Class → Class
  | .input .coin => .input .run
  | c => c

/-- **Every gate is its primitive's** (what the census checks, refusing the circuit otherwise): its primitive is
listed, its operation is in the primitive's class (`Class.listed`), and a gate at a dot step (`Place.dot`) is an
s8-family step. Its role is then its primitive's and its statics' (`IntGate.role`). -/
def Classed {ι : Type} (C : IntDag ι) : Prop :=
  ∀ g ∈ C, classOf g.prim = some g.op.cls.listed ∧ ∀ K L tail s, g.place = .dot K L tail s → g.op.IsMma

end ResourceAccounting.Gates
