module

public import Evaluate.Prims
public import Definitions.Core.Scalar
public import Definitions.Core.Fp32
public import Definitions.Core.Prims

/-! The evaluator's primitives, each equal to the definition of `Definitions.Core` it copies; the gates' equality to
the Boolean circuit's, the FP8 and FP4 steps and the PoUW words are not stated, because their definitions import
Mathlib. -/

@[expose] public section

namespace Properties.Evaluator

namespace Scalar

def F32IsNan : Prop := Evaluate.Scalar.f32IsNan = Verity.Scalar.f32IsNan
def Order : Prop := Evaluate.Scalar.order = Verity.Scalar.order
def F32NanPtx : Prop := Evaluate.Scalar.F32_NAN_PTX = Verity.Scalar.F32_NAN_PTX
def F32Fabs : Prop := Evaluate.Scalar.f32Fabs = Verity.Scalar.f32Fabs
def F32Neg : Prop := Evaluate.Scalar.f32Neg = Verity.Scalar.f32Neg
def F32Fmaxf : Prop := Evaluate.Scalar.f32Fmaxf = Verity.Scalar.f32Fmaxf
def I32Le : Prop := Evaluate.Scalar.i32Le = Verity.Scalar.i32Le
def I32Add : Prop := Evaluate.Scalar.i32Add = Verity.Scalar.i32Add
def BitAnd : Prop := Evaluate.Scalar.bitAnd = Verity.Scalar.bitAnd
def SelectI32 : Prop := Evaluate.Scalar.selectI32 = Verity.Scalar.selectI32
def SelectBf16 : Prop := Evaluate.Scalar.selectBf16 = Verity.Scalar.selectBf16

end Scalar

namespace Fp32

/-- Each decoded value as Security's of the same constructor and fields. -/
def Decoded.toCore : Evaluate.Fp32.Decoded → Verity.Fp32.Decoded
  | .nan => .nan
  | .inf s => .inf s
  | .fin s m => .fin s m

def Decoded.ofCore : Verity.Fp32.Decoded → Evaluate.Fp32.Decoded
  | .nan => .nan
  | .inf s => .inf s
  | .fin s m => .fin s m

def F32Nan : Prop := Evaluate.Fp32.F32_NAN = Verity.Fp32.F32_NAN
def Bf16NanTorch : Prop := Evaluate.Fp32.BF16_NAN_TORCH = Verity.Fp32.BF16_NAN_TORCH
def Bf16NanF2fp : Prop := Evaluate.Fp32.BF16_NAN_F2FP = Verity.Fp32.BF16_NAN_F2FP
def ZeroWord : Prop := Evaluate.Fp32.zeroWord = Verity.Fp32.zeroWord
def InfWord : Prop := Evaluate.Fp32.infWord = Verity.Fp32.infWord

/-- `decode` on every natural number: the evaluator's mask to 32 bits changes none of the bits Security's reads. -/
def Decode : Prop := ∀ u, Decoded.toCore (Evaluate.Fp32.decode u) = Verity.Fp32.decode u

def RoundRn : Prop := Evaluate.Fp32.roundRn = Verity.Fp32.roundRn
def RoundSum : Prop := Evaluate.Fp32.roundSum = Verity.Fp32.roundSum

def DecodedMul : Prop :=
  ∀ x y, Decoded.toCore (Evaluate.Fp32.Decoded.mul x y) = Verity.Fp32.Decoded.mul (Decoded.toCore x) (Decoded.toCore y)

def DecodedAddRn : Prop :=
  ∀ nan d x y,
    Evaluate.Fp32.Decoded.addRn nan d x y = Verity.Fp32.Decoded.addRn nan d (Decoded.toCore x) (Decoded.toCore y)

def F32GpuNan : Prop := Evaluate.Fp32.f32GpuNan = Verity.Fp32.f32GpuNan
def FtzBits : Prop := Evaluate.Fp32.ftzBits = Verity.Fp32.ftzBits

/-- The operations with the NaN as a parameter, at the NaN each of Security's fixes. -/
def AddWith : Prop := Evaluate.Fp32.addWith Evaluate.Scalar.F32_NAN_PTX = Verity.Fp32.f32AddV2
def MulWith : Prop := Evaluate.Fp32.mulWith Evaluate.Scalar.F32_NAN_PTX = Verity.Fp32.f32MulV2
def DivWith : Prop := Evaluate.Fp32.divWith Evaluate.Scalar.F32_NAN_PTX = Verity.Fp32.f32DivV2
def FmaWith : Prop := Evaluate.Fp32.fmaWith Evaluate.Fp32.F32_NAN = Verity.Fp32.f32FmaBits

def F32FmaBits : Prop := Evaluate.Fp32.f32FmaBits = Verity.Fp32.f32FmaBits
def F32Fma : Prop := Evaluate.Fp32.f32Fma = Verity.Fp32.f32Fma
def F32FmaFtz : Prop := Evaluate.Fp32.f32FmaFtz = Verity.Fp32.f32FmaFtz
def F32FmaSubFtz : Prop := Evaluate.Fp32.f32FmaSubFtz = Verity.Fp32.f32FmaSubFtz
def F32AddV2 : Prop := Evaluate.Fp32.f32AddV2 = Verity.Fp32.f32AddV2
def F32MulV2 : Prop := Evaluate.Fp32.f32MulV2 = Verity.Fp32.f32MulV2
def F32DivV2 : Prop := Evaluate.Fp32.f32DivV2 = Verity.Fp32.f32DivV2
def F32FmaV2 : Prop := Evaluate.Fp32.f32FmaV2 = Verity.Fp32.f32FmaV2
def F32AddFtzV2 : Prop := Evaluate.Fp32.f32AddFtzV2 = Verity.Fp32.f32AddFtzV2
def F32SubFtzV2 : Prop := Evaluate.Fp32.f32SubFtzV2 = Verity.Fp32.f32SubFtzV2
def F32MulFtzV2 : Prop := Evaluate.Fp32.f32MulFtzV2 = Verity.Fp32.f32MulFtzV2
def F32FmaFtzV2 : Prop := Evaluate.Fp32.f32FmaFtzV2 = Verity.Fp32.f32FmaFtzV2
def F32FmaSubFtzV2 : Prop := Evaluate.Fp32.f32FmaSubFtzV2 = Verity.Fp32.f32FmaSubFtzV2
def F32Gt : Prop := Evaluate.Fp32.f32Gt = Verity.Fp32.f32Gt
def F32MaxBits : Prop := Evaluate.Fp32.f32MaxBits = Verity.Fp32.f32MaxBits
def F32Max : Prop := Evaluate.Fp32.f32Max = Verity.Fp32.f32Max
def GuardNegInfZero : Prop := Evaluate.Fp32.guardNegInfZero = Verity.Fp32.guardNegInfZero
def Bf16ToF32Bits : Prop := Evaluate.Fp32.bf16ToF32Bits = Verity.Fp32.bf16ToF32Bits
def F32ToBf16RnBits : Prop := Evaluate.Fp32.f32ToBf16RnBits = Verity.Fp32.f32ToBf16RnBits
def Bf16Add : Prop := Evaluate.Fp32.bf16Add = Verity.Fp32.bf16Add
def Bf16AddF2fp : Prop := Evaluate.Fp32.bf16AddF2fp = Verity.Fp32.bf16AddF2fp
def Bf16MulF32 : Prop := Evaluate.Fp32.bf16MulF32 = Verity.Fp32.bf16MulF32
def Bf16MulBf16 : Prop := Evaluate.Fp32.bf16MulBf16 = Verity.Fp32.bf16MulBf16

end Fp32

namespace Casts

def BitLength : Prop := Evaluate.Casts.bitLength = Verity.TC.bitLength
def Bf16ToF32 : Prop := Evaluate.Casts.bf16ToF32 = Verity.Prims.bf16ToF32
/-- `F32ToBf16Rn_v1`, which Security doesn't name: its doc's `Verity.Fp32.f32ToBf16RnBits a BF16_NAN_TORCH`. -/
def F32ToBf16Rn : Prop :=
  Evaluate.Casts.f32ToBf16Rn = fun a => Verity.Fp32.f32ToBf16RnBits a Verity.Fp32.BF16_NAN_TORCH
/-- `cvt.rn.bf16.f32` on every natural number: the evaluator's mask to 32 bits changes neither the NaN test nor the 16
bits the rounding keeps. -/
def F2fpBf16 : Prop := Evaluate.Casts.f2fpBf16 = Verity.Prims.f2fpBf16
def E4m3Nan : Prop := Evaluate.Casts.E4M3_NAN = Verity.Prims.E4M3_NAN
def E4m3ToF32 : Prop := Evaluate.Casts.e4m3ToF32 = Verity.Prims.e4m3ToF32
def DyadicLt : Prop := Evaluate.Casts.dyadicLt = Verity.Prims.dyadicLt
def RoundEvenDyadic : Prop := Evaluate.Casts.roundEvenDyadic = Verity.Prims.roundEvenDyadic
def F32ToE4m3Sat : Prop := Evaluate.Casts.f32ToE4m3Sat = Verity.Prims.f32ToE4m3Sat
def E2m1NanCode : Prop := Evaluate.Casts.E2M1_NAN_CODE = Verity.Prims.E2M1_NAN_CODE
def E2m1Up : Prop := Evaluate.Casts.E2M1_UP = Verity.Prims.E2M1_UP
def F32ToE2m1Sat : Prop := Evaluate.Casts.f32ToE2m1Sat = Verity.Prims.f32ToE2m1Sat

end Casts

namespace TC

open Evaluate.TC.Spec (Pipeline Term Value Cls)

/-- Each pipeline as Security's of the same fields. -/
def Pipeline.toCore (p : Pipeline) : Verity.TC.Pipeline := ⟨p.groups, p.width, p.zeroExponent⟩

def Pipeline.ofCore (p : Verity.TC.Pipeline) : Pipeline := ⟨p.groups, p.width, p.zeroExponent⟩

/-- Each term as Security's of the same fields. -/
def Term.toCore (t : Term) : Verity.TC.Spec.Term := ⟨t.negative, t.exponent, t.magnitude⟩

def Term.ofCore (t : Verity.TC.Spec.Term) : Term := ⟨t.negative, t.exponent, t.magnitude⟩

/-- Each value as Security's of the same constructor, a finite one's term by `Term.toCore`. -/
def Value.toCore : Value → Verity.TC.Spec.Value
  | .nan => .nan
  | .inf s => .inf s
  | .fin t => .fin (Term.toCore t)

def Value.ofCore : Verity.TC.Spec.Value → Value
  | .nan => .nan
  | .inf s => .inf s
  | .fin t => .fin (Term.ofCore t)

/-- Each class as Security's of the same name. -/
def Cls.toCore : Cls → Verity.TC.Spec.Cls
  | .nan => .nan
  | .inf => .inf
  | .zero => .zero
  | .finite => .finite

def Cls.ofCore : Verity.TC.Spec.Cls → Cls
  | .nan => .nan
  | .inf => .inf
  | .zero => .zero
  | .finite => .finite

def BitLength : Prop := Evaluate.TC.Spec.bitLength = Verity.TC.bitLength
def PipelineK : Prop := ∀ p, Evaluate.TC.Spec.Pipeline.k p = Verity.TC.Pipeline.k (Pipeline.toCore p)
def HopperBf16WgmmaK16 : Prop :=
  Pipeline.toCore Evaluate.TC.Spec.HOPPER_BF16_WGMMA_K16 = Verity.TC.HOPPER_BF16_WGMMA_K16
def AmpereBf16M16n8k16 : Prop :=
  Pipeline.toCore Evaluate.TC.Spec.AMPERE_BF16_M16N8K16 = Verity.TC.AMPERE_BF16_M16N8K16
def Zero : Prop := Term.toCore Evaluate.TC.Spec.ZERO = Verity.TC.Spec.ZERO
def Bf16Product : Prop := ∀ a b, Term.toCore (Evaluate.TC.Spec.bf16Product a b) = Verity.TC.Spec.bf16Product a b
def Fp32Term : Prop := ∀ bits, Term.toCore (Evaluate.TC.Spec.fp32Term bits) = Verity.TC.Spec.fp32Term bits
def PackFp32 : Prop := ∀ t, Evaluate.TC.Spec.packFp32 t = Verity.TC.Spec.packFp32 (Term.toCore t)
def GroupSum : Prop :=
  ∀ terms width zeroExponent,
    Term.toCore (Evaluate.TC.Spec.groupSum terms width zeroExponent) =
      Verity.TC.Spec.groupSum (terms.map Term.toCore) width zeroExponent
def Bf16Class : Prop := ∀ w, Cls.toCore (Evaluate.TC.Spec.bf16Class w) = Verity.TC.Spec.bf16Class w
def Fp32Class : Prop := ∀ w, Cls.toCore (Evaluate.TC.Spec.fp32Class w) = Verity.TC.Spec.fp32Class w
def Bf16ProductTotal : Prop :=
  ∀ a b, Value.toCore (Evaluate.TC.Spec.bf16ProductTotal a b) = Verity.TC.Spec.bf16ProductTotal a b
def Fp32Value : Prop := ∀ bits, Value.toCore (Evaluate.TC.Spec.fp32Value bits) = Verity.TC.Spec.fp32Value bits
def ValueIsNaN : Prop := ∀ v, Evaluate.TC.Spec.Value.isNaN v = Verity.TC.Spec.Value.isNaN (Value.toCore v)
def ValueInfSign : Prop := ∀ v, Evaluate.TC.Spec.Value.infSign? v = Verity.TC.Spec.Value.infSign? (Value.toCore v)
def ValueTerm : Prop :=
  ∀ v, (Evaluate.TC.Spec.Value.term? v).map Term.toCore = Verity.TC.Spec.Value.term? (Value.toCore v)
def GroupStepTotal : Prop :=
  ∀ acc products width zeroExponent,
    Value.toCore (Evaluate.TC.Spec.groupStepTotal acc products width zeroExponent) =
      Verity.TC.Spec.groupStepTotal (Value.toCore acc) (products.map Value.toCore) width zeroExponent
def PackValue : Prop := ∀ v, Evaluate.TC.Spec.packValue v = Verity.TC.Spec.packValue (Value.toCore v)
def RunGroups : Prop :=
  ∀ width zeroExponent groups acc a b,
    Value.toCore (Evaluate.TC.Spec.runGroups width zeroExponent groups acc a b) =
      Verity.TC.Spec.runGroups width zeroExponent groups (Value.toCore acc) a b
def TcDotTotal : Prop :=
  ∀ p acc a b, Evaluate.TC.Spec.tcDotTotal p acc a b = Verity.TC.Spec.tcDotTotal (Pipeline.toCore p) acc a b

/-- `AmpereBF16TcDot16_v2`. -/
def AmpereBf16Dot16 : Prop :=
  Evaluate.TC.ampereBf16Dot16 = Verity.TC.Spec.tcDotTotal Verity.TC.AMPERE_BF16_M16N8K16
/-- `HopperBF16WgmmaDot16_v1`. -/
def HopperBf16Dot16 : Prop :=
  Evaluate.TC.hopperBf16Dot16 = Verity.TC.Spec.tcDotTotal Verity.TC.HOPPER_BF16_WGMMA_K16

end TC

namespace Gates

/-- `And<n>_v1` on bits is the conjunction of its operands. -/
def AndBits : Prop := ∀ xs : Array Bool, Evaluate.andBits (xs.map Bool.toNat) = (xs.all id).toNat

/-- `Xor<n>_v1` on bits is the XOR of its operands, as `Verity.Boolean.xorSum` folds it. -/
def XorBits : Prop := ∀ xs : Array Bool, Evaluate.xorBits (xs.map Bool.toNat) = (xs.toList.foldr Bool.xor false).toNat

/-- `Not_v1` on a bit is its negation. -/
def NotBit : Prop := ∀ b : Bool, Evaluate.notBit b.toNat = (!b).toNat

end Gates

end Properties.Evaluator
