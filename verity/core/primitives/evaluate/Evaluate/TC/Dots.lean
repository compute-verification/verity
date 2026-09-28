import Evaluate.TC.Spec
import Evaluate.TC.Fp4

/-!
# The tensor-core dot primitives

The seven accumulator steps `verity.ml.circuits.ops.prims` registers, each one accumulator coordinate of one instruction:
the accumulator word and the operand words in parameter order (the walk gives a gate the leaves of its operands, so an
`Array<16, BF16>` operand is 16 words). Each is the total step of `Evaluate.TC.Spec`, `Evaluate.TC.Fp8Atom` or
`Evaluate.TC.Fp4` on its pipeline, and names the Security definition it must equal on the primitive's domain (the
accumulator below `2^32`, each operand list of its length and every word below `2^w`); the FP4 steps add the rules the
primitives state for what the block-scaled model leaves out.
-/

namespace Evaluate.TC

open Fp8Atom Fp4

/-- `AmpereBF16TcDot16_v2`: `tc_dot_total(AMPERE_BF16_M16N8K16, acc, a, b)`, `mma.sync.m16n8k16.bf16` on the A100 and the
RTX 4090. Must equal `Verity.TC.Spec.tcDotTotal Verity.TC.AMPERE_BF16_M16N8K16 acc a b`. -/
def ampereBf16Dot16 (acc : Nat) (a b : List Nat) : Nat := Spec.tcDotTotal Spec.AMPERE_BF16_M16N8K16 acc a b

/-- `HopperBF16WgmmaDot16_v1`: `tc_dot_total(HOPPER_BF16_WGMMA_K16, acc, a, b)`, one coordinate of
`wgmma...m64nNk16.f32.bf16.bf16` on the H100. Must equal `Verity.TC.Spec.tcDotTotal Verity.TC.HOPPER_BF16_WGMMA_K16 acc a b`. -/
def hopperBf16Dot16 (acc : Nat) (a b : List Nat) : Nat := Spec.tcDotTotal Spec.HOPPER_BF16_WGMMA_K16 acc a b

/-- `HopperE4m3QgmmaDot32_v1`: `tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, acc, a, b)`, the 14-bit adder. Must equal
`Pouw.Fp8Atom.hopperStep acc (fun i => a[i]) (fun i => b[i])`. -/
def hopperE4m3Dot32 (acc : Nat) (a b : List Nat) : Nat := hopperE4m3K32.step acc (a.map code) (b.map code)

/-- `BlackwellE4m3QmmaDot32_v1`: `tc_dot_total_e4m3(BLACKWELL_SM120_E4M3_M16N8K32, acc, a, b)`, sm_120's 26-bit adder. Must
equal `Pouw.PearlC.sm120Step acc (fun i => a[i]) (fun i => b[i])`. -/
def blackwellE4m3Dot32 (acc : Nat) (a b : List Nat) : Nat := sm120E4m3.step acc (a.map code) (b.map code)

/-- `BlackwellE5m2QmmaDot32_v1`: `tc_dot_total_e5m2(BLACKWELL_SM120_E5M2_M16N8K32, acc, a, b)`. Security: none yet. -/
def blackwellE5m2Dot32 (acc : Nat) (a b : List Nat) : Nat := sm120E5m2.stepE5m2 acc (a.map code) (b.map code)

/-- The rules of the FP4 primitives outside the model, then `p.step`: a NaN accumulator gives `0x7FFFFFFF` and an infinite
one itself; else the model's word, and where the sum leaves binary32 the infinity of its sign. -/
def blockScaled (p : Fp4Atom) (acc : Nat) (a b sa sb : List Nat) : Nat :=
  let acc := acc &&& 0xFFFFFFFF
  if acc &&& 0x7F800000 = 0x7F800000 then (if acc &&& 0x7FFFFF ≠ 0 then 0x7FFFFFFF else acc)
  else
    let (a, b, sa, sb) := (a.map nib, b.map nib, sa.map byte, sb.map byte)
    match p.step acc a b sa sb with
    | some w => w
    | none => if p.sumNegative acc a b sa sb then 0xFF800000 else 0x7F800000

/-- `BlackwellNvf4OmmaDot64_v1`: a scale byte whose low seven bits are `0x7F` gives `0x7FFFFFFF`; else `blockScaled` on
`sm120Nvf4` with bit 7 of each scale byte cleared, whose sum cannot leave binary32 from a finite accumulator (the Python
model would raise there). Must equal, where neither rule applies, `Pouw.PearlC.Fp4.sm120Nvf4.step acc a b sa sb` (which is
then `some`). -/
def blackwellNvf4Dot64 (acc : Nat) (a b sa sb : List Nat) : Nat :=
  if (sa ++ sb).any (· &&& 0x7F == 0x7F) then 0x7FFFFFFF
  else blockScaled sm120Nvf4 acc a b (sa.map (· &&& 0x7F)) (sb.map (· &&& 0x7F))

/-- `BlackwellMxf4OmmaDot64_v1`: a scale byte `0xFF` gives `0x7FFFFFFF`; else `blockScaled` on `sm120Mxf4`. Must equal, where
neither rule applies and the sum stays in binary32, `Pouw.PearlC.Fp4.sm120Mxf4.step acc a b sa sb`. -/
def blackwellMxf4Dot64 (acc : Nat) (a b sa sb : List Nat) : Nat :=
  if sa.contains 0xFF || sb.contains 0xFF then 0x7FFFFFFF else blockScaled sm120Mxf4 acc a b sa sb

end Evaluate.TC
