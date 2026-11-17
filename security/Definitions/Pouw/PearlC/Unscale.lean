module

public import Definitions.Core.Fp32

/-! The served y, Pearl-C's unscale `PearlCUnscale_v1` at `BIAS` off: `y_ij = RN_bf16(RN_f32(RN_f32(U_ij · RN_f32(1/α_A,i)) · RN_f32(1/α_B,j)))`. -/

@[expose] public section

namespace Pouw.PearlC

open Verity.Fp32

/-- The FP32 word of 1.0. -/
def UNSCALE_ONE : Nat := 0x3F800000

/-- `PearlCUnscale_v1`'s word at `BIAS` off: y_ij from U_ij and its row's and column's α. -/
def unscale (u alphaA alphaB : Nat) : Nat :=
  f32ToBf16RnBits (f32MulV2 (f32MulV2 u (f32DivV2 UNSCALE_ONE alphaA)) (f32DivV2 UNSCALE_ONE alphaB)) BF16_NAN_F2FP

def unscaleTile (u : List (List Nat)) (alphaA alphaB : List Nat) : List (List Nat) :=
  (u.zip alphaA).map fun (row, a) => (row.zip alphaB).map fun (w, b) => unscale w a b

end Pouw.PearlC
