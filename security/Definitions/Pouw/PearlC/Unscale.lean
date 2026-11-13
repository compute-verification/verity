module

public import Definitions.Core.Fp32

/-!
# The served y: Pearl-C's unscale (`PearlCUnscale_v1`)

The Pearl-C kernels' epilogue (`store_y`), which serves y from a tile's checked U: each row's and column's reciprocal
ia = RN_f32(1/α) (`__frcp_rn`, `F32Div_v2` of 1.0), then each word
`__float2bfloat16_rn(__fmul_rn(__fmul_rn(U_ij, ia_i), ib_j))`:

    y_ij = RN_bf16(RN_f32(RN_f32(U_ij · RN_f32(1/α_A,i)) · RN_f32(1/α_B,j)))

with `F2fpBf16_v1`'s rounding (nearest even, every NaN `0x7FFF`). It is the registry's
`verity.definitions.registry.pearl_c_tile.PearlCUnscale_v1` at `BIAS` off, op for op, and core's pin
`verity.core.protocols.pouw.schemes.pearl_c_u.y` wherever that pin's words stay finite (the pin refuses the rest; so does
the check, `PouwCheck.PearlPlus.Rule.yOf`). The committed y (`served_y`, y's real rows as vLLM gets them) is this word:
torch's bias add (`BIAS`, `Bf16Add_v1`) comes after it and is not committed. α is `pearl_c.s5_stats`' of the rows' FP32
words. The H200's stored y is this on every real cell of a served decode tile (`test_pouw_pearl_c_u.H200_Y`).
-/

@[expose] public section

namespace Pouw.PearlC

open Verity.Fp32

/-- The FP32 word of 1.0. -/
def UNSCALE_ONE : Nat := 0x3F800000

/-- `PearlCUnscale_v1`'s word at `BIAS` off: y_ij from U_ij and its row's and column's α. -/
def unscale (u alphaA alphaB : Nat) : Nat :=
  f32ToBf16RnBits (f32MulV2 (f32MulV2 u (f32DivV2 UNSCALE_ONE alphaA)) (f32DivV2 UNSCALE_ONE alphaB)) BF16_NAN_F2FP

/-- `PearlCUnscale_v1{TM, TN, BIAS = 0}`: a TM × TN tile's served y from its U (by row, then column), A's rows' α and the
weight's rows' α. -/
def unscaleTile (u : List (List Nat)) (alphaA alphaB : List Nat) : List (List Nat) :=
  (u.zip alphaA).map fun (row, a) => (row.zip alphaB).map fun (w, b) => unscale w a b

end Pouw.PearlC
