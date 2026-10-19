import Definitions.Accounting.Isa.Classes

/-!
# sm_90a's instruction table (trusted)

The H200's (sm_90a) instruction classes (note:20261019T0520Z-draft-mm-isa): the mma.sync forms and the classes outside
the tensor cores (`Isa/Classes.lean`), and its wgmma forms (`Gmma`). The census's mma.sync kinds are `Sync`'s:
IMMA16832_S8 and IMMA16816_S8 are `IMMA16832` and `IMMA16816`, HMMA16816_F16 and HMMA16816_BF16 are `HMMA16816`,
HMMA1688_TF32 is `HMMA1688T`, BMMA168256_B1 is `BMMA168256`. sm_90a has no tcgen05 or tensor memory.
-/

namespace Accounting.Isa

/-- **sm_90a's wgmma forms** (PTX ISA 9.4 §9.7.17): the census's (`machine_model/data/devices/sm_90/census.csv`), and
`GMMASP` (every sparse form, §9.7.17.5.3) and `HGMMA_F16ACC` (an f16 accumulator). -/
inductive Gmma
  | IGMMA_S8_N8 | IGMMA_S8_N16 | IGMMA_S8_N32 | IGMMA_S8_N64 | IGMMA_S8_N128 | IGMMA_S8_N192 | IGMMA_S8_N256
  | IGMMA_S8_N256_RS | IGMMA_U8S8_N256
  | HGMMA_BF16_N128 | HGMMA_BF16_N256 | HGMMA_F16_N256 | HGMMA_TF32_N256
  | QGMMA_E4M3_N128 | QGMMA_E4M3_N256 | QGMMA_E5M2_N256
  | BGMMA_B1_N256
  | GMMASP | HGMMA_F16ACC
  deriving DecidableEq, Fintype, Repr

/-- **The wgmma rows**: a warp's wgmma m64nNk writes its quarter of the warpgroup's m64 × N accumulator, 16·N 32-bit
words (4096 at N = 256), or 8·N with an f16 accumulator. -/
def Gmma.row : Gmma → Row
  | .IGMMA_S8_N8 => Row.mma .wgmma .s8 .s8 .s32 128
  | .IGMMA_S8_N16 => Row.mma .wgmma .s8 .s8 .s32 256
  | .IGMMA_S8_N32 => Row.mma .wgmma .s8 .s8 .s32 512
  | .IGMMA_S8_N64 => Row.mma .wgmma .s8 .s8 .s32 1024
  | .IGMMA_S8_N128 => Row.mma .wgmma .s8 .s8 .s32 2048
  | .IGMMA_S8_N192 => Row.mma .wgmma .s8 .s8 .s32 3072
  | .IGMMA_S8_N256 => Row.mma .wgmma .s8 .s8 .s32 4096
  | .IGMMA_S8_N256_RS => Row.mma .wgmma .s8 .s8 .s32 4096
  | .IGMMA_U8S8_N256 => Row.mma .wgmma .u8 .s8 .s32 4096
  | .HGMMA_BF16_N128 => Row.mma .wgmma .bf16 .bf16 .f32 2048
  | .HGMMA_BF16_N256 => Row.mma .wgmma .bf16 .bf16 .f32 4096
  | .HGMMA_F16_N256 => Row.mma .wgmma .f16 .f16 .f32 4096
  | .HGMMA_TF32_N256 => Row.mma .wgmma .tf32 .tf32 .f32 4096
  | .QGMMA_E4M3_N128 => Row.mma .wgmma .e4m3 .e4m3 .f32 2048
  | .QGMMA_E4M3_N256 => Row.mma .wgmma .e4m3 .e4m3 .f32 4096
  | .QGMMA_E5M2_N256 => Row.mma .wgmma .e5m2 .e5m2 .f32 4096
  | .BGMMA_B1_N256 => Row.mma .wgmma .b1 .b1 .s32 4096
  | .GMMASP => Row.mma .wgmma .s8 .s8 .s32 4096
  | .HGMMA_F16ACC => Row.mma .wgmma .f16 .f16 .f16 2048

/-- **sm_90a's instruction classes**: an mma.sync form, a wgmma form, or a class outside the tensor cores. -/
inductive Sm90
  | sync (s : Sync)
  | gmma (g : Gmma)
  | base (c : Base)
  deriving DecidableEq, Fintype, Repr

/-- **sm_90a's rows**. -/
def Sm90.row : Sm90 → Row
  | .sync s => s.row
  | .gmma g => g.row
  | .base c => c.row

end Accounting.Isa
