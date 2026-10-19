import Definitions.Accounting.Isa.Classes

/-!
# sm_100a's instruction table (trusted)

The B200's (sm_100a) instruction classes (note:20261019T0520Z-draft-mm-isa): the mma.sync forms and the classes outside
the tensor cores (`Isa/Classes.lean`), its tcgen05.mma forms (`Tc05`) and its tensor-memory instructions (`Tmem`). The
census's mma.sync kinds are `Sync`'s: IMMA16832_S8 and IMMA16816_S8 are `IMMA16832` and `IMMA16816`. sm_100a has no
wgmma.
-/

namespace Accounting.Isa

/-- **sm_100a's tcgen05.mma forms** (PTX ISA 9.4 §9.7.18.10): the census's
(`machine_model/data/devices/sm_100/census.csv`), and at M = 128, N = 256 a form for each kind and A type the census
never probed (RT-B200-EMP-2; B's type any of its kind's, none of which holds an int8 operand): `UTCHMMA_BF16`
(kind::f16 .bf16), `UTCHMMA_TF32` (kind::tf32), `UTCQMMA_E5M2`, `UTCQMMA_E3M2`, `UTCQMMA_E2M3` and `UTCQMMA_E2M1`
(kind::f8f6f4), `UTCQMMA_SF` (kind::mxf8f6f4 .e4m3) and `UTCQMMA_SF_E5M2` … `UTCQMMA_SF_E2M1` (its other types),
`UTCOMMA_SF` (kind::mxf4 and kind::mxf4nvf4, RT-ISA-4), `UTCMMA_SP` (every sparse form), and over a CTA pair
(cta_group::2, M = 256) `UTCOMMA_2CTA` (kind::mxf4 and kind::mxf4nvf4, RT-ISA-4) and `UTCMMA_2CTA` (every other
kind). -/
inductive Tc05
  | UTCIMMA_M64_N64 | UTCIMMA_M64_N128 | UTCIMMA_M64_N256
  | UTCIMMA_M128_N32 | UTCIMMA_M128_N64 | UTCIMMA_M128_N128 | UTCIMMA_M128_N256
  | UTCHMMA_F16_M128_N256 | UTCQMMA_E4M3_M128_N128 | UTCQMMA_E4M3_M128_N256
  | UTCHMMA_BF16 | UTCHMMA_TF32 | UTCQMMA_E5M2 | UTCQMMA_E3M2 | UTCQMMA_E2M3 | UTCQMMA_E2M1
  | UTCQMMA_SF | UTCQMMA_SF_E5M2 | UTCQMMA_SF_E3M2 | UTCQMMA_SF_E2M3 | UTCQMMA_SF_E2M1
  | UTCOMMA_SF | UTCMMA_SP | UTCOMMA_2CTA | UTCMMA_2CTA
  deriving DecidableEq, Repr

-- Mathlib's `deriving Fintype` fails on an enumeration at this revision (its `Finset.mem_mk` rewrite misses).
instance : Fintype Tc05 :=
  Fintype.ofList (α := Tc05) [.UTCIMMA_M64_N64, .UTCIMMA_M64_N128, .UTCIMMA_M64_N256, .UTCIMMA_M128_N32,
    .UTCIMMA_M128_N64, .UTCIMMA_M128_N128, .UTCIMMA_M128_N256, .UTCHMMA_F16_M128_N256, .UTCQMMA_E4M3_M128_N128,
    .UTCQMMA_E4M3_M128_N256, .UTCHMMA_BF16, .UTCHMMA_TF32, .UTCQMMA_E5M2, .UTCQMMA_E3M2, .UTCQMMA_E2M3, .UTCQMMA_E2M1,
    .UTCQMMA_SF, .UTCQMMA_SF_E5M2, .UTCQMMA_SF_E3M2, .UTCQMMA_SF_E2M3, .UTCQMMA_SF_E2M1, .UTCOMMA_SF, .UTCMMA_SP,
    .UTCOMMA_2CTA, .UTCMMA_2CTA] fun x => by
    cases x <;> decide +kernel

/-- **The tcgen05.mma rows**: a tcgen05.mma writes its M × N tensor-memory accumulator, M·N 32-bit words (32768 at
M = 128, N = 256; 65536 for a CTA pair at M = 256), and nothing else. -/
def Tc05.row : Tc05 → Row
  | .UTCIMMA_M64_N64 => Row.mma .tcgen05 .s8 .s8 .s32 4096
  | .UTCIMMA_M64_N128 => Row.mma .tcgen05 .s8 .s8 .s32 8192
  | .UTCIMMA_M64_N256 => Row.mma .tcgen05 .s8 .s8 .s32 16384
  | .UTCIMMA_M128_N32 => Row.mma .tcgen05 .s8 .s8 .s32 4096
  | .UTCIMMA_M128_N64 => Row.mma .tcgen05 .s8 .s8 .s32 8192
  | .UTCIMMA_M128_N128 => Row.mma .tcgen05 .s8 .s8 .s32 16384
  | .UTCIMMA_M128_N256 => Row.mma .tcgen05 .s8 .s8 .s32 32768
  | .UTCHMMA_F16_M128_N256 => Row.mma .tcgen05 .f16 .f16 .f32 32768
  | .UTCQMMA_E4M3_M128_N128 => Row.mma .tcgen05 .e4m3 .e4m3 .f32 16384
  | .UTCQMMA_E4M3_M128_N256 => Row.mma .tcgen05 .e4m3 .e4m3 .f32 32768
  | .UTCHMMA_BF16 => Row.mma .tcgen05 .bf16 .bf16 .f32 32768
  | .UTCHMMA_TF32 => Row.mma .tcgen05 .tf32 .tf32 .f32 32768
  | .UTCQMMA_E5M2 => Row.mma .tcgen05 .e5m2 .e5m2 .f32 32768
  | .UTCQMMA_E3M2 => Row.mma .tcgen05 .e3m2 .e3m2 .f32 32768
  | .UTCQMMA_E2M3 => Row.mma .tcgen05 .e2m3 .e2m3 .f32 32768
  | .UTCQMMA_E2M1 => Row.mma .tcgen05 .e2m1 .e2m1 .f32 32768
  | .UTCQMMA_SF => Row.mma .tcgen05 .e4m3 .e4m3 .f32 32768
  | .UTCQMMA_SF_E5M2 => Row.mma .tcgen05 .e5m2 .e5m2 .f32 32768
  | .UTCQMMA_SF_E3M2 => Row.mma .tcgen05 .e3m2 .e3m2 .f32 32768
  | .UTCQMMA_SF_E2M3 => Row.mma .tcgen05 .e2m3 .e2m3 .f32 32768
  | .UTCQMMA_SF_E2M1 => Row.mma .tcgen05 .e2m1 .e2m1 .f32 32768
  | .UTCOMMA_SF => Row.mma .tcgen05 .e2m1 .e2m1 .f32 32768
  | .UTCMMA_SP => Row.mma .tcgen05 .s8 .s8 .s32 32768
  | .UTCOMMA_2CTA => Row.mma .tcgen05 .e2m1 .e2m1 .f32 65536
  | .UTCMMA_2CTA => Row.mma .tcgen05 .s8 .s8 .s32 65536

/-- **The census form that prices a tcgen05.mma form** (RT-B200-EMP-2): a form the census never probed at M = 128,
N = 256 its kind's census form there (kind::f16's, kind::f8f6f4's for kind::f8f6f4 and kind::mxf8f6f4), else the
int8 form. It writes the same 32768 words, and by the sheet's formula (its MACs over a quarter of its kind's peak)
RT-B200-EMP-2 finds each at the same tensor line, 512 SMSP-cycles, and one issue (mxf4 m128n256k64: 2,097,152 MACs
over 16384 / 4; tf32 k8 too). That reads kind::f8f6f4 and kind::mxf8f6f4 at FP8's peak for every type (K = 32) and
only kind::mxf4 at FP4's (K = 64); no run here measures the FP6 and FP4 types under kind::f8f6f4. Every other form,
the sparse and CTA-pair forms included, is its own. -/
def Tc05.twin : Tc05 → Tc05
  | .UTCHMMA_BF16 => .UTCHMMA_F16_M128_N256
  | .UTCHMMA_TF32 => .UTCIMMA_M128_N256
  | .UTCQMMA_E5M2 | .UTCQMMA_E3M2 | .UTCQMMA_E2M3 | .UTCQMMA_E2M1 => .UTCQMMA_E4M3_M128_N256
  | .UTCQMMA_SF | .UTCQMMA_SF_E5M2 | .UTCQMMA_SF_E3M2 | .UTCQMMA_SF_E2M3 | .UTCQMMA_SF_E2M1 =>
    .UTCQMMA_E4M3_M128_N256
  | .UTCOMMA_SF => .UTCIMMA_M128_N256
  | t => t

/-- **sm_100a's tensor-memory instructions** (PTX ISA 9.4 §9.7.18.8, §9.7.18.9): the census's tcgen05.ld shapes
(LDTM), `LDTM` (every other shape), `STTM` (tcgen05.st), `UTCCP` (tcgen05.cp) and `UTCSHIFT` (tcgen05.shift). -/
inductive Tmem
  | LDTM_32X32B_X1 | LDTM_32X32B_X4 | LDTM_32X32B_X16 | LDTM_32X32B_X32 | LDTM_32X32B_X64 | LDTM_32X32B_X128
  | LDTM_16X64B_X16 | LDTM_16X128B_X8 | LDTM_16X256B_X4
  | LDTM | STTM | UTCCP | UTCSHIFT
  deriving DecidableEq, Repr

instance : Fintype Tmem :=
  Fintype.ofList (α := Tmem) [.LDTM_32X32B_X1, .LDTM_32X32B_X4, .LDTM_32X32B_X16, .LDTM_32X32B_X32, .LDTM_32X32B_X64,
    .LDTM_32X32B_X128, .LDTM_16X64B_X16, .LDTM_16X128B_X8, .LDTM_16X256B_X4, .LDTM, .STTM, .UTCCP, .UTCSHIFT]
    fun x => by
    cases x <;> decide +kernel

/-- **The tensor-memory rows**: a tcgen05.ld writes its shape's registers (32x32b.xN: N words a lane; 16x64b.xN,
16x128b.xN, 16x256b.xN: N, 2N, 4N words a lane; at most 128 a lane, 4096 a warp instruction), a tcgen05.st its shape's
tensor-memory words (at most 4096), a tcgen05.cp its shape's (128x256b: 1024), and a tcgen05.shift moves at most 1024;
none computes. -/
def Tmem.row : Tmem → Row
  | .LDTM_32X32B_X1 => Row.side .tmem .lane .move true 32
  | .LDTM_32X32B_X4 => Row.side .tmem .lane .move true 128
  | .LDTM_32X32B_X16 => Row.side .tmem .lane .move true 512
  | .LDTM_32X32B_X32 => Row.side .tmem .lane .move true 1024
  | .LDTM_32X32B_X64 => Row.side .tmem .lane .move true 2048
  | .LDTM_32X32B_X128 => Row.side .tmem .lane .move true 4096
  | .LDTM_16X64B_X16 => Row.side .tmem .lane .move true 512
  | .LDTM_16X128B_X8 => Row.side .tmem .lane .move true 512
  | .LDTM_16X256B_X4 => Row.side .tmem .lane .move true 512
  | .LDTM => Row.side .tmem .lane .move true 4096
  | .STTM => Row.side .tmem .lane .move true 4096
  | .UTCCP => Row.side .tmem .async .move true 1024
  | .UTCSHIFT => Row.side .tmem .async .move true 1024

/-- **sm_100a's instruction classes**: an mma.sync form, a tcgen05.mma form, a tensor-memory instruction, or a class
outside the tensor cores. -/
inductive Sm100
  | sync (s : Sync)
  | tc05 (t : Tc05)
  | tmem (m : Tmem)
  | base (c : Base)
  deriving DecidableEq, Fintype, Repr

/-- **sm_100a's rows**. -/
def Sm100.row : Sm100 → Row
  | .sync s => s.row
  | .tc05 t => t.row
  | .tmem m => m.row
  | .base c => c.row

end Accounting.Isa
