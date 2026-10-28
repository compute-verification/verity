import Definitions.Accounting.HeadlineDc.Encoding

/-!
# B200 (sm_100)'s headline instance: DS-100 and `dither-2s-licpw-redux-ldtm0-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm100SheetHeadline` reads it.

* `classes`: the 52 instruction classes of the DS-100 census, in the order `Fin 52` indexes them. `NOROW` is A2's
  catch-all: every warp instruction no other class names counts there, at its issue slot alone.
* `ds100`, `sheet`: the B200's data sheet (DS-100), `benchmarks/pouw/machine_model/data/devices/sm_100/sheet.toml` as
  `Security/Definitions/Accounting/Sheets/B200.lean` states it: f_max 1965 MHz (the driver's reported maximum (the data
  sheet gives no clock)); each tensor class's dense MACs a warp instruction (sm_100's tcgen05 MMA whole (`m·n·k`), a 2:4
  sparse form at its dense twin's, its mma.sync whole, DMMA's at the f64 peak) and its operand type's dense MACs an SM
  clock; the Nsight Compute pipes ALU, IMAD, SHFL and UNIFORM, each at its width (the Programming Guide's, and the
  uniform datapath's the census's measured UP line, which no document states); no shared-memory row (DS-100 prices
  none). An MMA class the sheet rates on no peak (b1) and the memory, atomic and control classes are their issue slot
  alone (A2). Its machine reproduces every DS-100 coefficient.
* `d2s`, `model`: `dither-2s-licpw-redux-ldtm0-l32` as a TS model on the sheet's machine: writer kinds every MMA class
  of the census (ω its destination's 32-bit words a warp instruction, A6, sheet.toml's `omega`: an f64 element two) and
  the uniform `UNI` (2, a UR pair); the vector classes CUDA's lane classes and both readings' catch-alls (NOROW,
  UNLISTED: A6 counts a vector word's instructions in them, and an instruction no class names, IMAD.WIDE among them,
  counts in the catch-all); the fold routes the design's fold and its 12 other bit-exact programs (cp-seg4, cp-seg4u,
  cp-ats, cp-ats2, cp-ats8, cp-red, cp-red2, cp-red8, cp-red-generic, cp-red2-generic, cp-red8-generic, cp-xt: the
  checkpoint's bit-exact loops) on lanes, and alone on the uniform datapath at the least of their counts; the table
  level's bytes counted in ALU, which `b_T = 0` prices at nothing (the census has no load class); A6's floors `q = q_T =
  ½`, `b_T = 0`; 32768 committed words a unit of honest work (an UTCIMMA_M128_N256 at L = 32).
* `flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold (the fold
  routes' cheapest mix, by share of the words: cp-red2 65/72, cp-red8 7/72), no filler, and ε_skip's skippable work at
  FFMA's price, 0.698060% of ♭'s bound (`eps_budget.design_skip` on DS-100).
* `lamStar`: π*, the certificate minimising the bound over this menu (DS_issue 0.666667 + DS_alu 0.333333).
* `checks`: the checks a window, one a round (Theorem S's S).
-/

namespace Accounting.TS.Headline.Sm100

open Accounting.TS

/-- The DS-100 census's instruction classes, in index order. -/
def classes : List String := [
  "ALU", "ATOME32", "ATOMS32", "DMMA16816", "DMMA1684", "DMMA1688", "DMMA884", "FFMA", "IMAD", "IMMA16816_S8",
  "IMMA16832_S8", "LDS128", "LDTM_16X128B_X8", "LDTM_16X256B_X4", "LDTM_16X64B_X16", "LDTM_32X32B_X1",
  "LDTM_32X32B_X128", "LDTM_32X32B_X16", "LDTM_32X32B_X32", "LDTM_32X32B_X4", "LDTM_32X32B_X64", "NOROW", "REDG32",
  "REDUX", "SHFL", "STG128", "STG32", "STS128", "STS32", "UNI", "UNLISTED", "UTCHMMA_BF16_M128_N256",
  "UTCHMMA_F16_M128_N256", "UTCHMMA_TF32_M128_N256", "UTCIMMA_M128_N128", "UTCIMMA_M128_N256", "UTCIMMA_M128_N256_SP",
  "UTCIMMA_M128_N32", "UTCIMMA_M128_N64", "UTCIMMA_M64_N128", "UTCIMMA_M64_N256", "UTCIMMA_M64_N64",
  "UTCOMMA_MXF4NVF4_M128_N256", "UTCOMMA_MXF4_M128_N256", "UTCQMMA_E2M1_M128_N256", "UTCQMMA_E2M3_M128_N256",
  "UTCQMMA_E3M2_M128_N256", "UTCQMMA_E4M3_M128_N128", "UTCQMMA_E4M3_M128_N256", "UTCQMMA_E5M2_M128_N256",
  "UTCQMMA_MXF8F6F4_E4M3_M128_N256", "VIADD"]

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
  | alu
  | imad
  | shfl
  | uniform
  deriving DecidableEq

instance instFintypePipe : Fintype Pipe where
  elems := {Pipe.alu, Pipe.imad, Pipe.shfl, Pipe.uniform}
  complete x := by cases x <;> simp

/-- Each pipe's results an SM clock: the Programming Guide's, the uniform datapath's the census's UP line. -/
def Pipe.width : Pipe → ℚ
  | .alu => 64
  | .imad => 64
  | .shfl => 32
  | .uniform => 256000000 / 3914583

/-- The B200's data sheet, by class index. -/
def ds100 : SheetDataQ Pipe where
  fmax := 1965
  macs := [
    0, 0, 0, 2048, 512, 1024, 256, 0, 0, 2048, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 524288,
    524288, 262144, 524288, 1048576, 1048576, 131072, 262144, 262144, 524288, 131072, 2097152, 2097152, 1048576,
    1048576, 1048576, 524288, 1048576, 1048576, 1048576, 0]
  peak := [
    0, 0, 0, 64, 64, 64, 64, 0, 0, 8192, 8192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4096, 4096,
    2048, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 16384, 16384, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 0]
  pipe := [
    some .alu, none, none, none, none, none, none, none, some .imad, none, none, none, none, none, none, none, none,
    none, none, none, none, none, none, none, some .shfl, none, none, none, none, some .uniform, none, none, none, none,
    none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none]
  width := Pipe.width
  smem := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin 52) Pipe := ds100.sheet 52

/-- `dither-2s-licpw-redux-ldtm0-l32`'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
    (3, 256),  -- DMMA16816
    (4, 256),  -- DMMA1684
    (5, 256),  -- DMMA1688
    (6, 128),  -- DMMA884
    (9, 128),  -- IMMA16816_S8
    (10, 128),  -- IMMA16832_S8
    (31, 32768),  -- UTCHMMA_BF16_M128_N256
    (32, 32768),  -- UTCHMMA_F16_M128_N256
    (33, 32768),  -- UTCHMMA_TF32_M128_N256
    (34, 16384),  -- UTCIMMA_M128_N128
    (35, 32768),  -- UTCIMMA_M128_N256
    (36, 32768),  -- UTCIMMA_M128_N256_SP
    (37, 4096),  -- UTCIMMA_M128_N32
    (38, 8192),  -- UTCIMMA_M128_N64
    (39, 8192),  -- UTCIMMA_M64_N128
    (40, 16384),  -- UTCIMMA_M64_N256
    (41, 4096),  -- UTCIMMA_M64_N64
    (42, 32768),  -- UTCOMMA_MXF4NVF4_M128_N256
    (43, 32768),  -- UTCOMMA_MXF4_M128_N256
    (44, 32768),  -- UTCQMMA_E2M1_M128_N256
    (45, 32768),  -- UTCQMMA_E2M3_M128_N256
    (46, 32768),  -- UTCQMMA_E3M2_M128_N256
    (47, 16384),  -- UTCQMMA_E4M3_M128_N128
    (48, 32768),  -- UTCQMMA_E4M3_M128_N256
    (49, 32768),  -- UTCQMMA_E5M2_M128_N256
    (50, 32768),  -- UTCQMMA_MXF8F6F4_E4M3_M128_N256
    (29, 2)  -- UNI
  ]
  vec := [7, 8, 0, 24, 21, 30]  -- FFMA, IMAD, ALU, SHFL, NOROW, UNLISTED
  folds := [
    [(0, 17, 1024), (8, 1, 64), (24, 3, 4096)],
    [(0, 17, 1024), (8, 65, 4096), (23, 1, 4096), (25, 1, 16384)],
    [(0, 67, 4096), (8, 1, 64), (23, 1, 2048), (25, 1, 16384), (29, 1, 4096)],
    [(0, 17, 1024), (2, 1, 4096), (8, 1, 64), (11, 1, 16384), (25, 1, 16384), (27, 1, 16384)],
    [(0, 67, 4096), (2, 1, 2048), (8, 1, 64), (11, 1, 16384), (25, 1, 16384), (27, 1, 16384)],
    [(0, 1, 64), (2, 1, 512), (8, 1, 64), (11, 1, 16384), (25, 1, 16384), (27, 1, 16384)],
    [(0, 17, 1024), (8, 1, 64), (22, 1, 4096)],
    [(0, 67, 4096), (8, 1, 64), (22, 1, 2048)],
    [(0, 1, 64), (8, 1, 64), (22, 1, 512)],
    [(0, 17, 1024), (1, 1, 4096), (8, 1, 64)],
    [(0, 67, 4096), (1, 1, 2048), (8, 1, 64)],
    [(0, 1, 64), (1, 1, 512), (8, 1, 64)],
    [(0, 137, 8192), (8, 1, 64), (11, 1, 16384), (26, 1, 131072), (28, 1, 4096)],
    [(29, 133, 128)]]
  bytes := 0  -- ALU
  q := 1 / 2
  qT := 1 / 2
  Nw := 32768

/-- ♭'s filler a unit: none. -/
def flatFill : Sparse := []

/-- `dither-2s-licpw-redux-ldtm0-l32`'s TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin 52) (SheetRow Pipe) (Fin 27) Unit Empty (Fin 14) Unit :=
  d2s.model 52 27 14 sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := [(35, 1, 1)]

/-- ♭'s fold a unit: the lane fold for 32768 words, the fold routes' cheapest mix, by share of the words: cp-red2 65/72,
cp-red8 7/72. -/
def flatFold : Sparse := [(0, 1601, 3), (8, 512, 1), (22, 62, 3)]

/-- ♭'s skippable work a unit, at FFMA's price. -/
def flatSkip : Sparse := [(7, 745063066358, 99301939663)]

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin 52) := labelled 52 flatWords flatFold flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : SheetRow Pipe → ℝ
  | .issue => 666666667 / 1000000000
  | .pipe .alu => 333333333 / 1000000000
  | _ => 0

/-- The checks a window. -/
def checks : ℕ := 48000

end Accounting.TS.Headline.Sm100
