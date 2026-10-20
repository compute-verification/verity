import Definitions.Accounting.HeadlineDc.Encoding

/-!
# B200 (sm_100)'s headline instance: DS-100 and `dither-2s-licpw-redux-ldtm0-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm100SheetHeadline` reads it.

* `classes`: the 37 instruction classes of the DS-100 census, in the order `Fin 37` indexes them. `NOROW` is A2's
  catch-all: every warp instruction no other class names counts there, at its issue slot alone.
* `ds100`, `sheet`: the B200's data sheet (DS-100): f_max 1965 MHz (the driver's reported maximum (the data sheet gives
  no clock)); each tensor class's dense MACs a warp instruction (sm_100's tcgen05 MMA whole (`m·n·k`), its mma.sync as
  m16n8) and its operand type's dense MACs an SM clock; the Nsight Compute pipes ALU, IMAD and SHFL with their
  Programming Guide widths; no shared-memory row (DS-100 prices none). Its machine reproduces every DS-100 coefficient.
* `d2s`, `model`: `dither-2s-licpw-redux-ldtm0-l32` as a TS model on the sheet's machine: writer kinds every census MMA
  class (ω = m·n, its destination's 32-bit words a warp) and the uniform `UNI` (2, a UR pair: RT-HOPPER-7); the vector
  classes CUDA's lane classes; the fold routes the design's fold and its 12 other bit-exact programs (cp-seg4, cp-seg4u,
  cp-ats, cp-ats2, cp-ats8, cp-red, cp-red2, cp-red8, cp-red-generic, cp-red2-generic, cp-red8-generic, cp-xt: the
  checkpoint's bit-exact loops, RT2-HOPPER-9, -11, RT3-DC-1, -4 and RT4-SHEET-7) on lanes, and alone on the uniform
  datapath at the least of their counts; the table level's bytes counted in ALU, which `b_T = 0` prices at nothing (the
  census has no load class); A6's floors `q = q_T = ½`, `b_T = 0`; 32768 committed words a unit of honest work (an
  UTCIMMA_M128_N256 at L = 32).
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
  "ALU", "ATOME32", "ATOMS32", "FFMA", "IMAD", "IMMA16816_S8", "IMMA16832_S8", "LDS128", "LDTM_16X128B_X8",
  "LDTM_16X256B_X4", "LDTM_16X64B_X16", "LDTM_32X32B_X1", "LDTM_32X32B_X128", "LDTM_32X32B_X16", "LDTM_32X32B_X32",
  "LDTM_32X32B_X4", "LDTM_32X32B_X64", "NOROW", "REDG32", "REDUX", "SHFL", "STG128", "STG32", "STS128", "STS32", "UNI",
  "UTCHMMA_F16_M128_N256", "UTCIMMA_M128_N128", "UTCIMMA_M128_N256", "UTCIMMA_M128_N32", "UTCIMMA_M128_N64",
  "UTCIMMA_M64_N128", "UTCIMMA_M64_N256", "UTCIMMA_M64_N64", "UTCQMMA_E4M3_M128_N128", "UTCQMMA_E4M3_M128_N256",
  "VIADD"]

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
  | alu
  | imad
  | shfl
  deriving DecidableEq

instance instFintypePipe : Fintype Pipe where
  elems := {Pipe.alu, Pipe.imad, Pipe.shfl}
  complete x := by cases x <;> simp

/-- Each pipe's Programming Guide results an SM clock. -/
def Pipe.width : Pipe → ℕ
  | .alu => 64
  | .imad => 64
  | .shfl => 32

/-- The B200's data sheet, by class index. -/
def ds100 : SheetData Pipe where
  fmax := 1965
  macs := [
    0, 0, 0, 0, 0, 2048, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 524288, 524288, 1048576, 131072,
    262144, 262144, 524288, 131072, 524288, 1048576, 0]
  peak := [
    0, 0, 0, 0, 0, 8192, 8192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4096, 8192, 8192, 8192, 8192,
    8192, 8192, 8192, 8192, 8192, 0]
  pipe := [
    some .alu, none, none, none, some .imad, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, some .shfl, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none]
  width := Pipe.width
  smem := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin 37) Pipe := ds100.sheet 37

/-- `dither-2s-licpw-redux-ldtm0-l32`'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
    (5, 128),  -- IMMA16816_S8
    (6, 128),  -- IMMA16832_S8
    (26, 32768),  -- UTCHMMA_F16_M128_N256
    (27, 16384),  -- UTCIMMA_M128_N128
    (28, 32768),  -- UTCIMMA_M128_N256
    (29, 4096),  -- UTCIMMA_M128_N32
    (30, 8192),  -- UTCIMMA_M128_N64
    (31, 8192),  -- UTCIMMA_M64_N128
    (32, 16384),  -- UTCIMMA_M64_N256
    (33, 4096),  -- UTCIMMA_M64_N64
    (34, 16384),  -- UTCQMMA_E4M3_M128_N128
    (35, 32768),  -- UTCQMMA_E4M3_M128_N256
    (25, 2)  -- UNI
  ]
  vec := [3, 4, 0, 20]  -- FFMA, IMAD, ALU, SHFL
  folds := [
    [(0, 17, 1024), (4, 1, 64), (20, 3, 4096)],
    [(0, 17, 1024), (4, 65, 4096), (19, 1, 4096), (21, 1, 16384)],
    [(0, 67, 4096), (4, 1, 64), (19, 1, 2048), (21, 1, 16384), (25, 1, 4096)],
    [(0, 17, 1024), (2, 1, 4096), (4, 1, 64), (7, 1, 16384), (21, 1, 16384), (23, 1, 16384)],
    [(0, 67, 4096), (2, 1, 2048), (4, 1, 64), (7, 1, 16384), (21, 1, 16384), (23, 1, 16384)],
    [(0, 1, 64), (2, 1, 512), (4, 1, 64), (7, 1, 16384), (21, 1, 16384), (23, 1, 16384)],
    [(0, 17, 1024), (4, 1, 64), (18, 1, 4096)],
    [(0, 67, 4096), (4, 1, 64), (18, 1, 2048)],
    [(0, 1, 64), (4, 1, 64), (18, 1, 512)],
    [(0, 17, 1024), (1, 1, 4096), (4, 1, 64)],
    [(0, 67, 4096), (1, 1, 2048), (4, 1, 64)],
    [(0, 1, 64), (1, 1, 512), (4, 1, 64)],
    [(0, 137, 8192), (4, 1, 64), (7, 1, 16384), (22, 1, 131072), (24, 1, 4096)],
    [(25, 133, 128)]]
  bytes := 0  -- ALU
  q := 1 / 2
  qT := 1 / 2
  Nw := 32768

/-- ♭'s filler a unit: none. -/
def flatFill : Sparse := []

/-- `dither-2s-licpw-redux-ldtm0-l32`'s TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin 37) (SheetRow Pipe) (Fin 13) Unit Empty (Fin 14) Unit :=
  d2s.model 37 13 14 sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := [(28, 1, 1)]

/-- ♭'s fold a unit: the lane fold for 32768 words, the fold routes' cheapest mix, by share of the words: cp-red2 65/72,
cp-red8 7/72. -/
def flatFold : Sparse := [(0, 1601, 3), (4, 512, 1), (18, 62, 3)]

/-- ♭'s skippable work a unit, at FFMA's price. -/
def flatSkip : Sparse := [(3, 745063066358, 99301939663)]

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin 37) := labelled 37 flatWords flatFold flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : SheetRow Pipe → ℝ
  | .issue => 666666667 / 1000000000
  | .pipe .alu => 333333333 / 1000000000
  | _ => 0

/-- The checks a window. -/
def checks : ℕ := 48000

end Accounting.TS.Headline.Sm100
