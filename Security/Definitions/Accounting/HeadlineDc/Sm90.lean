import Definitions.Accounting.HeadlineDc.Encoding

/-!
# H200 (sm_90)'s headline instance: DS-90 and `dither-2s-licpw-redux-p1-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm90SheetHeadline` reads it.

* `classes`: the 39 instruction classes of the DS-90 census, in the order `Fin 39` indexes them. `NOROW` is A2's
  catch-all: every warp instruction no other class names counts there, at its issue slot alone.
* `ds90`, `sheet`: the H200's data sheet (DS-90): f_max 1980 MHz (EPS1 DC's measured f_max at default clocks; the
  sheet's TOPS are at 1830 MHz); each tensor class's dense MACs a warp instruction (sm_90's wgmma as one warp's m16
  slice of its m64 warpgroup instruction (`16·n·k`), its mma.sync as m16n8) and its operand type's dense MACs an SM
  clock; the Nsight Compute pipes ALU, IMAD and SHFL with their Programming Guide widths; no shared-memory row (DS-90
  prices none). Its machine reproduces every DS-90 coefficient.
* `d2s`, `model`: `dither-2s-licpw-redux-p1-l32` as a TS model on the sheet's machine: writer kinds every census MMA
  class (ω = m·n, its destination's 32-bit words a warp) and the uniform `UNI` (2, a UR pair: RT-HOPPER-7); the vector
  classes CUDA's lane classes; the fold routes the design's fold and its 12 other bit-exact programs (cp-seg4, cp-seg4u,
  cp-ats, cp-ats2, cp-ats8, cp-red, cp-red2, cp-red8, cp-red-generic, cp-red2-generic, cp-red8-generic, cp-xt: the
  checkpoint's bit-exact loops, RT2-HOPPER-9, -11, RT3-DC-1, -4 and RT4-SHEET-7) on lanes, and alone on the uniform
  datapath at the least of their counts; the table level's bytes counted in ALU, which `b_T = 0` prices at nothing (the
  census has no load class); A6's floors `q = q_T = ½`, `b_T = 0`; 4096 committed words a unit of honest work (an
  IGMMA_S8_N256 at L = 32).
* `flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold (the fold
  routes' cheapest, cp-red2 on every word), no filler, and ε_skip's skippable work at FFMA's price, 0.858332% of ♭'s
  bound (`eps_budget.design_skip` on DS-90).
* `lamStar`: π*, the certificate minimising the bound over this menu (DS_issue 1.000000).
* `checks`: the checks a window, one a round (Theorem S's S).
-/

namespace Accounting.TS.Headline.Sm90

open Accounting.TS

/-- The DS-90 census's instruction classes, in index order. -/
def classes : List String := [
  "ALU", "ATOME32", "ATOMS32", "BGMMA_B1_N256", "BMMA168256_B1", "FFMA", "HGMMA_BF16_N128", "HGMMA_BF16_N256",
  "HGMMA_F16_N256", "HGMMA_TF32_N256", "HMMA16816_BF16", "HMMA16816_F16", "HMMA1688_TF32", "IGMMA_S8_N128",
  "IGMMA_S8_N16", "IGMMA_S8_N192", "IGMMA_S8_N256", "IGMMA_S8_N256_RS", "IGMMA_S8_N32", "IGMMA_S8_N64", "IGMMA_S8_N8",
  "IGMMA_U8S8_N256", "IMAD", "IMMA16816_S8", "IMMA16832_S8", "ISETP", "LDS128", "NOROW", "QGMMA_E4M3_N128",
  "QGMMA_E4M3_N256", "QGMMA_E5M2_N256", "REDG32", "REDUX", "SHFL", "STG128", "STG32", "STS128", "STS32", "UNI"]

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

/-- The H200's data sheet, by class index. -/
def ds90 : SheetData Pipe where
  fmax := 1980
  macs := [
    0, 0, 0, 0, 0, 0, 32768, 65536, 65536, 32768, 2048, 2048, 1024, 65536, 8192, 98304, 131072, 131072, 16384, 32768,
    4096, 131072, 0, 2048, 4096, 0, 0, 0, 65536, 131072, 131072, 0, 0, 0, 0, 0, 0, 0, 0]
  peak := [
    0, 0, 0, 0, 0, 0, 2048, 2048, 2048, 1024, 2048, 2048, 1024, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 0,
    4096, 4096, 0, 0, 0, 4096, 4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0]
  pipe := [
    some .alu, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, some .imad, none, none, none, none, none, none, none, none, none, none, some .shfl, none,
    none, none, none, none]
  width := Pipe.width
  smem := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin 39) Pipe := ds90.sheet 39

/-- `dither-2s-licpw-redux-p1-l32`'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
    (3, 4096),  -- BGMMA_B1_N256
    (4, 128),  -- BMMA168256_B1
    (6, 2048),  -- HGMMA_BF16_N128
    (7, 4096),  -- HGMMA_BF16_N256
    (8, 4096),  -- HGMMA_F16_N256
    (9, 4096),  -- HGMMA_TF32_N256
    (10, 128),  -- HMMA16816_BF16
    (11, 128),  -- HMMA16816_F16
    (12, 128),  -- HMMA1688_TF32
    (13, 2048),  -- IGMMA_S8_N128
    (14, 256),  -- IGMMA_S8_N16
    (15, 3072),  -- IGMMA_S8_N192
    (16, 4096),  -- IGMMA_S8_N256
    (17, 4096),  -- IGMMA_S8_N256_RS
    (18, 512),  -- IGMMA_S8_N32
    (19, 1024),  -- IGMMA_S8_N64
    (20, 128),  -- IGMMA_S8_N8
    (21, 4096),  -- IGMMA_U8S8_N256
    (23, 128),  -- IMMA16816_S8
    (24, 128),  -- IMMA16832_S8
    (28, 2048),  -- QGMMA_E4M3_N128
    (29, 4096),  -- QGMMA_E4M3_N256
    (30, 4096),  -- QGMMA_E5M2_N256
    (38, 2)  -- UNI
  ]
  vec := [5, 22, 0, 33]  -- FFMA, IMAD, ALU, SHFL
  folds := [
    [(0, 17, 1024), (22, 1, 64), (33, 3, 4096)],
    [(0, 17, 1024), (22, 65, 4096), (32, 1, 4096), (34, 1, 16384)],
    [(0, 67, 4096), (22, 1, 64), (32, 1, 2048), (34, 1, 16384), (38, 1, 4096)],
    [(0, 17, 1024), (2, 1, 4096), (22, 1, 64), (26, 1, 16384), (34, 1, 16384), (36, 1, 16384)],
    [(0, 67, 4096), (2, 1, 2048), (22, 1, 64), (26, 1, 16384), (34, 1, 16384), (36, 1, 16384)],
    [(0, 1, 64), (2, 1, 512), (22, 1, 64), (26, 1, 16384), (34, 1, 16384), (36, 1, 16384)],
    [(0, 17, 1024), (22, 1, 64), (31, 1, 4096)],
    [(0, 67, 4096), (22, 1, 64), (31, 1, 2048)],
    [(0, 1, 64), (22, 1, 64), (31, 1, 512)],
    [(0, 17, 1024), (1, 1, 4096), (22, 1, 64)],
    [(0, 67, 4096), (1, 1, 2048), (22, 1, 64)],
    [(0, 1, 64), (1, 1, 512), (22, 1, 64)],
    [(0, 137, 8192), (22, 1, 64), (26, 1, 16384), (35, 1, 131072), (37, 1, 4096)],
    [(38, 133, 128)]]
  bytes := 0  -- ALU
  q := 1 / 2
  qT := 1 / 2
  Nw := 4096

/-- ♭'s filler a unit: none. -/
def flatFill : Sparse := []

/-- `dither-2s-licpw-redux-p1-l32`'s TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin 39) (SheetRow Pipe) (Fin 24) Unit Empty (Fin 14) Unit :=
  d2s.model 39 24 14 sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := [(16, 1, 1)]

/-- ♭'s fold a unit: the lane fold for 4096 words, the fold routes' cheapest, cp-red2 on every word. -/
def flatFold : Sparse := [(0, 67, 1), (22, 64, 1), (31, 2, 1)]

/-- ♭'s skippable work a unit, at FFMA's price. -/
def flatSkip : Sparse := [(5, 17166648254, 14797263819)]

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin 39) := labelled 39 flatWords flatFold flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : SheetRow Pipe → ℝ
  | .issue => 1 / 1
  | _ => 0

/-- The checks a window. -/
def checks : ℕ := 48000

end Accounting.TS.Headline.Sm90
