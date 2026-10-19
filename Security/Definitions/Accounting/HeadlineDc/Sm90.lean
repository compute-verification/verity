import Definitions.Accounting.HeadlineDc.Encoding

/-!
# H200 (sm_90)'s headline instance: DS-90 and `dither-2s-licpw-redux-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm90SheetHeadline` reads it.

* `classes`: the 28 instruction classes of the DS-90 census, in the order `Fin 28` indexes them.
* `ds90`, `sheet`: the H200's data sheet (DS-90): f_max 1980 MHz (the data sheet's boost clock); each tensor class's
  dense MACs a warp instruction (sm_90's wgmma as one warp's m16 slice of its m64 warpgroup instruction (`16·n·k`), its
  mma.sync as m16n8) and its operand type's dense MACs an SM clock; the Nsight Compute pipes ALU, IMAD and SHFL with
  their Programming Guide widths; no shared-memory row (DS-90 prices none). Its machine reproduces every DS-90
  coefficient.
* `d2s`, `model`: `dither-2s-licpw-redux-l32` as a TS model on the sheet's machine: writer kinds every census MMA class
  (ω = m·n, its destination's 32-bit words a warp) and the uniform `UNI` (1); the vector classes CUDA's lane classes;
  the design's fold on lanes and alone on the uniform datapath; the table level's bytes counted in ALU, which `b_T = 0`
  prices at nothing (the census has no load class); A6's floors `q = q_T = ½`, `b_T = 0`; 4096 committed words a unit of
  honest work (an IGMMA_S8_N256 at L = 32); its filler menu the one route ffill runs on (sm_90's uniform datapath has no
  FFMA).
* `flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold, the filler
  (ffill on lanes, `flatFill`), and ε_skip's skippable work at FFMA's price, 0.642654% of ♭'s bound
  (`eps_budget.design_skip` on DS-90).
* `lamStar`: π*, the certificate minimising the bound over this menu (DS_issue 1.000000).
* `checks`: the checks a window, one a round (Theorem S's S).
-/

namespace Accounting.TS.Headline.Sm90

open Accounting.TS

/-- The DS-90 census's instruction classes, in index order. -/
def classes : List String := [
  "ALU", "BGMMA_B1_N256", "BMMA168256_B1", "FFMA", "HGMMA_BF16_N128", "HGMMA_BF16_N256", "HGMMA_F16_N256",
  "HGMMA_TF32_N256", "HMMA16816_BF16", "HMMA16816_F16", "HMMA1688_TF32", "IGMMA_S8_N128", "IGMMA_S8_N16",
  "IGMMA_S8_N192", "IGMMA_S8_N256", "IGMMA_S8_N256_RS", "IGMMA_S8_N32", "IGMMA_S8_N64", "IGMMA_S8_N8",
  "IGMMA_U8S8_N256", "IMAD", "IMMA16816_S8", "IMMA16832_S8", "QGMMA_E4M3_N128", "QGMMA_E4M3_N256", "QGMMA_E5M2_N256",
  "SHFL", "UNI"]

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
    0, 0, 0, 0, 32768, 65536, 65536, 32768, 2048, 2048, 1024, 65536, 8192, 98304, 131072, 131072, 16384, 32768, 4096,
    131072, 0, 2048, 4096, 65536, 131072, 131072, 0, 0]
  peak := [
    0, 0, 0, 0, 2048, 2048, 2048, 1024, 2048, 2048, 1024, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 0, 4096,
    4096, 4096, 4096, 4096, 0, 0]
  pipe := [
    some .alu, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, some .imad, none, none, none, none, none, some .shfl, none]
  width := Pipe.width
  smem := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin 28) Pipe := ds90.sheet 28

/-- `dither-2s-licpw-redux-l32`'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
    (1, 4096),  -- BGMMA_B1_N256
    (2, 128),  -- BMMA168256_B1
    (4, 2048),  -- HGMMA_BF16_N128
    (5, 4096),  -- HGMMA_BF16_N256
    (6, 4096),  -- HGMMA_F16_N256
    (7, 4096),  -- HGMMA_TF32_N256
    (8, 128),  -- HMMA16816_BF16
    (9, 128),  -- HMMA16816_F16
    (10, 128),  -- HMMA1688_TF32
    (11, 2048),  -- IGMMA_S8_N128
    (12, 256),  -- IGMMA_S8_N16
    (13, 3072),  -- IGMMA_S8_N192
    (14, 4096),  -- IGMMA_S8_N256
    (15, 4096),  -- IGMMA_S8_N256_RS
    (16, 512),  -- IGMMA_S8_N32
    (17, 1024),  -- IGMMA_S8_N64
    (18, 128),  -- IGMMA_S8_N8
    (19, 4096),  -- IGMMA_U8S8_N256
    (21, 128),  -- IMMA16816_S8
    (22, 128),  -- IMMA16832_S8
    (23, 2048),  -- QGMMA_E4M3_N128
    (24, 4096),  -- QGMMA_E4M3_N256
    (25, 4096),  -- QGMMA_E5M2_N256
    (27, 1)  -- UNI
  ]
  vec := [3, 20, 0, 26]  -- FFMA, IMAD, ALU, SHFL
  folds := [
    [(0, 17, 1024), (20, 1, 64), (26, 3, 4096)],
    [(27, 135, 128)]]
  bytes := 0  -- ALU
  q := 1 / 2
  qT := 1 / 2
  Nw := 4096

/-- ♭'s filler a unit: ffill on lanes. -/
def flatFill : Sparse := [(3, 6, 1), (20, 3, 1)]

/-- `dither-2s-licpw-redux-l32`'s TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin 28) (SheetRow Pipe) (Fin 24) Unit Empty (Fin 2) Unit :=
  d2s.modelFill 28 24 2 sheet.machine flatFill

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := [(14, 1, 1)]

/-- ♭'s fold a unit: the lane fold for 4096 words. -/
def flatFold : Sparse := [(0, 68, 1), (20, 64, 1), (26, 3, 1)]

/-- ♭'s skippable work a unit, at FFMA's price. -/
def flatSkip : Sparse := [(3, 1863696948824044715, 1987146917594316933)]

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin 28) := labelled 28 flatWords flatFold flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : SheetRow Pipe → ℝ
  | .issue => 1 / 1
  | _ => 0

/-- The checks a window. -/
def checks : ℕ := 48000

end Accounting.TS.Headline.Sm90
