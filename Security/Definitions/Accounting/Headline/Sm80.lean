import Definitions.Accounting.Headline.Encoding

/-!
# sm_80's headline instance: DS-80 and Dither-2S at L = 16 (trusted; generated)

Written by `benchmarks/pouw/lean_headline.py` from the machine model's data (`benchmarks/pouw/machine_model/data`), and
checked against it by `benchmarks/pouw/tests/test_lean_headline.py`: edit the data and rerun the exporter, never this
file. `Accounting.Guarantees.TS.Sm80SheetHeadline` reads it.

* `classes`: the 35 instruction classes of the DS-80 census, in the order `Fin 35` indexes them.
* `ds80`, `sheet`: the A100's data sheet (DS-80): f_max 1410 MHz (the data sheet's boost clock); each tensor class's
  dense MACs a warp instruction (`m·n·k`, a 2:4 sparse form's dense equivalent) and its operand type's dense MACs an SM
  clock; the Nsight Compute pipes ALU, FMA, XU and SHFL with their Programming Guide widths; 128 B for every
  shared-memory class. Its machine reproduces every DS-80 coefficient.
* `d2s`, `model`: Dither-2S at L = 16 (`dither-2s-l16`) as a TS model on the sheet's machine: writer kinds the census's
  MMA classes (ω = m·n, f64's two words an element); the vector classes CUDA's lane classes; duo's fold on lanes (ALU 2
  + IMAD 1 lane instructions a word); A6's floors `q = q_T = ½`, `b_T = 0`; 128 committed words a unit of honest work
  (an IMMA.16816 at L = 16).
* `flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold, no filler, and
  ε_skip's skippable work at ALU's price, 0.146313% of ♭'s bound (`eps_budget.design_skip` on DS-80).
* `piStar`: π*, the facet (DS_alu) the machine model's LP certificate sits on at q* = ½.
* `checks`: the checks a window, one a round (Theorem S's S).
-/

namespace Accounting.TS.Headline.Sm80

open Accounting.TS

/-- The DS-80 census's instruction classes, in index order. -/
def classes : List String := [
  "ALU", "BMMA168128", "BMMA168256", "BMMA88128", "DMMA884", "FFMA", "HMMA16816", "HMMA16816F16", "HMMA1684T",
  "HMMA1688", "HMMA1688F16", "HMMA1688T", "HMMASP16816", "HMMASP16816T", "HMMASP16832", "IDP4A", "IMAD", "IMMA16816",
  "IMMA16832", "IMMA16832S4", "IMMA16864S4", "IMMA8816", "IMMA8832S4", "IMMASP168128S4", "IMMASP16832", "IMMASP16864",
  "IMMASP16864S4", "LDS128", "LDS32", "LDSM_X1", "LDSM_X2", "LDSM_X4", "NOROW", "SHFL", "XU"]

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
  | alu
  | fma
  | xu
  | shfl
  deriving DecidableEq

instance : Fintype Pipe where
  elems := {Pipe.alu, Pipe.fma, Pipe.xu, Pipe.shfl}
  complete x := by cases x <;> simp

/-- Each pipe's Programming Guide results an SM clock. -/
def Pipe.width : Pipe → ℕ
  | .alu => 64
  | .fma => 64
  | .xu => 16
  | .shfl => 32

/-- The A100's data sheet, by class index. -/
def ds80 : SheetData Pipe where
  fmax := 1410
  macs := [
    0, 16384, 32768, 8192, 256, 0, 2048, 2048, 512, 1024, 1024, 1024, 1024, 1024, 2048, 0, 0, 2048, 4096, 4096, 8192,
    1024, 2048, 8192, 2048, 4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0]
  peak := [
    0, 16384, 16384, 16384, 64, 0, 1024, 1024, 512, 1024, 1024, 512, 1024, 512, 1024, 0, 0, 2048, 2048, 4096, 4096,
    2048, 4096, 4096, 2048, 2048, 4096, 0, 0, 0, 0, 0, 0, 0, 0]
  pipe := [
    some .alu, none, none, none, none, some .fma, none, none, none, none, none, none, none, none, none, some .fma,
    some .fma, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    some .shfl, some .xu]
  width := Pipe.width
  smem := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 0, 0, 0]

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin 35) Pipe := ds80.sheet 35

/-- Dither-2S at L = 16's design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
    (1, 128),  -- BMMA168128
    (2, 128),  -- BMMA168256
    (3, 64),  -- BMMA88128
    (4, 128),  -- DMMA884
    (6, 128),  -- HMMA16816
    (7, 128),  -- HMMA16816F16
    (8, 128),  -- HMMA1684T
    (9, 128),  -- HMMA1688
    (10, 128),  -- HMMA1688F16
    (11, 128),  -- HMMA1688T
    (12, 128),  -- HMMASP16816
    (13, 128),  -- HMMASP16816T
    (14, 128),  -- HMMASP16832
    (17, 128),  -- IMMA16816
    (18, 128),  -- IMMA16832
    (19, 128),  -- IMMA16832S4
    (20, 128),  -- IMMA16864S4
    (21, 64),  -- IMMA8816
    (22, 64),  -- IMMA8832S4
    (23, 128),  -- IMMASP168128S4
    (24, 128),  -- IMMASP16832
    (25, 128),  -- IMMASP16864
    (26, 128)  -- IMMASP16864S4
  ]
  vec := [5, 16, 15, 0, 34, 33]  -- FFMA, IMAD, IDP4A, ALU, XU, SHFL
  folds := [
    [(0, 1, 16), (16, 1, 32)]]
  bytes := 28  -- LDS32
  q := 1 / 2
  qT := 1 / 2
  Nw := 128

/-- Dither-2S at L = 16's TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin 35) (SheetRow Pipe) (Fin 23) Unit Empty (Fin 1) Unit :=
  d2s.model 35 23 1 sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := [(17, 1, 1), (31, 48611, 500000)]

/-- ♭'s fold a unit: the lane fold for 128 words. -/
def flatFold : Sparse := [(0, 8, 1), (16, 4, 1)]

/-- ♭'s skippable work a unit, at ALU's price. -/
def flatSkip : Sparse := [(0, 23410046053071992, 1997073744243366001)]

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin 35) := labelled 35 flatWords flatFold [] flatSkip

/-- π*: the machine model's certificate at q* = ½. -/
def piStar : SheetRow Pipe := .pipe .alu

/-- The checks a window. -/
def checks : ℕ := 48000

end Accounting.TS.Headline.Sm80
