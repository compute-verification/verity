import Definitions.Accounting.Headline.Encoding

/-!
# sm_120's headline instance: DS-120 and H1 + fx1 (trusted; generated)

Written by `benchmarks/pouw/lean_headline.py` from the machine model's data (`benchmarks/pouw/machine_model/data`), and
checked against it by `benchmarks/pouw/tests/test_lean_headline.py`: edit the data and rerun the exporter, never this
file. `Accounting.Guarantees.TS.Sm120SheetHeadline` reads it.

* `classes`: the 122 instruction classes of the DS-120 census, in the order `Fin 122`
  indexes them.
* `ds120`, `sheet`: the RTX PRO 6000's data sheet (DS-120): f_max 2430 MHz (SPEC's P1); each tensor class's dense
  MACs a warp instruction (`m·n·k`, a 2:4 sparse form's dense equivalent) and its operand type's dense MACs an SM clock;
  the Nsight Compute pipe of IMAD, LOP3, XU and SHFL with its Programming Guide width; 128 B for every
  shared-memory class. Its machine reproduces every DS-120 coefficient.
* `h1fx1`, `model`: H1 + fx1 (`dither-c-peel-fx1`) as a TS model on the sheet's machine: writer kinds the census's MMA
  forms (ω = m·n), the generic MMA (128) and the uniform `UNI` (1) and `UNI64` (2); the vector classes CUDA's lane
  classes; fx1's fold on lanes (ALU 2 + IMAD 2 lane instructions a word) and alone on the uniform datapath (4 a word);
  A6's floors `q = q_T = ½`, `b_T = 0`; 128 committed words a unit of honest work (an IMMA.16832 at L = 32).
* `flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold, no filler,
  and ε_skip's skippable work at its issue price, 0.420701% of ♭'s bound (`eps_budget.design_skip`
  on DS-120).
* `piStar`: π*, the facet the machine model's LP certificate sits on at q* = ½.
* `checks`: the checks a window, one a round (Theorem S's S).
-/

namespace Accounting.TS.Headline.Sm120

open Accounting.TS

/-- The DS-120 census's instruction classes, in index order. -/
def classes : List String := [
  "ALU", "ALU64", "ATOME32", "ATOME32P", "ATOME64", "ATOMG32", "ATOMG64", "ATOMS32", "ATOMS64", "CVT64", "DMUL",
  "DSMST4P", "F2FS", "FFMA", "G4_16", "G4_1K", "G4_256", "G4_32", "HFMA2", "HMMA16816", "HMMA16816F16", "HMMA1684T",
  "HMMA1688T", "HMMASP16832", "IMAD", "IMADW", "IMMA16816", "IMMA16832", "IMMA8816", "IMMASP16832", "IMMASP16864",
  "LDC64", "LDCDIV", "LDCUNI", "LDG128", "LDG128CG", "LDG128NA", "LDG256", "LDG256CG", "LDG32", "LDG64", "LDGSTS16",
  "LDS128", "LDS32", "LDS64", "LDSM4T_X1", "LDSM4T_X2", "LDSM4_X1", "LDSM4_X2", "LDSM4_X4", "LDSM6T_X1", "LDSM6T_X2",
  "LDSM6_X1", "LDSM6_X2", "LDSM6_X4", "LDSM8T_X1", "LDSM8T_X2", "LDSM_X1", "LDSM_X1T", "LDSM_X2", "LDSM_X2T", "LDSM_X4",
  "LDSM_X4T", "MMA", "MOV64", "NOROW", "OMMASF16864", "QMMA16832", "QMMASF16832", "QMMASP16864", "RED32", "RED64",
  "REDAS32", "REDAS64", "REDG32", "REDG64", "REDV", "SEL", "SHFL", "STS128", "STSM8T_X1", "STSM8T_X2", "STSM8T_X4",
  "STSM_X1", "STSM_X1T", "STSM_X2", "STSM_X2T", "STSM_X4", "STSM_X4T", "SULD1", "SULD4", "SURED", "TEX1", "TEX2",
  "TEX4", "TEXBIL", "TLD1", "TLD2", "TLD2D1", "TLD2D4", "TLD4G", "TLD4w", "TMA512", "TMAred512", "UBLKCP4K", "UNI",
  "UNI64", "UTMA", "UTMA16K", "UTMA1K", "UTMA2K", "UTMA32K", "UTMA48K", "UTMA4K", "UTMA512", "UTMA64K", "UTMA8",
  "UTMA8K", "UTMABIG", "VIMNMX", "W2", "XU"]

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
  | imad
  | lop
  | xu
  | shfl
  deriving DecidableEq

instance : Fintype Pipe where
  elems := {Pipe.imad, Pipe.lop, Pipe.xu, Pipe.shfl}
  complete x := by cases x <;> simp

/-- Each pipe's Programming Guide results an SM clock. -/
def Pipe.width : Pipe → ℕ
  | .imad => 64
  | .lop => 64
  | .xu => 16
  | .shfl => 32

/-- The RTX PRO 6000's data sheet, by class index. -/
def ds120 : SheetData Pipe where
  fmax := 2430
  macs := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2048, 2048, 512, 1024, 2048, 0, 0, 2048, 4096, 1024, 2048,
    4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4096, 0, 0,
    8192, 4096, 4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  peak := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 512, 512, 256, 256, 512, 0, 0, 1024, 1024, 1024, 1024,
    1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1024, 0, 0,
    2048, 1024, 1024, 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  pipe := [
    some .lop, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, none, none, some .imad, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, some .shfl, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, none, none, none, none, none, none, some .xu]
  width := Pipe.width
  smem := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin 122) Pipe := ds120.sheet 122

/-- H1 + fx1's design data: writer kinds as (class, ω), vector classes, fold routes. -/
def h1fx1 : DesignData where
  writers := [
    (19, 128),  -- HMMA16816
    (20, 128),  -- HMMA16816F16
    (21, 128),  -- HMMA1684T
    (22, 128),  -- HMMA1688T
    (23, 128),  -- HMMASP16832
    (26, 128),  -- IMMA16816
    (27, 128),  -- IMMA16832
    (28, 64),  -- IMMA8816
    (29, 128),  -- IMMASP16832
    (30, 128),  -- IMMASP16864
    (66, 128),  -- OMMASF16864
    (67, 128),  -- QMMA16832
    (68, 128),  -- QMMASF16832
    (69, 128),  -- QMMASP16864
    (63, 128),  -- MMA
    (105, 1),  -- UNI
    (106, 2)  -- UNI64
  ]
  vec := [13, 119, 18, 24, 0, 12, 121, 78, 65, 77]  -- FFMA, VIMNMX, HFMA2, IMAD, ALU, F2FS, XU, SHFL, NOROW, SEL
  folds := [
    [(0, 1, 16), (24, 1, 16)],
    [(105, 4, 1)]]
  bytes := 39  -- LDG32
  q := 1 / 2
  qT := 1 / 2
  Nw := 128

/-- H1 + fx1's TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin 122) (SheetRow Pipe) (Fin 17) Unit Empty (Fin 2) Unit :=
  h1fx1.model 122 17 2 sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := [(27, 1, 1), (61, 25641, 1000000)]

/-- ♭'s fold a unit: fx1 on lanes for 128 words. -/
def flatFold : Sparse := [(0, 8, 1), (24, 8, 1)]

/-- ♭'s skippable work a unit, at its issue price (NOROW). -/
def flatSkip : Sparse := [(65, 14325412520309335111003, 199158597757329117000000)]

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin 122) := labelled 122 flatWords flatFold [] flatSkip

/-- π*: the machine model's certificate at q* = ½. -/
def piStar : SheetRow Pipe := .issue

/-- The checks a window. -/
def checks : ℕ := 48000

end Accounting.TS.Headline.Sm120
