import Definitions.Accounting.HeadlineDc.Encoding

/-!
# H200 (sm_90)'s headline instance: DS-90 and `dither-2s-licpw-redux-p1-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm90SheetHeadline` reads it.

* `classes`: the 50 instruction classes of the DS-90 census, in the order `Fin 50` indexes them. `NOROW` is A2's
  catch-all: every warp instruction no other class names counts there, at its issue slot alone.
* `ds90`, `sheet`: the H200's data sheet (DS-90), `benchmarks/pouw/machine_model/data/devices/sm_90/sheet.toml` as
  `Security/Definitions/Accounting/Sheets/H200.lean` states it: f_max 1980 MHz (EPS1 DC's measured f_max at default
  clocks; the sheet's TOPS are at 1830 MHz); each tensor class's dense MACs a warp instruction (sm_90's wgmma as one
  warp's quarter of its warpgroup instruction (`m·n·k/4`), a 2:4 sparse form at its dense twin's, its mma.sync whole,
  DMMA's at the f64 peak) and its operand type's dense MACs an SM clock; the Nsight Compute pipes ALU, IMAD, SHFL and
  UNIFORM, each at its width (the Programming Guide's, and the uniform datapath's the census's measured UP line, which
  no document states); no shared-memory row (DS-90 prices none). An MMA class the sheet rates on no peak (b1) and the
  memory, atomic and control classes are their issue slot alone (A2). Its machine reproduces every DS-90 coefficient.
* `d2s`, `model`: `dither-2s-licpw-redux-p1-l32` as a TS model on the sheet's machine: writer kinds every MMA class of
  the census (ω its destination's 32-bit words a warp instruction, A6, sheet.toml's `omega`: an f64 element two) and the
  uniform `UNI` (2, a UR pair); the vector classes CUDA's lane classes and both readings' catch-alls (NOROW, UNLISTED:
  A6 counts a vector word's instructions in them, and an instruction no class names, IMAD.WIDE among them, counts in the
  catch-all); the fold routes the design's fold and its 54 other programs (cp-seg4, cp-seg4u, cp-ats, cp-ats2, cp-ats8,
  cp-red, cp-red2, cp-red8, cp-red-generic, cp-red2-generic, cp-red8-generic, cp-xt: the checkpoint's bit-exact loops;
  li-bulk: li's XOR through the stores, a bulk XOR with the key slab and a load back, at its measured cost; and the
  counted folds, none built, which ♭ never folds by: li-rt3, li-wideh, li-rt3-wideh, cp-seg4-rt3, cp-seg4-wideh,
  cp-seg4-rt3-wideh, cp-seg4u-rt3, cp-seg4u-wideh, cp-seg4u-rt3-wideh, cp-ats-rt3, cp-ats-wideh, cp-ats-rt3-wideh,
  cp-ats2-rt3, cp-ats2-wideh, cp-ats2-rt3-wideh, cp-ats8-rt3, cp-ats8-wideh, cp-ats8-rt3-wideh, cp-red-rt3,
  cp-red-wideh, cp-red-rt3-wideh, cp-red2-rt3, cp-red2-wideh, cp-red2-rt3-wideh, cp-red8-rt3, cp-red8-wideh,
  cp-red8-rt3-wideh, cp-red-generic-rt3, cp-red-generic-wideh, cp-red-generic-rt3-wideh, cp-red2-generic-rt3,
  cp-red2-generic-wideh, cp-red2-generic-rt3-wideh, cp-red8-generic-rt3, cp-red8-generic-wideh,
  cp-red8-generic-rt3-wideh, cp-xt-rt3, cp-xt-wideh, cp-xt-rt3-wideh, the counted moves: each lane route and the fold
  with LI's 64 LOP3 a segment as a memory round trip of three issue slots (ATOM.E.XOR and two LD.E back), its 64 IMAD as
  IMAD.WIDE, or both; li-bulk-bt1024, li-bulk-trim-inf, li-bulk's BT1024 and trimmed W = ∞ members) on lanes, and alone
  on the uniform datapath at the least of their counts; the table level's bytes counted in ALU, which `b_T = 0` prices
  at nothing (the census has no load class); A6's floors `q = q_T = ½`, `b_T = 0`; 4096 committed words a unit of honest
  work (an IGMMA_S8_N256 at L = 32).
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
  "ALU", "ATOME32", "ATOMS32", "BGMMA_B1_N256", "BMMA168256_B1", "DMMA16816", "DMMA1684", "DMMA1688", "DMMA884", "FFMA",
  "HGMMA_BF16_N128", "HGMMA_BF16_N256", "HGMMA_BF16_N256_SP", "HGMMA_F16_N256", "HGMMA_F16_N256_SP", "HGMMA_TF32_N256",
  "HGMMA_TF32_N256_SP", "HMMA16816_BF16", "HMMA16816_F16", "HMMA1688_TF32", "IGMMA_S8_N128", "IGMMA_S8_N16",
  "IGMMA_S8_N192", "IGMMA_S8_N256", "IGMMA_S8_N256_RS", "IGMMA_S8_N256_SP", "IGMMA_S8_N32", "IGMMA_S8_N64",
  "IGMMA_S8_N8", "IGMMA_U8S8_N256", "IMAD", "IMMA16816_S8", "IMMA16832_S8", "ISETP", "LDS128", "NOROW",
  "QGMMA_E4M3_N128", "QGMMA_E4M3_N256", "QGMMA_E4M3_N256_SP", "QGMMA_E5M2_N256", "QGMMA_E5M2_N256_SP", "REDG32",
  "REDUX", "SHFL", "STG128", "STG32", "STS128", "STS32", "UNI", "UNLISTED"]

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
  | .uniform => 3200000000 / 49027679

/-- The H200's data sheet, by class index. -/
def ds90 : SheetDataQ Pipe where
  fmax := 1980
  macs := [
    0, 0, 0, 0, 0, 2048, 512, 1024, 256, 0, 32768, 65536, 65536, 65536, 65536, 32768, 32768, 2048, 2048, 1024, 65536,
    8192, 98304, 131072, 131072, 131072, 16384, 32768, 4096, 131072, 0, 2048, 4096, 0, 0, 0, 65536, 131072, 131072,
    131072, 131072, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  peak := [
    0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 2048, 2048, 2048, 2048, 2048, 1024, 1024, 2048, 2048, 1024, 4096, 4096, 4096,
    4096, 4096, 4096, 4096, 4096, 4096, 4096, 0, 4096, 4096, 0, 0, 0, 4096, 4096, 4096, 4096, 4096, 0, 0, 0, 0, 0, 0, 0,
    0, 0]
  pipe := [
    some .alu, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none,
    none, none, none, none, none, none, none, none, none, none, none, none, some .imad, none, none, none, none, none,
    none, none, none, none, none, none, none, some .shfl, none, none, none, none, some .uniform, none]
  width := Pipe.width
  smem := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin 50) Pipe := ds90.sheet 50

/-- `dither-2s-licpw-redux-p1-l32`'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
    (3, 4096),  -- BGMMA_B1_N256
    (4, 128),  -- BMMA168256_B1
    (5, 256),  -- DMMA16816
    (6, 256),  -- DMMA1684
    (7, 256),  -- DMMA1688
    (8, 128),  -- DMMA884
    (10, 2048),  -- HGMMA_BF16_N128
    (11, 4096),  -- HGMMA_BF16_N256
    (12, 4096),  -- HGMMA_BF16_N256_SP
    (13, 4096),  -- HGMMA_F16_N256
    (14, 4096),  -- HGMMA_F16_N256_SP
    (15, 4096),  -- HGMMA_TF32_N256
    (16, 4096),  -- HGMMA_TF32_N256_SP
    (17, 128),  -- HMMA16816_BF16
    (18, 128),  -- HMMA16816_F16
    (19, 128),  -- HMMA1688_TF32
    (20, 2048),  -- IGMMA_S8_N128
    (21, 256),  -- IGMMA_S8_N16
    (22, 3072),  -- IGMMA_S8_N192
    (23, 4096),  -- IGMMA_S8_N256
    (24, 4096),  -- IGMMA_S8_N256_RS
    (25, 4096),  -- IGMMA_S8_N256_SP
    (26, 512),  -- IGMMA_S8_N32
    (27, 1024),  -- IGMMA_S8_N64
    (28, 128),  -- IGMMA_S8_N8
    (29, 4096),  -- IGMMA_U8S8_N256
    (31, 128),  -- IMMA16816_S8
    (32, 128),  -- IMMA16832_S8
    (36, 2048),  -- QGMMA_E4M3_N128
    (37, 4096),  -- QGMMA_E4M3_N256
    (38, 4096),  -- QGMMA_E4M3_N256_SP
    (39, 4096),  -- QGMMA_E5M2_N256
    (40, 4096),  -- QGMMA_E5M2_N256_SP
    (48, 2)  -- UNI
  ]
  vec := [9, 30, 0, 43, 35, 49]  -- FFMA, IMAD, ALU, SHFL, NOROW, UNLISTED
  folds := [
    [(0, 17, 1024), (30, 1, 64), (43, 3, 4096)],
    [(0, 17, 1024), (30, 65, 4096), (42, 1, 4096), (44, 1, 16384)],
    [(0, 67, 4096), (30, 1, 64), (42, 1, 2048), (44, 1, 16384), (48, 1, 4096)],
    [(0, 17, 1024), (2, 1, 4096), (30, 1, 64), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384)],
    [(0, 67, 4096), (2, 1, 2048), (30, 1, 64), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384)],
    [(0, 1, 64), (2, 1, 512), (30, 1, 64), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384)],
    [(0, 17, 1024), (30, 1, 64), (41, 1, 4096)],
    [(0, 67, 4096), (30, 1, 64), (41, 1, 2048)],
    [(0, 1, 64), (30, 1, 64), (41, 1, 512)],
    [(0, 17, 1024), (1, 1, 4096), (30, 1, 64)],
    [(0, 67, 4096), (1, 1, 2048), (30, 1, 64)],
    [(0, 1, 64), (1, 1, 512), (30, 1, 64)],
    [(0, 137, 8192), (30, 1, 64), (34, 1, 16384), (45, 1, 131072), (47, 1, 4096)],
    [(0, 1, 1024), (1, 1, 64), (30, 1, 64), (43, 3, 4096), (49, 1, 32)],
    [(0, 17, 1024), (43, 3, 4096), (49, 1, 64)],
    [(0, 1, 1024), (1, 1, 64), (43, 3, 4096), (49, 3, 64)],
    [(0, 1, 1024), (1, 1, 64), (30, 65, 4096), (42, 1, 4096), (44, 1, 16384), (49, 1, 32)],
    [(0, 17, 1024), (30, 1, 4096), (42, 1, 4096), (44, 1, 16384), (49, 1, 64)],
    [(0, 1, 1024), (1, 1, 64), (30, 1, 4096), (42, 1, 4096), (44, 1, 16384), (49, 3, 64)],
    [(0, 3, 4096), (1, 1, 64), (30, 1, 64), (42, 1, 2048), (44, 1, 16384), (48, 1, 4096), (49, 1, 32)],
    [(0, 67, 4096), (42, 1, 2048), (44, 1, 16384), (48, 1, 4096), (49, 1, 64)],
    [(0, 3, 4096), (1, 1, 64), (42, 1, 2048), (44, 1, 16384), (48, 1, 4096), (49, 3, 64)],
    [(0, 1, 1024), (1, 1, 64), (2, 1, 4096), (30, 1, 64), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 1, 32)],
    [(0, 17, 1024), (2, 1, 4096), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 1, 64)],
    [(0, 1, 1024), (1, 1, 64), (2, 1, 4096), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 3, 64)],
    [(0, 3, 4096), (1, 1, 64), (2, 1, 2048), (30, 1, 64), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 1, 32)],
    [(0, 67, 4096), (2, 1, 2048), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 1, 64)],
    [(0, 3, 4096), (1, 1, 64), (2, 1, 2048), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 3, 64)],
    [(1, 1, 64), (2, 1, 512), (30, 1, 64), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 1, 32)],
    [(0, 1, 64), (2, 1, 512), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 1, 64)],
    [(1, 1, 64), (2, 1, 512), (34, 1, 16384), (44, 1, 16384), (46, 1, 16384), (49, 3, 64)],
    [(0, 1, 1024), (1, 1, 64), (30, 1, 64), (41, 1, 4096), (49, 1, 32)],
    [(0, 17, 1024), (41, 1, 4096), (49, 1, 64)],
    [(0, 1, 1024), (1, 1, 64), (41, 1, 4096), (49, 3, 64)],
    [(0, 3, 4096), (1, 1, 64), (30, 1, 64), (41, 1, 2048), (49, 1, 32)],
    [(0, 67, 4096), (41, 1, 2048), (49, 1, 64)],
    [(0, 3, 4096), (1, 1, 64), (41, 1, 2048), (49, 3, 64)],
    [(1, 1, 64), (30, 1, 64), (41, 1, 512), (49, 1, 32)],
    [(0, 1, 64), (41, 1, 512), (49, 1, 64)],
    [(1, 1, 64), (41, 1, 512), (49, 3, 64)],
    [(0, 1, 1024), (1, 65, 4096), (30, 1, 64), (49, 1, 32)],
    [(0, 17, 1024), (1, 1, 4096), (49, 1, 64)],
    [(0, 1, 1024), (1, 65, 4096), (49, 3, 64)],
    [(0, 3, 4096), (1, 33, 2048), (30, 1, 64), (49, 1, 32)],
    [(0, 67, 4096), (1, 1, 2048), (49, 1, 64)],
    [(0, 3, 4096), (1, 33, 2048), (49, 3, 64)],
    [(1, 9, 512), (30, 1, 64), (49, 1, 32)],
    [(0, 1, 64), (1, 1, 512), (49, 1, 64)],
    [(1, 9, 512), (49, 3, 64)],
    [(0, 9, 8192), (1, 1, 64), (30, 1, 64), (34, 1, 16384), (45, 1, 131072), (47, 1, 4096), (49, 1, 32)],
    [(0, 137, 8192), (34, 1, 16384), (45, 1, 131072), (47, 1, 4096), (49, 1, 64)],
    [(0, 9, 8192), (1, 1, 64), (34, 1, 16384), (45, 1, 131072), (47, 1, 4096), (49, 3, 64)],
    [(0, 13, 4096), (1, 1, 512), (30, 75, 4096), (33, 1, 4096), (44, 1, 256), (46, 1, 256), (48, 9, 4096), (49, 4145547, 1280000)],
    [(0, 5, 8192), (1, 1, 512), (30, 75, 4096), (33, 1, 4096), (44, 1, 256), (46, 1, 256), (48, 1, 2048), (49, 153, 8192)],
    [(0, 1, 4096), (1, 1, 512), (30, 65, 4096), (33, 1, 4096), (44, 1, 256), (46, 1, 256), (48, 1, 4096), (49, 49, 4096)],
    [(48, 133, 128)]]
  bytes := 0  -- ALU
  q := 1 / 2
  qT := 1 / 2
  Nw := 4096

/-- ♭'s filler a unit: none. -/
def flatFill : Sparse := []

/-- `dither-2s-licpw-redux-p1-l32`'s TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin 50) (SheetRow Pipe) (Fin 34) Unit Empty (Fin 56) Unit :=
  d2s.model 50 34 56 sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := [(23, 1, 1)]

/-- ♭'s fold a unit: the lane fold for 4096 words, the fold routes' cheapest, cp-red2 on every word. -/
def flatFold : Sparse := [(0, 67, 1), (30, 64, 1), (41, 2, 1)]

/-- ♭'s skippable work a unit, at FFMA's price. -/
def flatSkip : Sparse := [(9, 17166648254, 14797263819)]

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin 50) := labelled 50 flatWords flatFold flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : SheetRow Pipe → ℝ
  | .issue => 1 / 1
  | _ => 0

/-- The checks a window. -/
def checks : ℕ := 48000

end Accounting.TS.Headline.Sm90
