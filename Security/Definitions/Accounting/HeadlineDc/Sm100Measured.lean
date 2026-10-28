import Definitions.Accounting.HeadlineDc.Sm100

/-!
# B200 (sm_100)'s measured headline instance: DC-100-ri and `dither-2s-licpw-redux-ldtm0-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm100MeasuredHeadline` reads it.

* `rowNames`, `dc100ri`, `machine`: the measured vector DC-100-ri, 9 rows of SMSP-cycles a warp instruction at its
  epoch's census, each sparse over `Sm100.classes`; every facet SM-clocked, f_max 1965 MHz. `UNLISTED` is A2's catch-all
  here: every warp instruction no other class prices on DC-100-ri counts there, at the measured issue slot alone. The
  sheet's classes no run beside DC-100-ri's census timed (DMMA16816, DMMA1684, DMMA1688, DMMA884,
  UTCHMMA_BF16_M128_N256, UTCHMMA_TF32_M128_N256, UTCIMMA_M128_N256_SP, UTCOMMA_MXF4NVF4_M128_N256,
  UTCOMMA_MXF4_M128_N256, UTCQMMA_E2M1_M128_N256, UTCQMMA_E2M3_M128_N256, UTCQMMA_E3M2_M128_N256,
  UTCQMMA_E5M2_M128_N256, UTCQMMA_MXF8F6F4_E4M3_M128_N256) are read the same way, at the measured issue slot alone, and
  are off the writer menu, as they were when the catch-all counted them: `NoSum` carries their words.
* `d2s`, `model`: `dither-2s-licpw-redux-ldtm0-l32` on that machine, its writer kinds every MMA class it times (ω its
  destination's words, A6); `Sm100.d2s`'s vector classes, fold routes, floors and `N_w`.
* `flatFold`, `flatSkip`, `flat`: ♭ is `Sm100.flat`'s word production and filler, its fold DC-100-ri's (the fold routes'
  cheapest, cp-red2 on every word), with ε_skip's skippable work at DC-100-ri's 0.814462% (`eps_budget.design_skip` on
  DC-100-ri), at FFMA's price.
* `lamStar`: π*, the certificate minimising the bound over this menu (issue 1.000000).
-/

namespace Accounting.TS.Headline.Sm100Measured

open Accounting.TS

/-- DC-100-ri's rows, in index order. -/
def rowNames : List String := [
  "issue", "M", "A", "I", "SHFL", "TL", "TM100_0", "JI100_0", "UP"]

/-- DC-100-ri's coefficients, row by row: `(class index, numerator, denominator)`. -/
def dc100ri : List Sparse := [
  -- 0: issue
  [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1), (5, 1, 1), (6, 1, 1), (7, 1, 1), (8, 1, 1), (9, 1, 1),
   (10, 1, 1), (11, 1, 1), (12, 1, 1), (13, 1, 1), (14, 1, 1), (15, 1, 1), (16, 1, 1), (17, 1, 1), (18, 1, 1),
   (19, 1, 1), (20, 1, 1), (21, 1, 1), (22, 1, 1), (23, 1, 1), (24, 1, 1), (25, 1, 1), (26, 1, 1), (27, 1, 1),
   (28, 1, 1), (29, 4621557459999999, 5000000000000000), (30, 1, 1), (31, 1, 1), (32, 1, 1), (33, 1, 1), (34, 1, 1),
   (35, 1, 1), (36, 1, 1), (37, 1, 1), (38, 1, 1), (39, 1, 1), (40, 1, 1), (41, 1, 1), (42, 1, 1), (43, 1, 1),
   (44, 1, 1), (45, 1, 1), (46, 1, 1), (47, 1, 1), (48, 1, 1), (49, 1, 1), (50, 1, 1), (51, 1, 1)],
  -- 1: M
  [(9, 490339257494561, 62500000000000), (10, 3923192829282303, 500000000000000),
   (32, 5018695645016817, 10000000000000), (34, 25097075487987533, 100000000000000),
   (35, 5018695645016817, 10000000000000), (37, 490239219482687, 3125000000000),
   (38, 18824101664455307, 100000000000000), (39, 12547895316422789, 50000000000000),
   (40, 627327321630517, 1250000000000), (41, 12550786758618753, 100000000000000),
   (47, 25097075487987533, 100000000000000), (48, 5018695645016817, 10000000000000)],
  -- 2: A
  [(0, 3922117723932331, 2000000000000000)],
  -- 3: I
  [(8, 9805637555199611, 5000000000000000)],
  -- 4: SHFL
  [(24, 98061966062401, 25000000000000)],
  -- 5: TL
  [(12, 7874224693993849, 500000000000000), (13, 392933298943811, 12500000000000),
   (14, 1611612376280547, 100000000000000), (15, 15782103441903157, 5000000000000000),
   (16, 1642247432691011, 25000000000000), (17, 34072153918911, 4000000000000),
   (18, 1654593511461907, 100000000000000), (19, 17773123589147233, 5000000000000000),
   (20, 3333703374407893, 100000000000000)],
  -- 6: TM100_0
  [(18, 10337059, 625000), (35, 199132141, 10000000)],
  -- 7: JI100_0
  [(0, 122510713, 125000000), (7, 122510713, 125000000), (8, 122510713, 125000000), (9, 122510713, 125000000),
   (10, 122510713, 125000000), (12, 122510713, 125000000), (13, 122510713, 125000000), (14, 122510713, 125000000),
   (15, 122510713, 125000000), (16, 122510713, 125000000), (17, 122510713, 125000000), (18, 122510713, 125000000),
   (19, 122510713, 125000000), (20, 122510713, 125000000), (21, 122510713, 125000000), (24, 122510713, 125000000),
   (29, 539, 625000), (32, 122510713, 125000000), (34, 122510713, 125000000), (35, 10879029, 10000000),
   (37, 122510713, 125000000), (38, 122510713, 125000000), (39, 122510713, 125000000), (40, 122510713, 125000000),
   (41, 122510713, 125000000), (47, 122510713, 125000000), (48, 122510713, 125000000)],
  -- 8: UP
  [(29, 9786457500000001, 5000000000000000)]]

/-- The measured machine: DC-100-ri's rows over `Sm100.classes`. -/
noncomputable def machine : FacetCosts (Fin 52) (Fin 9) :=
  rowsMachine 52 9 dc100ri 1965

/-- `dither-2s-licpw-redux-ldtm0-l32`'s design data on DC-100-ri: writer kinds as (class, ω), and `Sm100.d2s`'s routes.
-/
def d2s : DesignData where
  writers := [
    (9, 128),  -- IMMA16816_S8
    (10, 128),  -- IMMA16832_S8
    (32, 32768),  -- UTCHMMA_F16_M128_N256
    (34, 16384),  -- UTCIMMA_M128_N128
    (35, 32768),  -- UTCIMMA_M128_N256
    (37, 4096),  -- UTCIMMA_M128_N32
    (38, 8192),  -- UTCIMMA_M128_N64
    (39, 8192),  -- UTCIMMA_M64_N128
    (40, 16384),  -- UTCIMMA_M64_N256
    (41, 4096),  -- UTCIMMA_M64_N64
    (47, 16384),  -- UTCQMMA_E4M3_M128_N128
    (48, 32768),  -- UTCQMMA_E4M3_M128_N256
    (29, 2)  -- UNI
  ]
  vec := Sm100.d2s.vec
  folds := Sm100.d2s.folds
  bytes := Sm100.d2s.bytes
  q := Sm100.d2s.q
  qT := Sm100.d2s.qT
  Nw := Sm100.d2s.Nw

/-- `dither-2s-licpw-redux-ldtm0-l32`'s TS model on the measured machine. -/
noncomputable def model : Model (Fin 52) (Fin 9) (Fin 13) Unit Empty (Fin 14) Unit :=
  d2s.model 52 13 14 machine

/-- ♭'s fold a unit on DC-100-ri: the lane fold for 32768 words, the fold routes' cheapest, cp-red2 on every word. -/
def flatFold : Sparse := [(0, 536, 1), (8, 512, 1), (22, 16, 1)]

/-- ♭'s skippable work a unit on DC-100-ri, at its FFMA price. -/
def flatSkip : Sparse := [(7, 1445669286395, 165309230717)]

/-- The floor program ♭ on DC-100-ri, labelled. -/
noncomputable def flat : Labelled (Fin 52) :=
  labelled 52 Sm100.flatWords flatFold Sm100.flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin 9 → ℝ :=
  cert 9 [(0, 1, 1)]

end Accounting.TS.Headline.Sm100Measured
