import Definitions.Accounting.HeadlineDc.Sm100

/-!
# B200 (sm_100)'s measured headline instance: DC-100-ri and `dither-2s-licpw-redux-ldtm0-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm100MeasuredHeadline` reads it.

* `rowNames`, `dc100ri`, `machine`: the measured vector DC-100-ri, 11 rows of SMSP-cycles a warp instruction at its
  epoch's census, each sparse over `Sm100.classes`; every facet SM-clocked, f_max 1965 MHz.
* `d2s`, `model`: `dither-2s-licpw-redux-ldtm0-l32` on that machine, its writer kinds every census MMA class (ω = m·n);
  `Sm100.d2s`'s vector classes, fold routes, floors and `N_w`.
* `flatFold`, `flatSkip`, `flat`: ♭ is `Sm100.flat`'s word production and filler, its fold DC-100-ri's (the fold routes'
  cheapest, cp-red2 on every word), with ε_skip's skippable work at DC-100-ri's 0.814462% (`eps_budget.design_skip` on
  DC-100-ri), at FFMA's price.
* `lamStar`: π*, the certificate minimising the bound over this menu (issue 1.000000).
-/

namespace Accounting.TS.Headline.Sm100Measured

open Accounting.TS

/-- DC-100-ri's rows, in index order. -/
def rowNames : List String := [
  "issue", "M", "A", "I", "SHFL", "TL", "TM100_0", "JI100_0", "JPA100_0", "JPA100_1", "UP"]

/-- DC-100-ri's coefficients, row by row: `(class index, numerator, denominator)`. -/
def dc100ri : List Sparse := [
  -- 0: issue
  [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1), (5, 1, 1), (6, 1, 1), (7, 1, 1), (8, 1, 1), (9, 1, 1),
   (10, 1, 1), (11, 1, 1), (12, 1, 1), (13, 1, 1), (14, 1, 1), (15, 1, 1), (16, 1, 1), (17, 1, 1), (18, 1, 1),
   (19, 1, 1), (20, 1, 1), (21, 1, 1), (22, 1, 1), (23, 1, 1), (24, 4621557459999999, 5000000000000000), (25, 1, 1),
   (26, 1, 1), (27, 1, 1), (28, 1, 1), (29, 1, 1), (30, 1, 1), (31, 1, 1), (32, 1, 1), (33, 1, 1), (34, 1, 1),
   (35, 1, 1)],
  -- 1: M
  [(4, 7845459510917193, 1000000000000000), (5, 3923192829282303, 500000000000000),
   (25, 5018695645016817, 10000000000000), (26, 25097075487987533, 100000000000000),
   (27, 5018695645016817, 10000000000000), (28, 490239219482687, 3125000000000),
   (29, 18824101664455307, 100000000000000), (30, 12547895316422789, 50000000000000),
   (31, 627327321630517, 1250000000000), (32, 12550786758618753, 100000000000000),
   (33, 25097075487987533, 100000000000000), (34, 5018695645016817, 10000000000000)],
  -- 2: A
  [(0, 3922117723932331, 2000000000000000)],
  -- 3: I
  [(3, 9805637555199611, 5000000000000000)],
  -- 4: SHFL
  [(19, 98061966062401, 25000000000000)],
  -- 5: TL
  [(7, 7874224693993849, 500000000000000), (8, 392933298943811, 12500000000000),
   (9, 1611612376280547, 100000000000000), (10, 15782103441903157, 5000000000000000),
   (11, 1642247432691011, 25000000000000), (12, 34072153918911, 4000000000000),
   (13, 1654593511461907, 100000000000000), (14, 17773123589147233, 5000000000000000),
   (15, 3333703374407893, 100000000000000)],
  -- 6: TM100_0
  [(13, 10337059, 625000), (27, 19911599, 1000000)],
  -- 7: JI100_0
  [(0, 122510713, 125000000), (1, 122510713, 125000000), (2, 122510713, 125000000), (3, 122510713, 125000000),
   (4, 122510713, 125000000), (5, 122510713, 125000000), (6, 122510713, 125000000), (7, 122510713, 125000000),
   (8, 122510713, 125000000), (9, 122510713, 125000000), (10, 122510713, 125000000), (11, 122510713, 125000000),
   (12, 122510713, 125000000), (13, 122510713, 125000000), (14, 122510713, 125000000), (15, 122510713, 125000000),
   (16, 122510713, 125000000), (17, 122510713, 125000000), (18, 122510713, 125000000), (19, 122510713, 125000000),
   (20, 122510713, 125000000), (21, 122510713, 125000000), (22, 122510713, 125000000), (23, 122510713, 125000000),
   (24, 539, 625000), (25, 122510713, 125000000), (26, 122510713, 125000000), (27, 108760423, 100000000),
   (28, 122510713, 125000000), (29, 122510713, 125000000), (30, 122510713, 125000000), (31, 122510713, 125000000),
   (32, 122510713, 125000000), (33, 122510713, 125000000), (34, 122510713, 125000000), (35, 122510713, 125000000)],
  -- 8: JPA100_0
  [(0, 3920543, 2000000), (27, 27417101, 2500000)],
  -- 9: JPA100_1
  [(0, 682401081, 1000000000), (27, 250834541, 500000)],
  -- 10: UP
  [(24, 9786457500000001, 5000000000000000)]]

/-- The measured machine: DC-100-ri's rows over `Sm100.classes`. -/
noncomputable def machine : Machine (Fin 36) (Fin 11) :=
  rowsMachine 36 11 dc100ri 1965

/-- `dither-2s-licpw-redux-ldtm0-l32`'s design data on DC-100-ri: writer kinds as (class, ω), and `Sm100.d2s`'s routes.
-/
def d2s : DesignData where
  writers := [
    (4, 128),  -- IMMA16816_S8
    (5, 128),  -- IMMA16832_S8
    (25, 32768),  -- UTCHMMA_F16_M128_N256
    (26, 16384),  -- UTCIMMA_M128_N128
    (27, 32768),  -- UTCIMMA_M128_N256
    (28, 4096),  -- UTCIMMA_M128_N32
    (29, 8192),  -- UTCIMMA_M128_N64
    (30, 8192),  -- UTCIMMA_M64_N128
    (31, 16384),  -- UTCIMMA_M64_N256
    (32, 4096),  -- UTCIMMA_M64_N64
    (33, 16384),  -- UTCQMMA_E4M3_M128_N128
    (34, 32768),  -- UTCQMMA_E4M3_M128_N256
    (24, 2)  -- UNI
  ]
  vec := Sm100.d2s.vec
  folds := Sm100.d2s.folds
  bytes := Sm100.d2s.bytes
  q := Sm100.d2s.q
  qT := Sm100.d2s.qT
  Nw := Sm100.d2s.Nw

/-- `dither-2s-licpw-redux-ldtm0-l32`'s TS model on the measured machine. -/
noncomputable def model : Model (Fin 36) (Fin 11) (Fin 13) Unit Empty (Fin 11) Unit :=
  d2s.model 36 13 11 machine

/-- ♭'s fold a unit on DC-100-ri: the lane fold for 32768 words, the fold routes' cheapest, cp-red2 on every word. -/
def flatFold : Sparse := [(0, 536, 1), (3, 512, 1), (17, 16, 1)]

/-- ♭'s skippable work a unit on DC-100-ri, at its FFMA price. -/
def flatSkip : Sparse := [(2, 1445669286395, 165309230717)]

/-- The floor program ♭ on DC-100-ri, labelled. -/
noncomputable def flat : Labelled (Fin 36) :=
  labelled 36 Sm100.flatWords flatFold Sm100.flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin 11 → ℝ :=
  cert 11 [(0, 1, 1)]

end Accounting.TS.Headline.Sm100Measured
