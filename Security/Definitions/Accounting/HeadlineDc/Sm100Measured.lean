import Definitions.Accounting.HeadlineDc.Sm100

/-!
# B200 (sm_100)'s measured headline instance: DC-100-ri and `dither-2s-licpw-redux-ldtm0-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm100MeasuredHeadline` reads it.

* `rowNames`, `dc100ri`, `machine`: the measured vector DC-100-ri, 12 rows of SMSP-cycles a warp instruction at its
  epoch's census, each sparse over `Sm100.classes`; every facet SM-clocked, f_max 1965 MHz.
* `d2s`, `model`: `dither-2s-licpw-redux-ldtm0-l32` on that machine, its writer kinds every census MMA class (ω = m·n);
  `Sm100.d2s`'s vector classes, fold routes, floors and `N_w`.
* `flatFold`, `flatSkip`, `flat`: ♭ is `Sm100.flat`'s word production and filler, its fold DC-100-ri's (the fold routes'
  cheapest mix, by share of the words: cp-seg4 0.9242, cp-seg4u 0.0758), with ε_skip's skippable work at DC-100-ri's
  0.806079% (`eps_budget.design_skip` on DC-100-ri), at FFMA's price.
* `lamStar`: π*, the certificate minimising the bound over this menu (issue 1.000000).
-/

namespace Accounting.TS.Headline.Sm100Measured

open Accounting.TS

/-- DC-100-ri's rows, in index order. -/
def rowNames : List String := [
  "issue", "M", "A", "I", "SHFL", "TL", "TM100_0", "JI100_0", "JPA100_0", "JPA100_1", "UP", "TMAB"]

/-- DC-100-ri's coefficients, row by row: `(class index, numerator, denominator)`. -/
def dc100ri : List Sparse := [
  -- 0: issue
  [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1), (5, 1, 1), (6, 1, 1), (7, 1, 1), (8, 1, 1), (9, 1, 1),
   (10, 1, 1), (11, 1, 1), (12, 1, 1), (13, 1, 1), (14, 1, 1), (15, 4621557459999999, 5000000000000000), (16, 1, 1),
   (17, 1, 1), (18, 1, 1), (19, 1, 1), (20, 1, 1), (21, 1, 1), (22, 1, 1), (23, 1, 1), (24, 1, 1), (25, 1, 1)],
  -- 1: M
  [(3, 7845459510917193, 1000000000000000), (4, 3923192829282303, 500000000000000),
   (16, 5018695645016817, 10000000000000), (17, 25097075487987533, 100000000000000),
   (18, 5018695645016817, 10000000000000), (19, 490239219482687, 3125000000000),
   (20, 18824101664455307, 100000000000000), (21, 12547895316422789, 50000000000000),
   (22, 627327321630517, 1250000000000), (23, 12550786758618753, 100000000000000),
   (24, 25097075487987533, 100000000000000), (25, 5018695645016817, 10000000000000)],
  -- 2: A
  [(0, 3922117723932331, 2000000000000000)],
  -- 3: I
  [(2, 9805637555199611, 5000000000000000)],
  -- 4: SHFL
  [(14, 98061966062401, 25000000000000)],
  -- 5: TL
  [(5, 7874224693993849, 500000000000000), (6, 392933298943811, 12500000000000),
   (7, 1611612376280547, 100000000000000), (8, 15782103441903157, 5000000000000000),
   (9, 1642247432691011, 25000000000000), (10, 34072153918911, 4000000000000),
   (11, 1654593511461907, 100000000000000), (12, 17773123589147233, 5000000000000000),
   (13, 3333703374407893, 100000000000000)],
  -- 6: TM100_0
  [(11, 10337059, 625000), (18, 19911599, 1000000)],
  -- 7: JI100_0
  [(0, 122510713, 125000000), (1, 122510713, 125000000), (2, 122510713, 125000000), (3, 122510713, 125000000),
   (4, 122510713, 125000000), (5, 122510713, 125000000), (6, 122510713, 125000000), (7, 122510713, 125000000),
   (8, 122510713, 125000000), (9, 122510713, 125000000), (10, 122510713, 125000000), (11, 122510713, 125000000),
   (12, 122510713, 125000000), (13, 122510713, 125000000), (14, 122510713, 125000000), (15, 539, 625000),
   (16, 122510713, 125000000), (17, 122510713, 125000000), (18, 108760423, 100000000), (19, 122510713, 125000000),
   (20, 122510713, 125000000), (21, 122510713, 125000000), (22, 122510713, 125000000), (23, 122510713, 125000000),
   (24, 122510713, 125000000), (25, 122510713, 125000000)],
  -- 8: JPA100_0
  [(0, 3920543, 2000000), (18, 27417101, 2500000)],
  -- 9: JPA100_1
  [(0, 682401081, 1000000000), (18, 250834541, 500000)],
  -- 10: UP
  [(15, 9786457500000001, 5000000000000000)],
  -- 11: TMAB
  []]

/-- The measured machine: DC-100-ri's rows over `Sm100.classes`. -/
noncomputable def machine : Machine (Fin 26) (Fin 12) :=
  rowsMachine 26 12 dc100ri 1965

/-- `dither-2s-licpw-redux-ldtm0-l32`'s design data on DC-100-ri: writer kinds as (class, ω), and `Sm100.d2s`'s routes.
-/
def d2s : DesignData where
  writers := [
    (3, 128),  -- IMMA16816_S8
    (4, 128),  -- IMMA16832_S8
    (16, 32768),  -- UTCHMMA_F16_M128_N256
    (17, 16384),  -- UTCIMMA_M128_N128
    (18, 32768),  -- UTCIMMA_M128_N256
    (19, 4096),  -- UTCIMMA_M128_N32
    (20, 8192),  -- UTCIMMA_M128_N64
    (21, 8192),  -- UTCIMMA_M64_N128
    (22, 16384),  -- UTCIMMA_M64_N256
    (23, 4096),  -- UTCIMMA_M64_N64
    (24, 16384),  -- UTCQMMA_E4M3_M128_N128
    (25, 32768),  -- UTCQMMA_E4M3_M128_N256
    (15, 2)  -- UNI
  ]
  vec := Sm100.d2s.vec
  folds := Sm100.d2s.folds
  bytes := Sm100.d2s.bytes
  q := Sm100.d2s.q
  qT := Sm100.d2s.qT
  Nw := Sm100.d2s.Nw

/-- `dither-2s-licpw-redux-ldtm0-l32`'s TS model on the measured machine. -/
noncomputable def model : Model (Fin 26) (Fin 12) (Fin 13) Unit Empty (Fin 4) Unit :=
  d2s.model 26 13 4 machine

/-- ♭'s fold a unit on DC-100-ri: the lane fold for 32768 words, the fold routes' cheapest mix, by share of the words:
cp-seg4 0.9242, cp-seg4u 0.0758. -/
def flatFold : Sparse := [(0, 1047120, 1927), (2, 512, 1), (14, 37022, 1927), (15, 1168, 1927)]

/-- ♭'s skippable work a unit on DC-100-ri, at its FFMA price. -/
def flatSkip : Sparse := [(1, 5223825302150538997404942486147, 597333394309954062500000000000)]

/-- The floor program ♭ on DC-100-ri, labelled. -/
noncomputable def flat : Labelled (Fin 26) :=
  labelled 26 Sm100.flatWords flatFold Sm100.flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin 12 → ℝ :=
  cert 12 [(0, 1, 1)]

end Accounting.TS.Headline.Sm100Measured
