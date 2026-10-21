import Definitions.Accounting.HeadlineDc.Sm90Measured

/-!
# H200 (sm_90)'s (I3) twin: DC-90-ri at q = 3/5 and `dither-2s-licpw-redux-p1-l32` (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.Sm90MeasuredI3Headline` reads it.

* `d2s`, `model`: `Sm90Measured.d2s` (its writer kinds, and `Sm90.d2s`'s vector classes, fold routes and `N_w`) at
  (I3)'s floors `q = q_T = 3/5`, `b_T = 0`, on `Sm90Measured.machine`. (I3) `ScalarRoute` there is
  `accounting-ts-scalar-route`, assumed: A6 gives it only at `q, q_T ≤ ½` (`SheetScalarRouteHalf`). ♭ is
  `Sm90Measured.flat`, since `setup`'s floor program doesn't read `q`.
* `lamStar`: π*, the certificate minimising the bound over this menu at `q = 3/5` (JI90_0 0.819832 + JI90_1 0.033778 +
  JPA90_0 0.146390).
-/

namespace Accounting.TS.Headline.Sm90MeasuredI3

open Accounting.TS

/-- `Sm90Measured.d2s` at (I3)'s floors `q = q_T = 3/5`. -/
def d2s : DesignData where
  writers := Sm90Measured.d2s.writers
  vec := Sm90Measured.d2s.vec
  folds := Sm90Measured.d2s.folds
  bytes := Sm90Measured.d2s.bytes
  q := 3 / 5
  qT := 3 / 5
  Nw := Sm90Measured.d2s.Nw

/-- `dither-2s-licpw-redux-p1-l32`'s TS model on the measured machine at (I3)'s floors. -/
noncomputable def model : Model (Fin 50) (Fin 15) (Fin 2) Unit Empty (Fin 56) Unit :=
  d2s.model 50 2 56 Sm90Measured.machine

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin 15 → ℝ :=
  cert 15 [(5, 12809871, 15625000), (6, 16889149, 500000000), (10, 73194979, 500000000)]

end Accounting.TS.Headline.Sm90MeasuredI3
