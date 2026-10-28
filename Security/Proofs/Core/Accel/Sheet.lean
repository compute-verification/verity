import Definitions.Core.Accel.H200
import Definitions.Accounting.Sheets.H200
import Mathlib.Tactic.NormNum

/-!
# The H200's tensor rates against its data sheet

The H200 instance's tensor rates, held to the sheet's dense peaks (`Accounting.TS.Sheets.H200`, A3), format by format.

* `tensor_sheet`: a tick's `tensor` units over a MAC's `weight` are the sheet's per-SM-clock peak times its SMs, for
  every type the sheet has a peak for. A tick is an SM clock, so the instance's per-clock rate is the sheet's, both
  ways.
* `tensorLong_sheet`: a tick's `tensorLong` units over a MAC's `weight`, at f_max, are at least the sheet's figure a
  second at the clock the sheet quotes it at (1830 MHz for s8, FP8, f16, bf16 and tf32, 1980 MHz for f64). So no
  format's long-window rate in the model is under the sheet's (θ at or above the sheet).

The long-window rate is set at or above anything the card might reach, so the check here fails a rate under the
sheet's, not one above it. The 1830 MHz cap's 499,712 a tick passes it with equality on the formats quoted at 1830 MHz
and fails it on f64,
which the sheet quotes at 1980 MHz: at weight 32, DMMA over a long window would get 15,616 MACs a tick against the
sheet's 16,896.
-/

open scoped NNRat

namespace Verity.Accel.H200

open Accounting.TS.Sheets.H200

/-- The instance's format for each of the sheet's tensor types. -/
def ofTy : Ty → Fmt
  | .s8 => .s8
  | .e4m3 => .e4m3
  | .e5m2 => .e5m2
  | .f16 => .f16
  | .bf16 => .bf16
  | .tf32 => .tf32
  | .f64 => .f64

/-- The clock the sheet quotes each type's figure at, in MHz. -/
def sheetMHz : Ty → ℝ
  | .f64 => fp64MHz
  | _ => tensorMHz

theorem tensor_sheet (t : Ty) : ((ρ .tensor / weight (ofTy t) : ℚ≥0) : ℝ) = smCount * t.peak := by
  cases t <;> norm_num [ρ, weight, ofTy, Ty.peak, smCount, peakS8, peakE4m3, peakE5m2, peakF16, peakBf16, peakTf32,
    peakF64]

theorem tensorLong_sheet (t : Ty) :
    smCount * t.peak * sheetMHz t ≤ ((ρ .tensorLong / weight (ofTy t) : ℚ≥0) : ℝ) * fmaxMHz := by
  cases t <;> norm_num [ρ, weight, ofTy, sheetMHz, Ty.peak, smCount, tensorMHz, fp64MHz, fmaxMHz, peakS8, peakE4m3,
    peakE5m2, peakF16, peakBf16, peakTf32, peakF64]

end Verity.Accel.H200
