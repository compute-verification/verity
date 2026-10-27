import Definitions.Core.Accel.H200
import Definitions.Accounting.Sheets.H200
import Mathlib.Tactic.NormNum

/-!
# The H200's tensor rates against its data sheet

The H200's table, held to the sheet's dense peaks (`Accounting.TS.Sheets.H200`, A3), format by format.

* `tensor_sheet`: the table's MACs a second in each format (`ρ tensor` over the format's `weight`) are at least the
  sheet's figure a second at the clock the sheet quotes it at (1830 MHz for s8, FP8, f16, bf16 and tf32, 1980 MHz for
  f64). So no format's rate in the table is under the sheet's (θ at or above the sheet).
* `tensor_sheet_clock`: at the table's clock they are the sheet's per-SM-clock peaks times its SMs, both ways.
-/

open scoped NNRat

namespace Verity.Accel.H200

open Accounting.TS.Sheets.H200

/-- The table's format for each of the sheet's tensor types. -/
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

theorem tensor_sheet_clock (t : Ty) :
    ((card.ρ .tensor / card.weight (ofTy t) : ℚ≥0) : ℝ) = smCount * t.peak * (fmaxMHz * 10 ^ 6) := by
  cases t <;> norm_num [card, clock, ofTy, Ty.peak, smCount, fmaxMHz, peakS8, peakE4m3, peakE5m2, peakF16, peakBf16,
    peakTf32, peakF64]

theorem tensor_sheet (t : Ty) :
    smCount * t.peak * (sheetMHz t * 10 ^ 6) ≤ ((card.ρ .tensor / card.weight (ofTy t) : ℚ≥0) : ℝ) := by
  cases t <;> norm_num [card, clock, ofTy, sheetMHz, Ty.peak, smCount, tensorMHz, fp64MHz, peakS8, peakE4m3,
    peakE5m2, peakF16, peakBf16, peakTf32, peakF64]

end Verity.Accel.H200
