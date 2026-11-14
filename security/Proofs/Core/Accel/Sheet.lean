module

public import Definitions.Core.Accel.Card
public import Definitions.Accounting.Sheets.H200
public import Mathlib.Tactic.NormNum

/-!
# The H200's figures, and its tensor rates against its data sheet

`sheetCard` is the H200 SXM 141 GB's figures as a card, handed in as data: each field is the value of the card file's
keys it names. No guarantee reads it; it is the instance the honest lemma is closed at (`h200_cardWithinAccel`,
`Proofs/Core/Accel/Witness.lean`). Its tensor rates are held to the sheet's dense peaks (`Accounting.TS.Sheets.H200`,
A3), format by format.

* `tensor_sheet`: `sheetCard`'s MACs a second in each format (`ρ tensor` over the format's `weight`) are at least the
  sheet's figure a second at the clock the sheet quotes it at (1830 MHz for s8, FP8, f16, bf16 and tf32, 1980 MHz for
  f64). So no format's rate in `sheetCard` is under the sheet's (θ at or above the sheet).
* `tensor_sheet_clock`: off f64 they are the sheet's figures exactly, its per-SM-clock peaks times its SMs at the
  1830 MHz tensor clock ceiling `sheetCard` takes the tensor at; f64's weight is rounded down, so its rate is above the
  sheet's.
-/

@[expose] public section

open scoped NNRat

namespace Verity.Accel.H200

open Accounting.TS.Sheets.H200

/-- The card file's `clock.sm` in Hz: every resource's clock but the tensor's. -/
def sheetClock : ℚ≥0 := 1980 * 10 ^ 6

/-- The card file's `clock.tensor` in Hz: the tensor's clock. -/
def sheetTensorClock : ℚ≥0 := 1830 * 10 ^ 6

/-- **The H200 SXM 141 GB's figures as a card**: sizes in 32-bit words, rates a second, each the card file's keys
named beside it. -/
def sheetCard : Card where
  size
    | .reg => 132 * (65536 + 4494) -- `sm.count × (store.reg + store.reg_extra)`
    | .smem => 132 * 65536 -- `sm.count × store.smem / 4`
    | .l2 => 50 * 2 ^ 20 / 4 -- `store.l2 / 4`
    | .dram => 144 * 2 ^ 30 / 4 -- `store.dram_installed / 4`
  size_pos := by intro s; cases s <;> decide
  ρ
    | .tensor => 132 * 4096 * sheetTensorClock -- `sm.count × rate.tensor × clock.tensor`
    | .issue => 132 * 4 * sheetClock -- `sm.count × rate.issue × clock.sm`
    | .alu | .imad | .f64 => 132 * 64 * sheetClock -- `sm.count × rate.alu` (`rate.imad`, `rate.f64`) `× clock.sm`
    | .shfl | .smem => 132 * 32 * sheetClock -- `sm.count × rate.shfl` (`rate.smem`) `× clock.sm`
    | .sfu => 132 * 16 * sheetClock -- `sm.count × rate.sfu × clock.sm`
    | .l2 => 132 * 512 * sheetClock -- `sm.count × rate.l2 × clock.sm`
    | .dram => 1229184 * 10 ^ 6 -- `memory.bandwidth / 4`
  weight
    | .f16 | .bf16 => 2 -- `weight.f16`, `weight.bf16`
    | .tf32 => 4 -- `weight.tf32`
    | .f64 => 59 / 2 -- `weight.f64`
    | .b1 => 1 / 8 -- `weight.b1`
    | _ => 1 -- `weight.s8`, `weight.u8`, `weight.e4m3`, `weight.e5m2`, `weight.e2m1`, `weight.nvf4`, `weight.mxf4`
  exact := 31 -- `model.exact`
  text := 144 * 2 ^ 30 / 16 -- `store.dram_installed / model.instruction_bytes`

/-- The card's format for each of the sheet's tensor types. -/
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

theorem tensor_sheet_clock (t : Ty) (ht : t ≠ .f64) :
    ((sheetCard.ρ .tensor / sheetCard.weight (ofTy t) : ℚ≥0) : ℝ) = smCount * t.peak * (sheetMHz t * 10 ^ 6) := by
  cases t <;> first
    | exact absurd rfl ht
    | norm_num [sheetCard, sheetTensorClock, ofTy, sheetMHz, Ty.peak, smCount, tensorMHz, peakS8, peakE4m3, peakE5m2,
        peakF16, peakBf16, peakTf32]

theorem tensor_sheet (t : Ty) :
    smCount * t.peak * (sheetMHz t * 10 ^ 6) ≤ ((sheetCard.ρ .tensor / sheetCard.weight (ofTy t) : ℚ≥0) : ℝ) := by
  cases t <;> norm_num [sheetCard, sheetTensorClock, ofTy, sheetMHz, Ty.peak, smCount, tensorMHz, fp64MHz, peakS8,
    peakE4m3, peakE5m2, peakF16, peakBf16, peakTf32, peakF64]

end Verity.Accel.H200
