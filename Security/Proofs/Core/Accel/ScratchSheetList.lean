import Definitions.Core.Accel.H200
import Definitions.Accounting.HeadlineDc.Sm90
import Mathlib.Tactic.NormNum

/-!
Scratch (CMP-400 plan, never landed): `Proofs/Core/Accel/Sheet.lean` re-pointed at a sheet in the list encoding
(`SheetDataQ`), the encoding compute-accounting's plan generates one sheet per card in. No generated sheet exists
yet, so this reads `main`'s one list-encoded H200 sheet, `Accounting.TS.Headline.Sm90.ds90`.

What carries over: each format's weight in the card's table is the sheet's s8 peak over the format's own peak, exact
off f64 and rounded below it at f64. What does not: the list encoding has no SM count and one clock (`fmax`, the SM
clock), so `Sheet.lean`'s two statements about MACs a second cannot be stated against it, and at `fmax` the
comparison goes the other way.
-/

open scoped NNRat

namespace Verity.Accel.H200

open Accounting.TS.Headline

/-- The list-encoded sheet's dense MACs an SM clock of class `k`'s operand type. Classes 23, 37, 39, 13, 11, 15 and 5
are `IGMMA_S8_N256`, `QGMMA_E4M3_N256`, `QGMMA_E5M2_N256`, `HGMMA_F16_N256`, `HGMMA_BF16_N256`, `HGMMA_TF32_N256` and
`DMMA16816`. -/
def peakAt (k : ℕ) : ℕ := Sm90.ds90.peak.getD k 0

theorem peaks :
    peakAt 23 = 4096 ∧ peakAt 37 = 4096 ∧ peakAt 39 = 4096 ∧ peakAt 13 = 2048 ∧ peakAt 11 = 2048 ∧
      peakAt 15 = 1024 ∧ peakAt 5 = 128 := by
  decide

/-- **The re-pointed form**: the table's weights are the sheet's peak ratios, exact off f64; f64's weight is below
the ratio. -/
theorem weight_sheet_list :
    ((card.weight .s8 : ℚ≥0) : ℝ) * (peakAt 23 : ℝ) = (peakAt 23 : ℝ) ∧
      ((card.weight .e4m3 : ℚ≥0) : ℝ) * (peakAt 37 : ℝ) = (peakAt 23 : ℝ) ∧
      ((card.weight .e5m2 : ℚ≥0) : ℝ) * (peakAt 39 : ℝ) = (peakAt 23 : ℝ) ∧
      ((card.weight .f16 : ℚ≥0) : ℝ) * (peakAt 13 : ℝ) = (peakAt 23 : ℝ) ∧
      ((card.weight .bf16 : ℚ≥0) : ℝ) * (peakAt 11 : ℝ) = (peakAt 23 : ℝ) ∧
      ((card.weight .tf32 : ℚ≥0) : ℝ) * (peakAt 15 : ℝ) = (peakAt 23 : ℝ) ∧
      ((card.weight .f64 : ℚ≥0) : ℝ) * (peakAt 5 : ℝ) ≤ (peakAt 23 : ℝ) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := peaks
  rw [h1, h2, h3, h4, h5, h6, h7]
  norm_num [card]

/-- **What is lost**: the list encoding has one clock, `fmax`, and no SM count. At `fmax` and the table's own 132
SMs, the sheet's s8 figure is above the table's tensor rate, so `tensor_sheet` does not carry over. -/
theorem tensor_below_at_fmax :
    ((card.ρ .tensor / card.weight .s8 : ℚ≥0) : ℝ) <
      132 * (peakAt 23 : ℝ) * ((Sm90.ds90.fmax : ℝ) * 10 ^ 6) := by
  rw [peaks.1]
  norm_num [card, tensorClock, Sm90.ds90]

end Verity.Accel.H200

#print axioms Verity.Accel.H200.weight_sheet_list
#print axioms Verity.Accel.H200.tensor_below_at_fmax
