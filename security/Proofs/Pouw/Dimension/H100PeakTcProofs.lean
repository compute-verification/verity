module

public import Definitions.Pouw.Assumptions.Dimension.H100PeakTc
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Dimension.H100WProofs

/-! The distinct-writes bound with no measurement at `tc`. -/

@[expose] public section

namespace Pouw.Dimension.H100

theorem floorTcPeakTc : FloorTc 32 8 h100PeakTc :=
  ⟨fun s => by cases s <;> decide, by decide, by decide, by decide, by decide, by decide⟩

theorem distinctWritesPeakTc : DistinctWritesPeakTc :=
  fun G F K hF S fpSem L P T₁ T₂ =>
    distinctWritesLegacy 32 8 h100PeakTc floorTcPeakTc (by decide) G F K hF S fpSem L P T₁ T₂

end Pouw.Dimension.H100
