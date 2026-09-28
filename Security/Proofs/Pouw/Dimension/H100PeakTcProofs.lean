import Models.Pouw.Assumptions.Dimension.H100PeakTc
import Proofs.Pouw.Dimension.H100WProofs

/-!
# The citation with no measurement at `tc`: proofs

`floorTcPeakTc` is by `decide`, and `distinctWritesPeakTc` instantiates `distinctWritesLegacy` at `a = 32`, `c = 8`.
-/

namespace Pouw.Dimension.H100

/-- **`h100PeakTc` meets `FloorTc 32 8`.** -/
theorem floorTcPeakTc : FloorTc 32 8 h100PeakTc :=
  ⟨fun s => by cases s <;> decide, by decide, by decide, by decide, by decide, by decide⟩

/-- **The distinct-writes bound with no measurement at `tc`.** -/
theorem distinctWritesPeakTc : DistinctWritesPeakTc :=
  fun G F K hF S fpSem L P T₁ T₂ =>
    distinctWritesLegacy 32 8 h100PeakTc floorTcPeakTc (by decide) G F K hF S fpSem L P T₁ T₂

end Pouw.Dimension.H100
