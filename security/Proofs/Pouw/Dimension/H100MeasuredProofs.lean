module

public import Definitions.Pouw.Assumptions.Dimension.H100Measured
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# `MH100`'s measured table satisfies `H32`: proof
-/

@[expose] public section

namespace Pouw.Dimension.H100

/-- **`H32` holds at the measured prices** (Round 7). -/
theorem h32Measured : H32 h100Measured :=
  ⟨fun s => by cases s <;> decide, by decide⟩

end Pouw.Dimension.H100
