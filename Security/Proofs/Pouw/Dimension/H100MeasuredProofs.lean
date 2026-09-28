import Models.Pouw.Assumptions.Dimension.H100Measured
import Models.Pouw.Guarantees.Dimension

/-!
# `MH100`'s measured table satisfies `H32`: proof
-/

namespace Pouw.Dimension.H100

/-- **`H32` holds at the measured prices** (Round 7). -/
theorem h32Measured : H32 h100Measured :=
  ⟨fun s => by cases s <;> decide, by decide⟩

end Pouw.Dimension.H100

theorem Pouw.SecurityProofs.Dimension.H100.H32Measured : Pouw.Guarantees.Dimension.H100.H32Measured := @Pouw.Dimension.H100.h32Measured
