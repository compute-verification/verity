import Definitions.Pouw.Assumptions.Dimension.H100Floor
import Definitions.Pouw.Assumptions.Dimension.H200Measured

/-!
# `MH100`'s H200 table satisfies `H32`, and is the floor table: proof
-/

namespace Pouw.Dimension.H100

/-- **`H32` holds at the H200's measured prices**, with no slack. -/
theorem h32H200Measured : H32 h200Measured :=
  ⟨fun s => by cases s <;> decide, by decide⟩

/-- **The H200's measured table is the floor table**: every class at its bound. -/
theorem h200MeasuredFloor : h200Measured = h100Floor := rfl

end Pouw.Dimension.H100
