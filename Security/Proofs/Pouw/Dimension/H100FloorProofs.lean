import Definitions.Pouw.Assumptions.Dimension.H100Floor
import Proofs.Pouw.Dimension.H100WProofs
import Proofs.Pouw.Dimension.H100LoopFreeProofs

/-!
# `MH100`'s floor table and the hardware citations: proofs

`h32Floor` is by `decide`. The two bounds are `distinctWritesMH100On` with `h32LoopFree` and `h32Floor`.
-/

namespace Pouw.Dimension.H100

/-- **`H32` holds at the floor table.** -/
theorem h32Floor : H32 h100Floor :=
  ⟨fun s => by cases s <;> decide, by decide⟩

theorem distinctWritesLoopFree : DistinctWritesLoopFree :=
  fun G F hF S fpSem L P T => distinctWritesMH100On h100LoopFree h32LoopFree G F hF S fpSem L P T

theorem distinctWritesFloor : DistinctWritesFloor :=
  fun G F hF S fpSem L P T => distinctWritesMH100On h100Floor h32Floor G F hF S fpSem L P T

end Pouw.Dimension.H100
