import Models.Pouw.Assumptions.Dimension.H100LoopFree
import Models.Pouw.Guarantees.Dimension

/-!
# `MH100`'s loop-free table satisfies `H32`: proof
-/

namespace Pouw.Dimension.H100

/-- **`H32` holds at the loop-free prices** (Round 8). -/
theorem h32LoopFree : H32 h100LoopFree :=
  ⟨fun s => by cases s <;> decide, by decide⟩

end Pouw.Dimension.H100

theorem Pouw.SecurityProofs.Dimension.H100.H32LoopFree : Pouw.Guarantees.Dimension.H100.H32LoopFree := @Pouw.Dimension.H100.h32LoopFree
