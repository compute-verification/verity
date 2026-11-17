module

public import Definitions.Pouw.Assumptions.Dimension.H100LoopFree
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! `MH100`'s loop-free table satisfies `H32`. -/

@[expose] public section

namespace Pouw.Dimension.H100

theorem h32LoopFree : H32 h100LoopFree :=
  ⟨fun s => by cases s <;> decide, by decide⟩

end Pouw.Dimension.H100
