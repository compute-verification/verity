module

public import Definitions.Pouw.Assumptions.Dimension.H100Assumptions
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! `MH100`'s measured price table `h100Measured`, which does not bound the hardware from below (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

/-- **The measured H100 table** (draft semantics), each entry rounded down. It is a measurement, not a floor, and does
not bound the hardware from below (cite `h100LoopFree` or `h100Floor`). Its entries stay as measured so that no record
that reads it moves. -/
def h100Measured : Prices100 where
  mac := nominalMac
  tc := 32
  ffma := 33
  int32 := 64
  fp64 := 34
  other := 40

end Pouw.Dimension.H100
