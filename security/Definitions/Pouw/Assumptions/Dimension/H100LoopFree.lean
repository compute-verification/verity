module

public import Definitions.Pouw.Assumptions.Dimension.H100Assumptions
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! `MH100`'s loop-free price table, a measurement and not a floor, whose bounds are additive-W1 work and not time (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

/-- **The loop-free H100 table** (draft semantics), each entry the least loop-free price measured in its class,
rounded down. It is a measurement, not a floor: `int32` and `other` are the least measured forms, and the floor is
`h100Floor`. -/
def h100LoopFree : Prices100 where
  mac := nominalMac
  tc := 32
  ffma := 32
  int32 := 34
  fp64 := 32
  other := 36

end Pouw.Dimension.H100
