module

public import Definitions.Pouw.PearlC.Game

/-! A unit's replayed debit under Pearl-C's accounting is nonnegative. -/

@[expose] public section

namespace Pouw.PearlC

theorem inertWorkOf_nonneg (p : Params) (c : Costs) (d : InertWork) : 0 ≤ inertWorkOf p c d := by
  unfold inertWorkOf
  split_ifs <;> positivity

end Pouw.PearlC
