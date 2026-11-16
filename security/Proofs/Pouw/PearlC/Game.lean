module

public import Definitions.Pouw.PearlC.Game

/-!
# Pearl-C's accounting (proofs)

A unit's replayed debit is nonnegative (`inertWorkOf_nonneg`).
-/

@[expose] public section

namespace Pouw.PearlC

theorem inertWorkOf_nonneg (p : Params) (c : Costs) (d : InertWork) : 0 ≤ inertWorkOf p c d := by
  unfold inertWorkOf
  split_ifs <;> positivity

end Pouw.PearlC
