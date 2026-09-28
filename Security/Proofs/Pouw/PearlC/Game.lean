import Definitions.Pouw.PearlC.Game

/-!
# Pearl-C's accounting (proofs)

A unit's replayed debit is nonnegative (`debitOf_nonneg`).
-/

namespace Pouw.PearlC

theorem debitOf_nonneg (p : Params) (c : Costs) (d : Debit) : 0 ≤ debitOf p c d := by
  unfold debitOf
  split_ifs <;> positivity

end Pouw.PearlC
