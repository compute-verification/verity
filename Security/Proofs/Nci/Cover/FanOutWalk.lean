import Proofs.Nci.Cover.FanOutPlan
import Proofs.Nci.Cover.Paths

/-!
# The fan-out proof: walks along keys, and what they charge

`walk` follows keys back from a held exit of a value; `readSet_card_le` bounds the values a block reads; `lookupCost`
bounds the lookups of one token a block holds; `charge` bounds a token's whole products by its spent values when the
block holds none of its roots.
-/

namespace Nci.Cover.SecurityProofs

theorem walk : FanOutPlan.Walk := by
  sorry

theorem readSet_card_le : FanOutPlan.ReadSetCardLe := by
  sorry

theorem lookupCost : FanOutPlan.LookupCost := by
  sorry

theorem charge : FanOutPlan.Charge := by
  sorry

end Nci.Cover.SecurityProofs
