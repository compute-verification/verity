import Proofs.Nci.Cover.FanOutRootCost
import Proofs.Nci.Cover.Additive

/-!
# The fan-out bounds

The whole products of an inference block split by token. A token whose root the block holds spends at least `d − E`
values (`rootCost`, `lookupCost`), and its whole products number at most `c` per updated parameter. Any other token's
are at most `aOut` per value it spends (`charge`). The spent values of distinct tokens are distinct, so they number at
most the block's reads (`readSet_card_le`), or `ValueReadHardness` prices them at `f` bits each beside the parameters.
The rest is arithmetic: linear for `FanOutIntensity`, AM–GM (`four_mul_le_sq`) and the larger endpoint for
`FanOutReadIntensity`. The cover bounds are `cover_sum` on the intensities.
-/

namespace Nci.Cover.SecurityProofs

theorem FanOutIntensity : Guarantees.FanOutIntensity := by
  sorry

theorem FanOutCoverBound : Guarantees.FanOutCoverBound := by
  sorry

theorem FanOutReadIntensity : Guarantees.FanOutReadIntensity := by
  sorry

theorem FanOutReadCoverBound : Guarantees.FanOutReadCoverBound := by
  sorry

end Nci.Cover.SecurityProofs
