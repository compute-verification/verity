import Proofs.Nci.Cover.Additive

/-!
# The cover theorem: the width bound

With `Width f`, each use of a block's whole products costs `f` gates of its region (`whole_use_entry` makes the
intersection nonempty), so `f #J + b #L ≤ k`, and `4 b f #J #L ≤ k²`. The rest are at most `k`, as in the area bound.
-/

namespace Nci.Cover.SecurityProofs

theorem WidthIntensity : Guarantees.WidthIntensity := by
  sorry

theorem WidthCoverBound : Guarantees.WidthCoverBound := by
  sorry

end Nci.Cover.SecurityProofs
