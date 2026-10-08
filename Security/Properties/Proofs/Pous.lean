import Properties.Pous

/-!
# Proofs of the PoUS setup check's properties (`Properties.Pous`)

The registered point's numbers, by reduction (`rfl`, `decide`). The rest of `Properties.Pous` is open.
-/

namespace PousCheck.Properties.SecurityProofs

set_option exponentiation.threshold 20000 in
theorem deployed : Deployed := by
  unfold Deployed
  exact ⟨rfl, rfl, rfl, by decide, by decide, by decide, rfl, rfl, by decide⟩

end PousCheck.Properties.SecurityProofs
