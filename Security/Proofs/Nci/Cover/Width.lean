import Proofs.Nci.Cover.Additive

/-!
# The cover theorem: the width bound

With `Width f`, each use of a block's whole products costs `f` gates of its region (`whole_use_entry` makes the
intersection nonempty), so `f #J + b #L ≤ k`, and `4 b f #J #L ≤ k²`. The rest are at most `k`, as in the area bound.
-/

namespace Nci.Cover.SecurityProofs

theorem WidthIntensity : Guarantees.WidthIntensity := by
  intro C J L _ _ M A s k D f b hb hCH hW hU hk I hI
  refine finished_le_of_charge M A (c := f) hb (Nat.cast_nonneg f) hCH hU hI fun j hj => ?_
  obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hj
  obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hp
  have hne := whole_use_entry M A hb (updateHardness_of_conditional M A hCH) hU hk hI p.1
    (Finset.mem_image_of_mem _ hp)
  exact_mod_cast hW I hI.1.le p hpP hpI hne

theorem WidthCoverBound : Guarantees.WidthCoverBound := by
  intro C J L _ _ M A s k D f b 𝓘 hb hCH hW hU hk h𝓘 hcov
  exact cover_sum M hcov (by positivity) fun I hI => WidthIntensity C M A s k D f b hb hCH hW hU hk I (h𝓘 I hI)

end Nci.Cover.SecurityProofs
