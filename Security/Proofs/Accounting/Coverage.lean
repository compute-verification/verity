import Definitions.Accounting.Guarantees.Coverage
import Proofs.Core.Game.Prob

/-! `Nci.SecurityProofs.CapabilityFloor : Nci.Guarantees.CapabilityFloor`: below `min Lc Tmax`, a computation in `𝒜`
has no covered structure (each costs at least `Lc`), so its success there is among coverage's uncovered successes below
`Tmax`. -/

/-- **Covered success below the floor is rare** (`Nci.Guarantees.CapabilityFloor`). -/
theorem Nci.SecurityProofs.CapabilityFloor : Nci.Guarantees.CapabilityFloor := by
  intro In L m G 𝒜 Core cost Tmax Lc η hcov hfloor C pos
  refine le_trans (Verity.Game.prob_mono (fun p hp => ?_) _ _) (hcov C pos)
  obtain ⟨hA, hS, hc⟩ := hp
  exact ⟨hA, hS, lt_of_lt_of_le hc (min_le_right _ _),
    fun hCore => absurd (hfloor C p.1 hA hCore) (not_le.2 (lt_of_lt_of_le hc (min_le_left _ _)))⟩
