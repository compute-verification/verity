import Proofs.Nci.Cover.PartialSumCube

/-!
# The partial-sum bounds (WORK IN PROGRESS: `sorry` stubs)

`PartialSumsRead` is proved in `PartialSumRead` and `PartialSumIntensity` in `PartialSumCube`. The cover bounds and
`KeyedActivationReads` are `sorry` stubs here, which later commits on this branch replace with their proofs.
-/

namespace Nci.Cover.SecurityProofs

theorem PartialSumCoverBound : Guarantees.PartialSumCoverBound := by
  sorry

theorem KeyedActivationReads : Guarantees.KeyedActivationReads := by
  sorry

theorem PartialSumKeyedCoverBound : Guarantees.PartialSumKeyedCoverBound := by
  sorry

end Nci.Cover.SecurityProofs
