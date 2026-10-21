import Proofs.Nci.Cover.PartialSumCover
import Proofs.Nci.Cover.PartialSumKeyed

/-!
# The partial-sum bounds

`PartialSumsRead` is proved in `PartialSumRead`, `PartialSumIntensity` in `PartialSumCube`, `PartialSumCoverBound` in
`PartialSumCover` and `KeyedActivationReads` in `PartialSumKeyed`. `PartialSumKeyedCoverBound` is the cover bound at
`f = 1` with `KeyedActivationReads` for its premise.
-/

namespace Nci.Cover.SecurityProofs

theorem PartialSumKeyedCoverBound : Guarantees.PartialSumKeyedCoverBound := by
  intro C J L Tok Val Mat Row Col _ _ _ _ _ _ M d aOut c V w K Q s k E n b bcol μ 𝓘 hb hμ0 hμ1 hH hCI hk hEd hn hW
    h𝓘 hcov
  have h := PartialSumCoverBound C M d aOut c V w K Q s k 1 (min (d - E) n) b μ 𝓘 hb
    (lt_min (Nat.sub_pos_of_lt hEd) hn) hμ0 hμ1 hH
    (KeyedActivationReads C M d aOut c V w K Q s k E n bcol hCI hk hW) h𝓘 hcov
  rwa [Nat.cast_one] at h

end Nci.Cover.SecurityProofs
