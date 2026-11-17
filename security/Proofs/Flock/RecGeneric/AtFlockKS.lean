module

public import Proofs.Flock.RecGeneric.AtFlockEvery
public import Proofs.Flock.RecGeneric.AtFlockRecovers
public import Proofs.Flock.RecGeneric.AtFlockTerms
public import Proofs.Flock.RecGeneric.AtFlockContract
public import Proofs.Flock.RecGeneric.AtFlockServer

/-! L1 at Flock: the recursive verifier's knowledge soundness at every statement, `Flock.Guarantees.RecursiveKSEvery`, with no hypothesis. -/

@[expose] public section

namespace Flock.SecurityProofs

open Claims.RecGeneric

/-- **Knowledge soundness of the recursive verifier at every statement, at the deployed reader.** -/
theorem RecursiveKSEvery : Flock.Guarantees.RecursiveKSEvery :=
  recursiveKSEvery_of_recovers
    (fun _ _ _ _ _ => recoversLinkOn_of _ _ _ _ _ _ (forkPlansLeBinnerV _ _ _ _)
      (capPlanIsFCI _ _ _ _ (ivStarOfBound _)) (conflictPlanIsF0I _ _ _ (ivStarOfBound _)))
    (contractDense (ivStarOfBound _) (openViewOfBound _))

end Flock.SecurityProofs
