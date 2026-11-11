module

public import Proofs.Flock.RecGeneric.AtFlockEvery
public import Proofs.Flock.RecGeneric.AtFlockRecovers
public import Proofs.Flock.RecGeneric.AtFlockTerms
public import Proofs.Flock.RecGeneric.AtFlockContract
public import Proofs.Flock.RecGeneric.AtFlockServer

/-!
# L1 at Flock: the recursive verifier's knowledge soundness at every statement

`Flock.Guarantees.RecursiveKSEvery` with no hypothesis: `recursiveKSEvery_of_recovers` (the deployment's premises of
`RecoversLinkOn`, and the lift from the dense statements to every statement) at F9a (`recoversLinkOn_of`), whose three
term comparisons are F6's (`forkPlansLeBinnerV`, `capPlanIsFCI`, `conflictPlanIsF0I`), and F7 (`contractDense`), the
last two reading F1 (`ivStarOfBound`, `openViewOfBound`) at the deployed bound record.
-/

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
