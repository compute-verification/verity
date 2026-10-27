import Proofs.Flock.Claims.Recursion.HresAsm

/-!
# `RecursiveKS`, proved

The recursive verifier's knowledge soundness at the deployed reader (`Flock.Guarantees.RecursiveKS`), from KS's residual
leaf at the deployed readers (`RecDeploy.ResLower.recursiveKS_deployed`, which takes no hypothesis).
-/

namespace Flock.SecurityProofs

/-- **Knowledge soundness of the recursive verifier at the deployed reader.** -/
theorem RecursiveKS : Flock.Guarantees.RecursiveKS :=
  RecDeploy.ResLower.recursiveKS_deployed

end Flock.SecurityProofs
