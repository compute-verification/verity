import Proofs.Flock.Recursive.KSCoinTreeStatement
import Proofs.Flock.Recursive.KSProved
import Proofs.Flock.Claims.Recursion.DeployNil

/-!
# `RecursiveKSDeployed`, proved

`RecursiveKS` at Hello-committed coins (`Flock.Guarantees.RecursiveKSDeployed`): `Claims.Recursion.ksLin2OnC_of_ksLin2On`
at `RecursiveKS`, whose compiled verifier answers each round with its coin's bytes or with nothing
(`RecDeploy.answersCoinOrNil_V`), and the bytes determine the coin (`Claims.Bridge.coin_bytes_injective`).
-/

namespace Flock.SecurityProofs

/-- **Knowledge soundness of the recursive verifier at Hello-committed coins.** -/
theorem RecursiveKSDeployed : Flock.Guarantees.RecursiveKSDeployed := by
  intro δ₁ b aux nOf hd recRest
  refine Claims.Recursion.ksLin2OnC_of_ksLin2On (RecursiveKS b aux nOf hd recRest) FlockDeploy.Coin.bytes
    Claims.Bridge.coin_bytes_injective (fun x => ?_) δ₁
  exact RecDeploy.answersCoinOrNil_V _ _ _ x

end Flock.SecurityProofs
