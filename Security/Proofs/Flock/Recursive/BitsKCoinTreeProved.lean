module

public import Proofs.Flock.Recursive.BitsKCoinTreeStatement
public import Proofs.Flock.Claims.Recursion.DeployNil

/-!
# `RecursiveBitsKDeployed`, proved

`RecDeploy.recursiveBitsK_deployed` at Hello-committed coins (`Flock.Guarantees.RecursiveBitsKDeployed`):
`Claims.Recursion.bitsKAtC_of_bitsKAt` at `recursiveBitsK_deployed`, whose compiled verifier answers each round with its
coin's bytes or with nothing (`RecDeploy.answersCoinOrNil_V`), and the bytes determine the coin
(`Claims.Bridge.coin_bytes_injective`).
-/

@[expose] public section

namespace Flock.SecurityProofs

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at Hello-committed coins.** -/
theorem RecursiveBitsKDeployed : Flock.Guarantees.RecursiveBitsKDeployed := by
  intro δ₁ b aux nOf hd recRest
  refine Claims.Recursion.bitsKAtC_of_bitsKAt (RecDeploy.recursiveBitsK_deployed b aux nOf hd recRest)
    FlockDeploy.Coin.bytes Claims.Bridge.coin_bytes_injective (fun x => ?_) δ₁
  exact RecDeploy.answersCoinOrNil_V _ _ _ x

end Flock.SecurityProofs
