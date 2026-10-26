import Proofs.Flock.Recursive.KS
import Proofs.Flock.Claims.Recursion.DeployedBrings

/-!
# `RecursiveKSAt` from chain B's leaves at the deployed brings

`recursiveKSAt_of_leaves_deployed` is `recursiveKSAt_of_leaves` at `RecDeploy.deployedB b`, whose setup is the
compiled verifier's own evaluations (`deployedSetup`: check 5's and the registered reads'), with `hset` discharged by
`resSetup_deployedB`. The other fields of the brings stay `b`'s.
-/

namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open RecDeploy RecDeploy.DecompBR RecDeploy.RestLow

/-- **`RecursiveKSAt` from chain B's leaves at the deployed brings**: `recursiveKSAt_of_leaves` at `deployedB b`, with
the setup's listing (`hset`) proved (`resSetup_deployedB`). Its other leaves are `recursiveKSAt_of_leaves`', at
`deployedB b`. -/
theorem recursiveKSAt_of_leaves_deployed (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (hPOk : PartsOkD)
    (hEJ : Top.RowEJoint DeployedR.rdR key FlockDeploy.sha512
      (recL DeployedR.rdR (deployedB b) aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      (bdR DeployedR.rdR wE DeployedR.finalR w) (DeployedR.layR OpenRows.elemR) Claims.Recursion.Form.Dense
      (acceptsL DeployedR.rdR (deployedB b) aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR)))
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon)
    (hF : RestRead.FinalReads DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hRd : RestRead.ReadsAt DeployedR.rdR DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hL : ResListedD) (hQ : OpenOwned key)
    (hres : ResLowJ DeployedR.rdR (deployedB b) aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR)
      resPD (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead ownRD)
    (hCap : CapLower.SideCapR FlockDeploy.sha512
      (recL DeployedR.rdR (deployedB b) aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB (deployedB b) aux nOf hd recRest) (cmtTr recRead)
      (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) DeployedR.rdR (Rows.rowDPieces key) Rows.posAt) :
    Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w (deployedB b) aux nOf hd
      recRest :=
  recursiveKSAt_of_leaves wE w (deployedB b) aux nOf hd recRest key hPOk hEJ S hF hRd hL hQ hres
    (resSetup_deployedB b) hCap

end Flock.SecurityProofs
