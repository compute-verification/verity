import Proofs.Flock.Recursive.KS
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.DeployedBringsLam
import Proofs.Flock.Claims.Recursion.RestFinal
import Proofs.Flock.Claims.Recursion.RestOpenedSized

/-!
# `RecursiveKSAt` from chain B's leaves at the deployed brings

`recursiveKSAt_of_leaves_deployed` is `recursiveKSAt_of_leaves` at `RecDeploy.deployedB b`, whose setup is the
compiled verifier's own evaluations (`deployedSetup`: check 5's and the registered reads'), with `hset` discharged by
`resSetup_deployedB`, the last message's reading by RESTREAD's `finalReads` (`hF`) and `readsAt_elemR_closed` (`hRd`,
which also closes the cap tie's `sideCapR_leaf`). The other fields of the brings stay `b`'s.
-/

namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open RecDeploy RecDeploy.DecompBR RecDeploy.RestLow

/-- **`RecursiveKSAt` from chain B's leaves at the deployed brings**: `recursiveKSAt_of_leaves` at `deployedB b`, with
the setup's listing (`hset`) proved (`resSetup_deployedB`) and the last message's reading (`hF`, `hRd`) closed
(`finalReads`, `readsAt_elemR_closed`). Its other leaves are `recursiveKSAt_of_leaves`', at
`deployedB b`. -/
theorem recursiveKSAt_of_leaves_deployed (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hres : ResLower.ResLoweredPD DeployedR.rdR wE DeployedR.finalR w (bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead) resPD ownRD
      (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead) :
    Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w (deployedB b) aux nOf hd
      recRest :=
  recursiveKSAt_of_leaves wE w (deployedB b) aux nOf hd recRest key (RestRead.finalReads OpenRows.elemR)
    RestElem.readsAt_elemR_closed hres (resSetup_deployedB b)

/-- **`RecursiveKS` from chain B's residual leaf**: `recursiveKSAt_of_leaves_deployed` at the deployed weights, for every
brings and record, wherever the residual's lowering (`ResLoweredPD`, at each part's decision) holds there at some key. -/
theorem recursiveKS_of_resLowered
    (hres : ∀ (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      ∃ key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ,
        ResLower.ResLoweredPD DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
          (bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead) resPD
          ownRD (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead) :
    Flock.Guarantees.RecursiveKS := by
  intro b aux nOf hd recRest
  obtain ⟨key, h⟩ := hres b aux nOf hd recRest
  exact recursiveKSAt_of_leaves_deployed DeployedR.wER DeployedR.wR b aux nOf hd recRest key h

end Flock.SecurityProofs
