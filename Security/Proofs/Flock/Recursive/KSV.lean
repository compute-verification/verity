import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.PartAcceptsSetup
import Proofs.Flock.Claims.Recursion.RegSetupOpens

/-!
# `RecursiveKSAt` from `DecompBV`

`recursiveKSAt_ofV` is `recursiveKSAt_of` with `DecompBP` replaced by `DecompBV` at `PartAcceptsV`: each part's check
read at the run's V\* statement for it, and the collision outcome over the recorded calls and the compiled verifier's
own evaluations at the run's V\* statements (`setupCalls`), which list the registered reads' leaves and climbs wherever
the brings do (`RecDeploy.RegSetup`, `RecDeploy.regListed_recBound`). `DecompBP` gives `DecompBV`
(`decompBV_of_decompBP`), so this asks no more; a lowering under `JointV` may read two parts' commit strings as one
through the setup's openings (`RecDeploy.bc_eq_of_setup`).
-/

namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

theorem recursiveKSAt_ofV (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : RecDeploy.MoreSlotsTrue (RecDeploy.deployZkRec _root_.Flock.Readable.relation) b.st)
    (hD : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBV FlockDeploy.sha512
        (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR rest)
          (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
            RecDeploy.rowInDeployed final w).IVStar)
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
          RecDeploy.rowInDeployed final w).IVStar
        (RecDeploy.DeployedR.layR rest) shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512
          (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
            (RecDeploy.DeployedR.layR rest)
            (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
              (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
              RecDeploy.rowInDeployed final w).IVStar)
          (RecDeploy.deployZkRec _root_.Flock.Readable.relation) (RecDeploy.DeployedR.layR rest) shape
          (RecDeploy.showsPart RecDeploy.recRead { b with tr := RecDeploy.recTr aux nOf hd recRest })
          (RecDeploy.cmtTr RecDeploy.recRead))) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  RecDeploy.ksLin2_deployedR_layRV rd b aux nOf hd recRest rest wE FlockDeploy.joinR final w hIn hM hD

end Flock.SecurityProofs
