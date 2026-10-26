import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.KS.PlainRConflict
import Proofs.Flock.Claims.Recursion.LastCallsR

/-!
# `RecursiveKSAt` from its open premises

`recursiveKSAt_of` is `ksLin2_deployedR_partAccepts` at the deployed join (`FlockDeploy.joinR`), with conflicts tied
to collisions at the plain record at the deployed reader (`plainR_conflictTied`) and both halves of its last-message
binding (`rColsBind_sha512`, `rCallsBind_sha512`) proved. What it takes: the table bound at the abort law at that
record (`htab`), `StepsLogFree` at its bound record (`hLog`), and `DecompB` at what each part's circuit accepts
(`hD`). `recursiveKSAt_of_decomp` takes that last one as `RecKSDecompR`, which `recKSDecompR_of_targets` proves from
the verdict's rows.
-/

namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

theorem recursiveKSAt_of (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainR _root_.Flock.Readable.relation rd) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep)
    (hLog : (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w).StepsLogFree FlockDeploy.sha512)
    (hD : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512
        (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.layoutBTops (RecDeploy.finOpen rest))
          (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
            RecDeploy.rowInDeployed final w).IVStar)
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
          RecDeploy.rowInDeployed final w).IVStar
        (RecDeploy.layoutBTops (RecDeploy.finOpen rest)) shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512
          (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
            (RecDeploy.layoutBTops (RecDeploy.finOpen rest))
            (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
              (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
              RecDeploy.rowInDeployed final w).IVStar)
          (RecDeploy.deployZkRec _root_.Flock.Readable.relation) (RecDeploy.layoutBTops (RecDeploy.finOpen rest)) shape
          (RecDeploy.showsPart RecDeploy.recRead { b with tr := RecDeploy.recTr aux nOf hd recRest })
          (RecDeploy.cmtTr RecDeploy.recRead))) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  RecDeploy.ksLin2_deployedR_partAccepts rd b aux nOf hd recRest rest wE FlockDeploy.joinR final w
    (FlockDeploy.KS.plainR_conflictTied _ rd) htab (RecDeploy.rColsBind_sha512 rd) (RecDeploy.rCallsBind_sha512 rd)
    hLog hD

/-- **`RecursiveKSAt` from `RecKSDecompR`**: `recursiveKSAt_of` with `DecompB` at the dense form as
`RecDeploy.RecKSDecompR`, V\*'s layout at the last part's pieces `rest` and the record past its link points
`recRest`. -/
theorem recursiveKSAt_of_decomp (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (htab : Claims.Recursion.TableBoundAbortLin FlockDeploy.sha512
      (FlockDeploy.deployPlainR _root_.Flock.Readable.relation rd) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false) FlockDeploy.KS.CpDep)
    (hLog : (RecDeploy.bdR rd wE final w).StepsLogFree FlockDeploy.sha512)
    (hD : RecDeploy.RecKSDecompR rd b aux nOf hd recRest rest wE final w) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of rd final rest wE w b aux nOf hd recRest htab hLog fun shape hs => by
    obtain rfl := List.mem_singleton.1 hs
    exact hD

end Flock.SecurityProofs
