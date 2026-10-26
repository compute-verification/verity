import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.KS.PlainRConflict
import Proofs.Flock.Claims.KS.PlainAccPrR

/-!
# `RecursiveKSAt` from its open premises

`recursiveKSAt_of` is `ksLin2_deployedR_partAccepts` at the deployed join (`FlockDeploy.joinR`). At the plain record at
the deployed reader, conflicts are tied to collisions (`plainR_conflictTied`) and the table bound at the abort law is
`tableBoundAbortLin_VR`, from the reader's accepting last (`accLast_VR`), its free decision (`decideFree_VR`) and its
three column facts: every read row inside the reader's columns (`rdAllAt_R` from `PlainRdInR`, `hIn`), none past the
schedule's top level (`htopL`), and `CapHitAt` (`hcap`). It also takes the reader's columns half and calls half
of its last-message binding (`hC`, `hK`), `StepsLogFree` at its bound record (`hLog`), and `DecompB` at what each
part's circuit accepts (`hD`). `recursiveKSAt_of_decomp` takes that last one as `RecKSDecompR`, which
`recKSDecompR_of_targets` proves from the verdict's rows.
-/

namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

theorem recursiveKSAt_of (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.KS.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (htopL : FlockDeploy.KS.LevelTopAt (FlockDeploy.KS.VR rd) fun _ => FlockDeploy.KS.plainRawLevelColsRV rd)
    (hcap : FlockDeploy.KS.CapHitAt (FlockDeploy.KS.VR rd) fun _ => FlockDeploy.KS.plainRawLevelColsRV rd)
    (hC : RecDeploy.RColsBind FlockDeploy.sha512 rd) (hK : RecDeploy.RCallsBind FlockDeploy.sha512 rd)
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
    (FlockDeploy.KS.plainR_conflictTied _ rd)
    (FlockDeploy.KS.tableBoundAbortLin_VR rd _ rfl (fun x => FlockDeploy.KS.εtable_nonneg false x)
      (fun _ => FlockDeploy.KS.plainRawLevelColsRV rd) (FlockDeploy.KS.accLast_VR rd) (FlockDeploy.KS.decideFree_VR rd)
      (FlockDeploy.KS.rdAllAt_R rd hIn) htopL hcap)
    hC hK hLog hD

/-- **`RecursiveKSAt` from `RecKSDecompR`**: `recursiveKSAt_of` with `DecompB` at the dense form as
`RecDeploy.RecKSDecompR`, V\*'s layout at the last part's pieces `rest` and the record past its link points
`recRest`. -/
theorem recursiveKSAt_of_decomp (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.KS.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (htopL : FlockDeploy.KS.LevelTopAt (FlockDeploy.KS.VR rd) fun _ => FlockDeploy.KS.plainRawLevelColsRV rd)
    (hcap : FlockDeploy.KS.CapHitAt (FlockDeploy.KS.VR rd) fun _ => FlockDeploy.KS.plainRawLevelColsRV rd)
    (hC : RecDeploy.RColsBind FlockDeploy.sha512 rd) (hK : RecDeploy.RCallsBind FlockDeploy.sha512 rd)
    (hLog : (RecDeploy.bdR rd wE final w).StepsLogFree FlockDeploy.sha512)
    (hD : RecDeploy.RecKSDecompR rd b aux nOf hd recRest rest wE final w) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of rd final rest wE w b aux nOf hd recRest hIn htopL hcap hC hK hLog fun shape hs => by
    obtain rfl := List.mem_singleton.1 hs
    exact hD

end Flock.SecurityProofs
