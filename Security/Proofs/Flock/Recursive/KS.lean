import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.NonVacuous
import Proofs.Flock.Claims.KS.PlainRConflict
import Proofs.Flock.Claims.KS.PlainAccPrR
import Proofs.Flock.Claims.KS.PlainRCapPrefix
import Proofs.Flock.Claims.Recursion.LastCallsR
import Proofs.Flock.Claims.Recursion.KSLin2LayR
import Proofs.Flock.Claims.Recursion.RestLowR
import Proofs.Flock.Claims.Recursion.RestVerdictBasesHold
import Proofs.Flock.Claims.KS.PlainSimR

/-!
# `RecursiveKSAt` from its open premises

`recursiveKSAt_of` is `ksLin2_deployedR_layRP` at the deployed join (`FlockDeploy.joinR`), at the layout of record
`DeployedR.layR rest`. At the plain record at the deployed reader, conflicts are tied to collisions
(`plainR_conflictTied`) and the table bound at the abort law is `plainR_tableBoundAbortLin_R`, from `PlainRdInR` alone
(`hIn`; the cap prefix is `capPrefixR`, proved with no hypotheses). Its last-message binding is `rColsBind_sha512` and
`rCallsBind_sha512`, and it needs no `StepsLogFree` (the finders search the inner verifier's calls at each draw's
coins). It also takes the extra calls `layR` records true at the outer relation (`hM`, `MoreSlotsTrue`) and `DecompBP`
at what each part's circuit accepts at its own witness `ws[s]` (`hD`, `PartAcceptsP`). `recursiveKSAt_of_decomp` takes
`DecompB` at `PartAccepts` instead, as `RecKSDecompL` at `layR`, which pins (`decompBP_of_decompB`), and
`recursiveKSAt_of_leaves` proves that from chain B's leaves (`RestLow.recKSDecompD_of_leavesJ`).
-/


namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

theorem recursiveKSAt_of (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : RecDeploy.MoreSlotsTrue (RecDeploy.deployZkRec _root_.Flock.Readable.relation) b.st)
    (hD : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBP FlockDeploy.sha512
        (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR rest)
          (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
            RecDeploy.rowInDeployed final w).IVStar)
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
          RecDeploy.rowInDeployed final w).IVStar
        (RecDeploy.DeployedR.layR rest) shape
        (Claims.Recursion.VStarB.PartAcceptsP FlockDeploy.sha512
          (RecDeploy.recBound _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
            (RecDeploy.DeployedR.layR rest)
            (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
              (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
              RecDeploy.rowInDeployed final w).IVStar)
          (RecDeploy.deployZkRec _root_.Flock.Readable.relation) (RecDeploy.DeployedR.layR rest) shape
          (RecDeploy.showsPart RecDeploy.recRead { b with tr := RecDeploy.recTr aux nOf hd recRest })
          (RecDeploy.cmtTr RecDeploy.recRead))) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  RecDeploy.ksLin2_deployedR_layRP rd b aux nOf hd recRest rest wE FlockDeploy.joinR final w hIn hM hD

/-- **`RecursiveKSAt` from `RecKSDecompL` at the layout of record**: `recursiveKSAt_of` with `DecompB` at the dense
form as `RecDeploy.DecompBR.RecKSDecompL` at `DeployedR.layR rest`, the last part's pieces `rest` and the record
past its link points `recRest`. -/
theorem recursiveKSAt_of_decomp (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : RecDeploy.MoreSlotsTrue (RecDeploy.deployZkRec _root_.Flock.Readable.relation) b.st)
    (hD : RecDeploy.DecompBR.RecKSDecompL rd b aux nOf hd recRest wE final w (RecDeploy.DeployedR.layR rest)) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of rd final rest wE w b aux nOf hd recRest hIn hM fun shape hs => by
    obtain rfl := List.mem_singleton.1 hs
    exact Claims.Recursion.VStarB.decompBP_of_decompB hD

section Leaves

open RecDeploy RecDeploy.DecompBR RecDeploy.RestLow

/-- **`RecursiveKSAt` from chain B's leaves**: `recursiveKSAt_of_decomp` with `RecKSDecompL` at the layout of record
from `recKSDecompD_of_leavesJ`, at the decision's positions `Rows.posAt`. Every hypothesis is a named open leaf: row
R's target (`hRdep`; row D's is RECL's `targetDMerkle_PD`, row E's `targetE_accRC`), row E's chain at the joint
witnesses (`hEJ`, RECE's `RowEJoint`), the JSON parser's spec (`S`, which also gives the base checks,
`basesHold_spec`), the last message's reading (`hF`, `hRd`), the rounds no part holds (`hSRJ`), the pieces' owners
(`hOwn`) and lowering (`hres`), the copies' openings and directions (`hO`, `hDirs`), the caps (`hOut`, `hCap`) and the
extra calls (`hM`). The reader's bound is proved (`plainRdInR`). -/
theorem recursiveKSAt_of_leaves (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (elemAt : Msg → List Msg → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (ownR : Msg → ResPiece → Option ℕ)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation) b.st)
    (hRdep : Top.TargetRDep rd (PD rd resP key elemAt vvAt (OpenLower.sideAtR rd) Rows.posAt) (Top.accRC rd key))
    (hEJ : Top.RowEJoint rd key FlockDeploy.sha512 (recL rd b aux nOf hd recRest wE final w (DeployedR.layR rest))
      (bdR rd wE final w) (DeployedR.layR rest) Claims.Recursion.Form.Dense
      (acceptsL rd b aux nOf hd recRest wE final w (DeployedR.layR rest)))
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon)
    (hF : RestRead.FinalReads final (DeployedR.restR rest))
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR rest))
    (hSRJ : SaltRowsJ rd wE final w (DeployedR.layR rest)) (hOwn : PiecesOwned resP key ownR)
    (hres : ResLowJ rd b aux nOf hd recRest wE final w (DeployedR.layR rest) resP elemAt vvAt ownR)
    (hO : OpenLower.CopyOpensR FlockDeploy.sha512 (recL rd b aux nOf hd recRest wE final w (DeployedR.layR rest))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR rest)
      (bdR rd wE final w) (OpenLower.sideAtR rd) (Rows.rowDPieces key))
    (hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512
      (recL rd b aux nOf hd recRest wE final w (DeployedR.layR rest))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR rest)
      (bdR rd wE final w) (Rows.rowDPieces key) Rows.posAt)
    (hOut : CapLower.OutBindR FlockDeploy.sha512 (recL rd b aux nOf hd recRest wE final w (DeployedR.layR rest))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR rest)
      (bdR rd wE final w) (Rows.rowDPieces key) Rows.posAt)
    (hCap : CapLower.SideCapR FlockDeploy.sha512 (recL rd b aux nOf hd recRest wE final w (DeployedR.layR rest))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR rest)
      (bdR rd wE final w) rd (Rows.rowDPieces key) Rows.posAt) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of_decomp rd final rest wE w b aux nOf hd recRest (fun x _ => FlockDeploy.KS.plainRdInR rd x) hM
    (recKSDecompD_of_leavesJ rd b aux nOf hd recRest rest wE final w resP key elemAt vvAt ownR hRdep hEJ S
      (basesHold_spec S rd) hF hRd hSRJ hOwn hres hO hDirs hOut hCap)

end Leaves

end Flock.SecurityProofs
