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
import Proofs.Flock.Claims.Recursion.SlotsPorted
import Proofs.Flock.Claims.Recursion.MoreSlotsAll
import Proofs.Flock.Claims.Recursion.VerdictRowRLeaf
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.OpenDirsDeployed
import Proofs.Flock.Claims.Recursion.InnerAtCoins
import Proofs.Flock.Claims.Recursion.InnerExtras

/-!
# `RecursiveKSAt` from its open premises

`recursiveKSAt_of` is `ksLin2_deployedR_layRP` at the deployed join (`FlockDeploy.joinR`), at the layout of record
`DeployedR.layR rest`. At the plain record at the deployed reader, conflicts are tied to collisions
(`plainR_conflictTied`) and the table bound at the abort law is `plainR_tableBoundAbortLin_R`, from `PlainRdInR` alone
(`hIn`; the cap prefix is `capPrefixR`, proved with no hypotheses). Its last-message binding is `rColsBind_sha512` and
`rCallsBind_sha512`, and it needs no `StepsLogFree` (the finders search the inner verifier's calls at each draw's
coins). The extra calls `layR` records are true at the outer relation (`moreSlotsTrue_deployed`). It takes `DecompBP`
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
  RecDeploy.ksLin2_deployedR_layRP rd b aux nOf hd recRest rest wE FlockDeploy.joinR final w hIn
    (RecDeploy.MoreSlots.moreSlotsTrue_deployed _) hD

/-- **`RecursiveKSAt` from `RecKSDecompL` at the layout of record**: `recursiveKSAt_of` with `DecompB` at the dense
form as `RecDeploy.DecompBR.RecKSDecompL` at `DeployedR.layR rest`, the last part's pieces `rest` and the record
past its link points `recRest`. -/
theorem recursiveKSAt_of_decomp (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hD : RecDeploy.DecompBR.RecKSDecompL rd b aux nOf hd recRest wE final w (RecDeploy.DeployedR.layR rest)) :
    Flock.Guarantees.RecursiveKSAt rd final rest wE w b aux nOf hd recRest :=
  recursiveKSAt_of rd final rest wE w b aux nOf hd recRest hIn fun shape hs => by
    obtain rfl := List.mem_singleton.1 hs
    exact Claims.Recursion.VStarB.decompBP_of_decompB hD

section Leaves

open RecDeploy RecDeploy.DecompBR RecDeploy.RestLow

/-- **`CopyOpensR` doesn't read `DE`'s columns or weights, or `γ`'s weight**: only `final`, `DE`'s rounds and
`opened`. -/
theorem copyOpensR_bdDeployed {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
    {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {L : Claims.Recursion.VStarB.LayoutB}
    {side : Msg → List Msg → OpenPiece → OpenSide} {pieces : Msg → List OpenPiece}
    {nextE : Msg → List Msg → List Coin → Claims.OC Out}
    {colsE colsE' : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)} {wE wE' : Msg → ℝ}
    {join : List Msg → Msg} {rowIn : Msg → Msg} {final : Msg → Option (List (List (Msg × Salt)) × Msg)}
    {w w' : Msg → ℝ}
    (h : OpenLower.CopyOpensR H D shape untr b cmt L
      (bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowIn final w) side pieces) :
    OpenLower.CopyOpensR H D shape untr b cmt L
      (bdDeployed _root_.Flock.Readable.relation nextE colsE' wE' join rowIn final w') side pieces :=
  fun x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hst q hq ho ht os last hfin hl =>
    h x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO wt p hst q hq ho ht os last hfin hl

/-- **`RecursiveKSAt` from chain B's leaves, at the deployed readers**: `recursiveKSAt_of_decomp` at the element-level
reader `DeployedR.rdR` and the last message's pieces `OpenRows.elemR`, with `RecKSDecompL` at the layout of record
from `recKSDecompD_of_leavesJ`, at the decision's positions `Rows.posAt` and row R's readers
(`elemReadC (Top.accRC rdR key)`, `resVRead`). Proved: the extra calls (`moreSlotsTrue_deployed`), the reader's bound
(`plainRdInR`), the rounds no part holds (`saltRowsJ_layR`), row R's target from the pieces `resP` has
(`targetRDep_readers`), the copies' openings (`copyOpensR_layR`) and their directions (`copyDirsTieR_layR` from
`dirsSideR_of`, check 5's `m` from `staged_innerAt` and its coin tie from `posTieR_dense`). Open, each a named leaf:
the pieces `resP` has (`hResP`) and their owners (`hOwn`) and lowering (`hres`), row E's chain at the joint witnesses
(`hEJ`), the JSON parser's spec (`S`, which also gives the base checks, `basesHold_spec`), the last message's reading
(`hF`, `hRd`), the setup's listing (`hset`) and the caps (`hOut`, `hCap`). -/
theorem recursiveKSAt_of_leaves (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (ownR : Msg → ResPiece → Option ℕ)
    (hResP : ResPiecesAt DeployedR.rdR resP key (OpenLower.sideAtR DeployedR.rdR) Rows.posAt)
    (hEJ : Top.RowEJoint DeployedR.rdR key FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      (bdR DeployedR.rdR wE DeployedR.finalR w) (DeployedR.layR OpenRows.elemR) Claims.Recursion.Form.Dense
      (acceptsL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR)))
    (S : Claims.JsonSpec FlockComplete.printJ Flock.canon)
    (hF : RestRead.FinalReads DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hRd : RestRead.ReadsAt DeployedR.rdR DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hOwn : PiecesOwned resP key ownR)
    (hres : ResLowJ DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR) resP
      (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead ownR)
    (hset : ∀ xO, ∀ u ∈ ResLower.resSetup xO, u ∈ b.setup xO)
    (hOut : CapLower.OutBindR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) (Rows.rowDPieces key) Rows.posAt)
    (hCap : CapLower.SideCapR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) DeployedR.rdR (Rows.rowDPieces key) Rows.posAt) :
    Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w b aux nOf hd recRest := by
  have hO : OpenLower.CopyOpensR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) (OpenLower.sideAtR DeployedR.rdR) (Rows.rowDPieces key) :=
    copyOpensR_bdDeployed (OpenRows.copyOpensR_layR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (Rows.rowDPieces key)
      (fun _ => rfl) (fun _ h => h) (fun _ => rfl) (OpenLower.piecesInRangeR_rowDPieces key))
  have hI : ∀ s x c (st0 : Claims.Recursion.Stmt), RecDeploy.parse x = some st0 → Claims.Recursion.Form.Dense st0.form →
      OpenDirs.InnerMR recRead (bB b aux nOf hd recRest) s x c :=
    fun s x c _ hst hdn xO hsh i hins f unit inn hf hload => by
      obtain ⟨v, hv, hm, -⟩ := OpenLower.staged_innerAt (b := bB b aux nOf hd recRest)
        (Bk := (FlockDeploy.V false).k) (s := s) (c := c) hst hdn le_rfl xO hsh i hins f unit inn hf hload
      exact ⟨v, hv, hm⟩
  have hpos : ∀ s x c (st0 : Claims.Recursion.Stmt), RecDeploy.parse x = some st0 → Claims.Recursion.Form.Dense st0.form →
      OpenDirs.DirsCoinR recRead (bB b aux nOf hd recRest) (Rows.rowDPieces key) Rows.posAt
        (bdR DeployedR.rdR wE DeployedR.finalR w).DE.V.k s x c :=
    fun s x c _ hst hdn => OpenLower.posTieR_dense (b := bB b aux nOf hd recRest)
      (Bk := (bdR DeployedR.rdR wE DeployedR.finalR w).DE.V.k) (s := s) (c := c) hst hdn
      (OpenLower.piecesInRangeR_rowDPieces key) le_rfl
  have hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) (Rows.rowDPieces key) Rows.posAt :=
    OpenDirs.copyDirsTieR_layR (Rows.rowDPieces key) Rows.posAt
      (OpenDirs.dirsSideR_of (Rows.rowDPieces key) Rows.posAt (fun _ => rfl) (fun _ h => h)
        (OpenLower.piecesInRangeR_rowDPieces key) hset hI hpos)
  exact recursiveKSAt_of_decomp DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w b aux nOf hd recRest
    (fun x _ => FlockDeploy.KS.plainRdInR DeployedR.rdR x)
    (recKSDecompD_of_leavesJ DeployedR.rdR b aux nOf hd recRest OpenRows.elemR wE DeployedR.finalR w resP key
      (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead ownR (targetRDep_readers DeployedR.rdR hResP) hEJ S
      (basesHold_spec S DeployedR.rdR) hF hRd (saltRowsJ_layR DeployedR.rdR wE w OpenRows.elemR) hOwn hres hO hDirs
      hOut hCap)

end Leaves

end Flock.SecurityProofs
