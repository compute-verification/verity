import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Recursive.KSV
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
import Proofs.Flock.Claims.Recursion.OutTopRows
import Proofs.Flock.Claims.Recursion.ResPiecesD
import Proofs.Flock.Claims.Recursion.ResLowerD
import Proofs.Flock.Claims.Recursion.SideCapPos
import Proofs.Flock.Claims.Recursion.VerdictRowEFree

/-!
# `RecursiveKSAt` from its open premises

`recursiveKSAt_of` is `ksLin2_deployedR_layRP` at the deployed join (`FlockDeploy.joinR`), at the layout of record
`DeployedR.layR rest`. At the plain record at the deployed reader, conflicts are tied to collisions
(`plainR_conflictTied`) and the table bound at the abort law is `plainR_tableBoundAbortLin_R`, from `PlainRdInR` alone
(`hIn`; the cap prefix is `capPrefixR`, proved with no hypotheses). Its last-message binding is `rColsBind_sha512` and
`rCallsBind_sha512`, and it needs no `StepsLogFree` (the finders search the inner verifier's calls at each draw's
coins). The extra calls `layR` records are true at the outer relation (`moreSlotsTrue_deployed`). It takes `DecompBP`
at what each part's circuit accepts at its own witness `ws[s]` (`hD`, `PartAcceptsP`). `recursiveKSAt_of_decomp` takes
`DecompB` at `PartAccepts` instead, as `RecKSDecompL` at `layR`, which pins (`decompBP_of_decompB`).
`recursiveKSAt_of_leaves` proves `DecompBV` from chain B's leaves (`RestLow.recKSDecompDV_of_leavesJD`) and takes it
through `recursiveKSAt_ofVD`, each part's check read at its decision (`ResLower.showsDec`, which the run supplies):
pinned because the cap tie binds across parts through the parts' own witnesses' calls (`OutBindRP`), and at the V\* statements (`JointV`), which reads every part's check at its statement.
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

/-- **`RecursiveKSAt` from chain B's leaves, at the deployed readers**: `recursiveKSAt_of` at the element-level
reader `DeployedR.rdR` and the last message's pieces `OpenRows.elemR`, through `recursiveKSAt_ofVD`, with `DecompBV`
at each part's decision at the layout of record from `recKSDecompDV_of_leavesJD`, at the decision's positions
`Rows.posAt`, row R's readers (`elemReadC (Top.accRC rdR key)`, `resVRead`) and the concrete `rec-residuals` pieces and owners (`resPD`, `ownRD`).
Proved: the extra calls (`moreSlotsTrue_deployed`), the reader's bound (`plainRdInR`), the rounds no part holds
(`saltRowsJ_layR`), row R's target (`targetRDep_readers`) from the pieces `resPD` has (`resPiecesAt_D`), the pieces'
owners (`piecesOwned_D`), the latter two from V*'s list and the parts' sessions' checks as `readX` holds them, the copies' openings (`copyOpensR_layR`) and their directions (`copyDirsTieR_layR` from
`dirsSideR_of`, check 5's `m` from `staged_innerAt` and its coin tie from `posTieR_dense`) and the level parts' `out`
rows bound to RecCap's `top` rows (`outBindRP_deployed`), and the reader's side of the cap tie from `hRd`
(`sideCapR_leaf`), the base checks (`basesHold_spec`) and row E's chain (`Top.rowEJoint_holds`). Open, each a named
leaf: each algebra part's share of the pieces (`hres`,
`ResLoweredPD`, each part at its own witness and decision, which gives `ResLowJPD` by `resLowJPD_of_resLoweredPD`),
the last message's reading (`hF`, `hRd`), and the setup's listing
(`hset`). -/
theorem recursiveKSAt_of_leaves (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hF : RestRead.FinalReads DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hRd : RestRead.ReadsAt DeployedR.rdR DeployedR.finalR (DeployedR.restR OpenRows.elemR))
    (hres : ResLower.ResLoweredPD DeployedR.rdR wE DeployedR.finalR w (bB b aux nOf hd recRest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead (cmtTr recRead) resPD ownRD (Rows.elemReadC (Top.accRC DeployedR.rdR key))
      Rows.resVRead)
    (hset : ∀ xO, ∀ u ∈ ResLower.resSetup xO, u ∈ b.setup xO) :
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
  have hOut : CapLower.OutBindRP FlockDeploy.sha512
      (recL DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w (DeployedR.layR OpenRows.elemR))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd recRest) (cmtTr recRead) (DeployedR.layR OpenRows.elemR)
      (bdR DeployedR.rdR wE DeployedR.finalR w) (Rows.rowDPieces key) Rows.posAt :=
    CapLower.outBindRP_deployed (b := bB b aux nOf hd recRest) (elem := OpenRows.elemR)
      (IV := (bdR DeployedR.rdR wE DeployedR.finalR w).IVStar) (shape := Claims.Recursion.Form.Dense)
      (Bd := bdR DeployedR.rdR wE DeployedR.finalR w) (Rows.rowDPieces key) (fun _ => rfl) (fun _ h => h)
      (OpenLower.piecesInRangeR_rowDPieces key) (fun _ => le_rfl)
  refine recursiveKSAt_ofVD DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w b aux nOf hd recRest
    (fun x _ => FlockDeploy.KS.plainRdInR DeployedR.rdR x) fun shape hs => ?_
  obtain rfl := List.mem_singleton.1 hs
  exact recKSDecompDV_of_leavesJD DeployedR.rdR b aux nOf hd recRest OpenRows.elemR wE DeployedR.finalR w resPD key
    (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead ownRD
    (targetRDep_readers DeployedR.rdR (resPiecesAt_D DeployedR.rdR key _ _))
    (Top.rowEJoint_holds DeployedR.rdR key _ _ _ _ _ _)
    (basesHold_spec DeployedR.rdR) hF hRd (saltRowsJ_layR DeployedR.rdR wE w OpenRows.elemR) (piecesOwned_D key)
    (resLowJPD_of_resLoweredPD DeployedR.rdR b aux nOf hd recRest wE DeployedR.finalR w OpenRows.elemR _ _ _ _ hres)
    hO hDirs
    hOut (CapLower.sideCapR_leaf wE w b aux nOf hd recRest key hRd)

end Leaves

end Flock.SecurityProofs
