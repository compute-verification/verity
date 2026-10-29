import Proofs.Flock.Claims.Recursion.ResLowerDD
import Proofs.Flock.Claims.Recursion.RestLowR
import Proofs.Flock.Claims.Recursion.PartAcceptsSetup

/-!
# KS's route with each part's decision (`showsDec`)

The residual leaf holds only at each part's own decision (`ResLower.ResLoweredPD`, `DecompBR.ResLowJPD`: `JointV` over
`ResLower.showsDec`), so KS reads its parts' checks at `showsDec` where it lowers `DecompBV`. The run supplies the
decision: at an accepted run each part's staged statement shows `showsPart` (`stagedB_deployRec_part`) and is decided
(`ResLower.partDec_run`), so `StagedB` holds at `showsDec` (`stagedB_deployRec_dec`). The other leaves stay at
`showsPart`, read through `showsDec → showsPart` (`jointV_shows`, `partOpensB_shows`).

* `ksLin2_stepsC_atVD`, `ksLin2_stepsC_layRVD`, `ksLin2_deployedR_layRVD`: `ksLin2_stepsC_atV`'s chain at `showsDec`.
* `DecompBR.RecKSDecompLVD`: `RecKSDecompLV` at `showsDec`; `resLowJPD_of_resLoweredPD`.
* `RestLow.recKSDecompLV_of_leavesJD`, `RestLow.recKSDecompDV_of_leavesJD`: the leaves' assembly with `hres` at
  `ResLowJPD`, landing on `RecKSDecompLVD`.
-/

namespace RecDeploy.OpenLower

open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-- **`PartOpensB` is antitone in what each part shows.** -/
theorem partOpensB_shows {Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {DO : Claims.Flock.Deploy Coin Col T} {L : LayoutB} {shape : Claims.Recursion.Form → Prop}
    {shows shows' : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    (hw : ∀ {s x xO c}, shows' s x xO c → shows s x xO c) (h : PartOpensB H D DO L shape shows cmt s) :
    PartOpensB H D DO L shape shows' cmt s :=
  fun x st hx hf xO wt hc hT => (h x st hx hf xO wt hc hT).imp_left fun h' c hs => h' c (hw hs)

end RecDeploy.OpenLower

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **An accepted run's V\* statement for part `s` is staged at the run's rounds before the sessions.** -/
theorem runVstar_getD_stagedAcc {W Ω : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (x : Msg) (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((deployRec I.V ivScope b).V.k x) → Coin)
    (hacc : (Claims.run H P (deployRec I.V ivScope b).V x ω r).out = .acc) {s : ℕ}
    (hs : s < (deployRec I.V ivScope b).nS x) :
    (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
  change (Claims.run H P (V I.V b.st) x ω r).out = .acc at hacc
  change s < nS x at hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  show (vstar b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
  rw [vstar_getD b.st x _ hs]
  unfold vstarAt
  rw [ite_eq_left hnum]

/-- **An accepted run's V\* statements are staged at one prefix of its messages** (`runVstar_getD_stagedAcc`). -/
theorem okRuns_deployRec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (shape : Claims.Recursion.Form → Prop) :
    OkRuns H (deployRec I.V ivScope b) I shape fun x xs ws => ResLower.XsStaged b.st x xs ws.length := by
  intro x st hst hf Ω P ω r hacc ws hl
  exact ⟨_, fun s hs => runVstar_getD_stagedAcc H I ivScope b x P ω r hacc (lt_of_lt_of_eq hs hl)⟩

/-- **`StagedB` at `showsDec`**: `stagedB_deployRec_part`, and each part's staged statement on an accepted run decided
(`ResLower.partDec_run`). -/
theorem stagedB_deployRec_dec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (untr : TrReader) (L : Claims.Recursion.VStarB.LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBackB H I.V b shape untr)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope b) I L shape) :
    Claims.Recursion.VStarB.StagedB H (deployRec I.V ivScope b) I L shape (ResLower.showsDec untr b) (cmtTr untr) := by
  intro x st hst hf Ω P ω r hacc s hs
  obtain ⟨h1, h2⟩ := stagedB_deployRec_part H I ivScope b untr L shape hrb hHeld x st hst hf Ω P ω r hacc s hs
  refine ⟨⟨h1, fun i spec sts hsu => ?_⟩, h2⟩
  have hv := runVstar_getD_stagedAcc H I ivScope b x P ω r hacc hs
  change (Claims.run H P (V I.V b.st) x ω r).out = .acc at hacc
  change s < nS x at hs
  rw [hv] at hsu ⊢
  exact ResLower.partDec_run H I.V b.st x P ω r hacc hs hsu

end RecDeploy

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin FinalP nextR)
open Claims.Recursion (Bound Inner)

section LayR

open Flock

/-- **`ksLin2_stepsC_atV` at `showsDec`**: `DecompBV` with each part's check at its decision, `StagedB` there from
`stagedB_deployRec_dec`, the openings at `showsPart` (`partOpensB_shows`). -/
theorem ksLin2_stepsC_atVD (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) (hHW : L.HeldWhole)
    (wE : Msg → ℝ) (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    ReadersTrue FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (deployZkRec _root_.Flock.Readable.relation) →
    (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L
      Claims.Recursion.Form.Dense) →
    (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
      Claims.Recursion.Form.Dense (showsPart recRead bR) (cmtTr recRead) s) →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (ResLower.showsDec recRead bR) (cmtTr recRead))
        fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR Bd D hRT hHe hPO hD
  obtain ⟨hKS, hEC, hCF⟩ := hB
  have hrb : ReadsBackB FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (fun _ => True) recRead :=
    readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl
  refine ksLin2_route_stagedVS _root_.Flock.Readable.relation _root_.Flock.Readable.relation bR L hRows hHW
    FlockDeploy.KS.CpDep (fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) nextE colsE wE join rowInDeployed final
    w hKS (binds_deployed _ _ _ wE join final w) hEC hCF FlockDeploy.KS.zk_KS_outer ?_ hRT
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  exact ⟨ResLower.showsDec recRead bR, cmtTr recRead, hHe,
    fun decI => stagedB_deployRec_dec FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } recRead L _
      (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    fun s => OpenLower.partOpensB_shows (fun h => h.1) (hPO s),
    fun decI => okRuns_deployRec FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } _, hD _ hsh⟩

/-- **`ksLin2_stepsC_layRV` at `showsDec`**: `ksLin2_stepsC_atVD` at `layR`. -/
theorem ksLin2_stepsC_layRVD (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (ResLower.showsDec recRead bR) (cmtTr recRead))
        fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  fun hD => ksLin2_stepsC_atVD nextE colsE b aux nOf hd recRest (DeployedR.layR elem) (readsRows_layR elem)
    (heldWhole_layR elem) wE join final w hB (readersTrue_layR elem _ _ _ hmSlotsCommit_of_full hM)
    (heldB_dense_layR elem nextE colsE _ wE join final w) (partOpensB_dense_layR elem _ _) hD

/-- **`ksLin2_deployedR_layRV` at `showsDec`**: `ksLin2_stepsC_layRVD` at `stepsBase_R`. -/
theorem ksLin2_deployedR_layRVD (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hM : MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation)
      ({ b with tr := recTr aux nOf hd recRest } : Brings).st) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := DeployedR.layR elem
    let Bd := bdDeployed _root_.Flock.Readable.relation (nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompBVS FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAcceptsV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (ResLower.showsDec recRead bR) (cmtTr recRead))
        fun x xs ws => ResLower.XsStaged bR.st x xs ws.length) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_stepsC_layRVD _ _ b aux nOf hd recRest elem wE join final w (stepsBase_R rd wE join final w hIn) hM

end LayR

end RecDeploy

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

noncomputable section

section Statement

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)

/-- **`RecKSDecompLV` at each part's decision**: `DecompBVS` at `acceptsLVD` and the parts' statements staged at one
prefix of messages (`ResLower.XsStaged`, which an accepted run's are: `okRuns_deployRec`), the form
`Flock.SecurityProofs.recursiveKSAt_ofVD` takes. -/
def RecKSDecompLVD : Prop :=
  DecompBVS FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (bdR rd wE final w).IVStar L
    Claims.Recursion.Form.Dense (acceptsLVD rd b aux nOf hd rest wE final w L)
    fun x xs ws => ResLower.XsStaged (bB b aux nOf hd rest).st x xs ws.length

end Statement

end

/-- **`ResLowJPD` at the deployed layout from `ResLoweredPD`**: one statement at the brings `bB`, `Dense`, `recRead`
and `cmtTr recRead`. -/
theorem resLowJPD_of_resLoweredPD (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (pieces : Msg → List ResPiece) (ownQ : Msg → ResPiece → Option ℕ)
    (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
    (h : ResLower.ResLoweredPD rd wE final w (bB b aux nOf hd rest) elem Claims.Recursion.Form.Dense recRead
      (cmtTr recRead) pieces ownQ elemAt vvAt) :
    ResLowJPD rd b aux nOf hd rest wE final w (DeployedR.layR elem) pieces elemAt vvAt ownQ :=
  fun x st hx hf c xs ws hJ hms s hs => h x st hx hf c xs ws hJ hms s hs

end RecDeploy.DecompBR

namespace RecDeploy.RestLow

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR)
open Claims.Recursion.VStarB
open RecDeploy.VPieces (VPiece vrows vholds noB vown)
open RecDeploy.OpenLower (ownQR Joint JointP joint_of_jointP decompBP_of_jointP)
open RecDeploy.RestRead (DenseX kI)
open RecDeploy.DecompBR

noncomputable section

section Assembly

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (L : LayoutB)
  (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
  (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
  (ownR : Msg → ResPiece → Option ℕ)

/-- **`recKSDecompLV_of_leavesJ` at each part's decision**: both shares under `JointV` over `showsDec`, the residuals
read there (`ResLowJPD`) and every other leaf at `showsPart` (`jointV_shows`). -/
theorem recKSDecompLV_of_leavesJD
    (hRdep : Top.TargetRDep rd (PD rd resP key elemAt vvAt side posAt) (Top.accRC rd key))
    (hDm : Top.TargetDMerkle rd (PD rd resP key elemAt vvAt side posAt) (Top.accRC rd key))
    (hEJ : Top.RowEJoint rd key FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L) (bdR rd wE final w) L
      Claims.Recursion.Form.Dense (acceptsLV rd b aux nOf hd rest wE final w L))
    (hB : BasesHold rd) (hF : FinalReadsL final L) (hLn : RoundLensL L) (hRd : ReadsAtL rd final L)
    (hSR : SaltRowRest rd b aux nOf hd rest wE final w L) (hOwn : PiecesOwned resP key ownR)
    (hlowR : RestLowJ rd b aux nOf hd rest wE final w L)
    (hres : ResLowJPD rd b aux nOf hd rest wE final w L resP elemAt vvAt ownR)
    (hSch : OpenLower.SideSchR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key))
    (hO : OpenLower.CopyOpensR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key))
    (hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w)
      (Rows.rowDPieces key) posAt)
    (hTop : OpenLower.CopyTopRP FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w L)
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) L (bdR rd wE final w) side
      (Rows.rowDPieces key) posAt) :
    RecKSDecompLVD rd b aux nOf hd rest wE final w L :=
  recKSDecompLV_of_leavesJS rd b aux nOf hd rest wE final w L resP key elemAt vvAt side posAt ownR
    (ResLower.showsDec recRead (bB b aux nOf hd rest)) (fun h => h.1)
    (fun x xs ws => ResLower.XsStaged (bB b aux nOf hd rest).st x xs ws.length) hRdep hDm hEJ hB hF hLn hRd hSR hOwn
    hlowR hres hSch hO hDirs hTop

end Assembly

section AtNext

variable (rd : Msg → Option FinalP) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
  (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
  (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
  (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
  (resP : Msg → List ResPiece) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
  (elemAt : Msg → List Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (vvAt : Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → _root_.Flock.F128)
  (ownR : Msg → ResPiece → Option ℕ)

/-- **`recKSDecompDV_of_leavesJ` at each part's decision**: `recKSDecompLV_of_leavesJD` at the layout of record, with
`hres` at `ResLowJPD`. -/
theorem recKSDecompDV_of_leavesJD
    (hRdep : Top.TargetRDep rd (PD rd resP key elemAt vvAt (OpenLower.sideAtR rd) Rows.posAt) (Top.accRC rd key))
    (hEJ : Top.RowEJoint rd key FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      (bdR rd wE final w) (DeployedR.layR elem) Claims.Recursion.Form.Dense
      (acceptsLV rd b aux nOf hd rest wE final w (DeployedR.layR elem)))
    (hB : BasesHold rd) (hF : RestRead.FinalReads final (DeployedR.restR elem))
    (hRd : RestRead.ReadsAt rd final (DeployedR.restR elem))
    (hSRJ : SaltRowsJ rd wE final w (DeployedR.layR elem)) (hOwn : PiecesOwned resP key ownR)
    (hres : ResLowJPD rd b aux nOf hd rest wE final w (DeployedR.layR elem) resP elemAt vvAt ownR)
    (hO : OpenLower.CopyOpensR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (OpenLower.sideAtR rd) (Rows.rowDPieces key))
    (hDirs : OpenLower.CopyDirsTieR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (Rows.rowDPieces key) Rows.posAt)
    (hOut : CapLower.OutBindRP FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) (Rows.rowDPieces key) Rows.posAt)
    (hCap : CapLower.SideCapR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key) Rows.posAt) :
    RecKSDecompLVD rd b aux nOf hd rest wE final w (DeployedR.layR elem) :=
  recKSDecompLV_of_leavesJD rd b aux nOf hd rest wE final w (DeployedR.layR elem) resP key elemAt vvAt
    (OpenLower.sideAtR rd) Rows.posAt ownR hRdep (targetDMerkle_PD rd resP key elemAt vvAt _) hEJ hB hF
    (RestRead.roundLens (DeployedR.restR elem))
    (fun x hx c ws hl hT os last hf hos =>
      hRd x hx c ws hl (fun e he => hT e (allCalls_layR elem x ws he)) os last hf hos)
    (saltRowRest_of rd b aux nOf hd rest wE final w _ hSRJ) hOwn
    (restLowJ_of rd b aux nOf hd rest wE final w elem hSRJ) hres
    (OpenLower.sideSchR_sideAtR FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key))
    hO hDirs
    (CapLower.capTop_ofP FlockDeploy.sha512 (recL rd b aux nOf hd rest wE final w (DeployedR.layR elem))
      Claims.Recursion.Form.Dense recRead (bB b aux nOf hd rest) (cmtTr recRead) (DeployedR.layR elem)
      (bdR rd wE final w) rd (Rows.rowDPieces key) Rows.posAt (fun _ => rfl) (fun _ h => h) hOut hCap)

end AtNext

end

end RecDeploy.RestLow
