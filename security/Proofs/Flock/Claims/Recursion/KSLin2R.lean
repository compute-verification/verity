module

public import Proofs.Flock.Claims.Recursion.KSFlockLinR
public import Proofs.Flock.Claims.Recursion.LastCallsR

/-! `KSLin2` and `BitsLin2` at the deployed reader's steps (`ksLin2_deployedR_partAccepts`), bounding no number by
themselves: `PlainRdInR` and `DecompB` at `PartAccepts` are open. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg Out OC)
open Claims.Recursion (Bound)
open FlockDeploy (Coin FinalP nextR)

def StepsBaseC (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSFlockLin FlockDeploy.sha512
      (bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w).DE FlockDeploy.KS.CpDep ∧
    (bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w).LastBindsEC
      FlockDeploy.sha512 ∧
    (bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w).LastCoinFree
      FlockDeploy.sha512

theorem ksLin2_stepsC_partAccepts (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D hD
  obtain ⟨hKS, hEC, hCF⟩ := hB
  have hrb : ReadsBackB FlockDeploy.sha512 Bd.IVStar { bR with rd := L.readers } (fun _ => True) recRead :=
    readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl
  refine ksLin2_route_staged _root_.Flock.Readable.relation _root_.Flock.Readable.relation bR L
    (Claims.Recursion.VStarB.LayoutB.readsRows_of_rowIn fun _ _ => rfl) (heldWhole_layoutBTops (finOpen rest))
    FlockDeploy.KS.CpDep nextE colsE wE join rowInDeployed final w hKS (binds_deployed _ _ _ wE join final w) hEC hCF
    FlockDeploy.KS.zk_KS_outer ?_ (readersTrue_layoutBTops (finOpen rest) Bd.IVStar bR _ hmSlotsCommit_of_full)
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨chk, hPC, hDc⟩ := (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _ _).2 (hD _ hsh)
  have hHe : ∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L
      Claims.Recursion.Form.Dense := fun _ =>
    heldB_layoutBTops _ _ _ _ (fun _ => rfl) (finOpen rest) _ schedMatches_dense (capPorts _) (rowsCover _)
      (finPieces_finOpen rest _)
  exact ⟨chk, showsPart recRead bR, cmtTr recRead, hHe,
    fun decI => stagedB_deployRec_part FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck { bR with rd := L.readers } recRead L _
      (fun x st hst _ => hrb x st hst trivial) (hHe decI),
    hPC, fun _ => partOpensB_recRead (finOpen rest) bR (fun _ _ h => h) schedMatches_dense, hDc⟩

theorem bitsLin2_stepsC_partAccepts (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hB : StepsBaseC nextE colsE wE join final w) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation nextE colsE wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D hD
  exact Claims.Recursion.bitsLin2On_of_ksLin2On
    (sessionAccepts Bd.IVStar (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck
      { bR with rd := L.readers })
    (ksLin2_stepsC_partAccepts nextE colsE b aux nOf hd recRest rest wE join final w hB hD)

theorem plainTrimR_cols_congr (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) {v v' : Claims.Seen} (h : FlockDeploy.colsRV rd v = FlockDeploy.colsRV rd v') :
    (FlockDeploy.deployPlainTrimR rel rd).cols v = (FlockDeploy.deployPlainTrimR rel rd).cols v' := by
  show (fun j => (FlockDeploy.colsRV rd v j).map fun p => (p.1, FlockDeploy.trimCol p.2)) =
    fun j => (FlockDeploy.colsRV rd v' j).map fun p => (p.1, FlockDeploy.trimCol p.2)
  rw [h]

theorem lastStepBinds_RT (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg) (rowIn : Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (h : RLastBinds FlockDeploy.sha512 rd) :
    (bdDeployed rel (nextR rd) (FlockDeploy.deployPlainTrimR rel rd).cols wE join rowIn final w).LastStepBinds
      FlockDeploy.sha512 := by
  intro x ms l l' cs hk hc hearly hacc hacc' hne
  exact lastStepBinds_R rel rd wE join rowIn final w h x ms l l' cs hk hc hearly hacc hacc'
    (hne.imp_left fun hc0 heq => hc0 (plainTrimR_cols_congr rel rd heq))

theorem lastBindsEC_RT (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (h : RLastBinds FlockDeploy.sha512 rd) :
    (bdDeployed rel (nextR rd) (FlockDeploy.deployPlainTrimR rel rd).cols wE join rowInDeployed final w).LastBindsEC
      FlockDeploy.sha512 :=
  Bound.lastBindsEC_of_lastStep (binds_deployed rel _ _ wE join final w) (lastStepFree_of_R _ rd _ rfl)
    (lastStepBinds_RT rel rd wE join rowInDeployed final w h)

theorem stepsBase_R (rd : Msg → Option FinalP) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x) :
    StepsBaseC (nextR rd) (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join final w :=
  ⟨ksFlockLin_R rd wE join final w hIn,
    lastBindsEC_RT _ rd wE join final w (rLastBinds_of (rColsBind_sha512 rd) (rCallsBind_sha512 rd)),
    lastCoinFree_R _ rd _ wE join rowInDeployed final w FlockDeploy.sha512⟩

/-- `Flock.SecurityProofs.RecursiveKS` proves `KSLin2` at the deployed reader with none of its premises. -/
theorem ksLin2_deployedR_partAccepts (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_stepsC_partAccepts _ _ b aux nOf hd recRest rest wE join final w
    (stepsBase_R rd wE join final w hIn)

/-- A bound on a number under `StrictBirthday`, `ExpectedCR` and `ksLin2_deployedR_partAccepts`'s open premises, not a
proved security level. -/
theorem bitsLin2_deployedR_partAccepts (rd : Msg → Option FinalP) (b : Brings)
    (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  bitsLin2_stepsC_partAccepts _ _ b aux nOf hd recRest rest wE join final w
    (stepsBase_R rd wE join final w hIn)

end RecDeploy
