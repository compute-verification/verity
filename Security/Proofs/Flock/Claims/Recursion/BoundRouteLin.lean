import Proofs.Flock.Claims.Recursion.BoundBridge
import Proofs.Flock.Claims.Recursion.KSLin

/-!
# `KSBoundLin` at the deployment, on `KSBoundRoute`'s premises

`KSBoundRouteLin` is `KSBoundRoute` with `KSBoundLin` in place of `KSBoundDeployed`: the same premises, (P1) to (P4)
at the deployed bound record, Flock's KS at the deployed outer `--zk` record, the staging off the run, V*'s parts and
its readers, and the recursion's KS with its deviation term linear (`Claims.Recursion.KSLin`). Its proof is
`ksBoundRoute_of_bridge`'s, with `KSLin_of_flockKS_binds` at the normalized record in place of `KS_of_flockKS_binds`,
carried back by `ksLin_of_nf` as `ksA_of_nf` carries `KSA`: the normalized record changes only the readers' messages,
which neither `E*`'s output nor any term of the bound reads. `ksBoundRouteLin` is it at the bridge
(`bridge_bdDeployed`).
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

section Normal

variable {Col T : Type} [DecidableEq Col] {Col' T' : Type} [DecidableEq Col']

/-- **`KSLin` at the normalized record is `KSLin` at the deployed one**: `E*`'s output is the same (`E_eval_nf`), and
the bound reads no inner view. -/
theorem ksLin_of_nf [Zero Col'] (H : Msg → Msg) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (ivScope : Msg → Bool) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T')
    (h : Claims.Recursion.KSLin H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.KSLin H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO hdec x Ω _ _ P M
  have hE : Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P =
    Claims.PrBad H (deployRec Bd.IVStar ivScope (nfBrings Bd b)).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope (nfBrings Bd b)) (Inner.ofFlock Bd.deploy decI) DO decO)
      x P := by
    unfold Claims.PrBad
    exact Claims.Pr_congr fun s => by
      rw [E_eval_nf H Bd decI ivScope b DO decO x P s.1 s.2.1 s.2.2]
      try exact Iff.rfl
  exact hE.trans_le (h decO hdec x Ω P M)

end Normal

/-- **`KSBoundRoute` with the deviation term linear**: `KSBoundRoute`'s premises at every choice the code side makes
give `KSBoundLin` at the deployed records. -/
def KSBoundRouteLin (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) : Prop :=
  ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
    let Bd := bdDeployed rel nextE colsE wE join rowIn final w
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.WeightCovers → Bd.LastBinds FlockDeploy.sha512 →
    Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
    Claims.Recursion.KSBoundLin FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO)

/-- **`KSBoundRouteLin` from the bridge**: `ksBoundRoute_of_bridge`'s facts at the normalized record, with
`KSLin_of_flockKS_binds` for `KS_of_flockKS_binds`, carried back by `ksLin_of_nf`. -/
theorem ksBoundRouteLin_of_bridge (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512)
    (hB : ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      Claims.Recursion.Bridge FlockDeploy.sha512 (bdDeployed rel nextE colsE wE join rowIn final w)) :
    KSBoundRouteLin rel relO b L := by
  intro nextE colsE wE join rowIn final w Bd D bL hP1 hBinds hWC hLB hF hst hT hL hRRP decI hdec
  obtain ⟨decC, hRP, hshapes⟩ := hst
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (run FlockDeploy.sha512 (replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x) :=
    fun x ms c => Bd.run_ivstar_length FlockDeploy.sha512 (xIOf x) ms c
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense, Claims.Recursion.Form.Holo, Claims.Recursion.Form.Sparse],
      Claims.Recursion.VStarAccepts FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨chk, shows, cmt, hHe, hSh, hCk, hOp, hDc⟩ := hshapes shape hs
    exact vstarAccepts_bound FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape decC chk shows cmt hplay
      (hRP decI) (hHe decI) (hSh decI) hCk hOp hDc hRows hRRP
  have hA := innerAcceptsColl_bound FlockDeploy.sha512 I ivScope b L (deployZkRec relO) hT
    (hacc _ (by simp)) (hacc _ (by simp)) (hacc _ (by simp))
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hLc := callsLogColl FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT hL
  have hFk := forksBind_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hLB hTc hLc hA
    fun decO x Ω P ω r c ws hg => readersRun_played FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hplay hT hRRP hg
  let bN := nfBrings Bd bL
  have hKS : I.KSAbort FlockDeploy.sha512 := hB nextE colsE wE join rowIn final w decI hdec hBinds hWC (hP1 decI hdec)
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hpc := Bd.publicCoin FlockDeploy.sha512 decI FlockDeploy.decCoin hdecC
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bN fun x => V_k_pos_flock false x
  have hLn := callsLogColl_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hLc
  have hAn := innerAcceptsColl_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hA
  have hKSn := Claims.Recursion.KSLin_of_flockKS_binds FlockDeploy.sha512 (deployRec I.V ivScope bN) I
    (deployZkRec relO) FlockDeploy.decCoin hF hKS (refuses I.V ivScope bN FlockDeploy.sha512)
    (innerRounds I ivScope bN) (sessionNums I.V ivScope bN relO) (tableErrNonneg I.V ivScope bN)
    (sessionRounds I ivScope bN relO) (sessionAccepts I.V ivScope bN) (sessionRuns_sha512 I ivScope bN relO)
    (innerViewRun FlockDeploy.sha512 I ivScope bN (deployZkRec relO)) hTc hLn hFk hAn hM hpc
    (callsNil I.V ivScope bN FlockDeploy.sha512)
  exact ksLin_of_nf FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hKSn

/-- **`KSBoundRouteLin`**: `ksBoundRouteLin_of_bridge` at the bridge, which holds at every choice the code side makes
(`bridge_bdDeployed`). -/
theorem ksBoundRouteLin (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (hRows : L.ReadsRows FlockDeploy.sha512) : KSBoundRouteLin rel relO b L :=
  ksBoundRouteLin_of_bridge rel relO b L hRows fun nextE colsE wE join rowIn final w =>
    bridge_bdDeployed rel nextE colsE wE join rowIn final w

end RecDeploy
