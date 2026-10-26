import Proofs.Flock.Claims.Recursion.BoundRouteMerged
import Proofs.Flock.Claims.Recursion.BoundMergedLin

/-!
# `KSBoundLin` at the deployment with no bridge and no `WeightCovers` (route 2B, linearly)

`RecDeploy.KSBoundRouteLin` reaches `KSBoundLin` through the bridge, so it reads (P3, `Bound.WeightCovers`):
`γ`'s weight at `IVStar` must cover `DE`'s plus the bridge's second plurality. `Bound.KSLin_of_plainKSAbort_binds`
reaches `KSLin` from (P1) at `DE` through one plurality, with `P_O`'s bad draws linear in `Fdev`, so the route needs
neither: `γ`'s weight at `IVStar` need only be at least `DE`'s.

`RecDeploy.KSBoundRouteLinMerged` is `KSBoundRouteMerged`'s premises with `KSBoundLin` for its conclusion, and
`RecDeploy.ksBoundRouteLin_merged` proves it outright: `ksBoundRoute_merged`'s facts at `deployRecC`, with
`KSLin_of_plainKSAbort_binds` there, carried to the deployed record by `ksLin_of_C` as `ksA_of_C` carries `KSA`.
`RecDeploy.ksBoundDeployedLin_merged_wE` is the route at `w := wE`, where the weight premise holds.
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

section Cols

variable {Col T : Type}

/-- **`KSLin` at `deployRecC` is `KSLin` at the deployed record**: `E*`'s output is the same (`E_eval_C`), and the
bound reads no inner view. -/
theorem ksLin_of_C {Col' T' : Type} [DecidableEq Col] [DecidableEq Col'] [Zero Col'] (H : Msg → Msg)
    (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T) (ivScope : Msg → Bool) (b : Brings)
    (DO : Claims.Flock.Deploy Coin Col' T') (hCF : Bd.LastCoinFree H)
    (h : Claims.Recursion.KSLin H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO) :
    Claims.Recursion.KSLin H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO := by
  intro decO hdec x Ω _ _ P M
  have hE : Claims.PrBad H (deployRec Bd.IVStar ivScope b).V
      (Claims.Recursion.R (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRec Bd.IVStar ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P =
    Claims.PrBad H (deployRecC Bd ivScope b).V
      (Claims.Recursion.R (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI))
      (Claims.Recursion.E H (deployRecC Bd ivScope b) (Inner.ofFlock Bd.deploy decI) DO decO) x P := by
    unfold Claims.PrBad
    exact Claims.Pr_congr fun s => by
      rw [E_eval_C H Bd decI ivScope b DO decO hCF x P s.1 s.2.1 s.2.2]
      try exact Iff.rfl
  exact hE.trans_le (h decO hdec x Ω P M)

end Cols

/-- **`KSBoundRouteMerged` with the deviation term linear**: its premises give `KSBoundLin` at the deployed records. -/
def KSBoundRouteLinMerged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) : Prop :=
  ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
    let Bd := bdDeployed rel nextE colsE wE join rowIn final w
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → (∀ x, wE x ≤ w x) → Bd.LastBindsEC FlockDeploy.sha512 →
    Bd.LastCoinFree FlockDeploy.sha512 →
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

/-- **`KSBoundRouteLinMerged`, with no bridge**: `ksBoundRoute_merged`'s argument with
`Bound.KSLin_of_plainKSAbort_binds` at `deployRecC` for `Bound.KS_of_plainKSAbort_binds`, carried back by
`ksLin_of_C`. -/
theorem ksBoundRouteLin_merged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) : KSBoundRouteLinMerged rel relO b L := by
  intro nextE colsE wE join rowIn final w Bd D bL hP1 hBinds hw hEC hCF hF hst hT hL hRRP decI hdec
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
  have hFk := forksBind_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hEC hCF hTc hLc hA
    fun decO x Ω P ω r c ws hg => readersRun_played FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hplay hT hRRP hg
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bL fun x => V_k_pos_flock false x
  have hLn := callsLogColl_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hLc
  have hAn := innerAcceptsColl_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hA
  have hKSn := Claims.Recursion.Bound.KSLin_of_plainKSAbort_binds (Bd := Bd) (decI := decI)
    (H := FlockDeploy.sha512) (D := deployRecC Bd ivScope bL) (DO := deployZkRec relO) FlockDeploy.decCoin hF
    (hP1 decI hdec) hw hdecC hBinds (refuses I.V ivScope bL FlockDeploy.sha512) (innerRounds I ivScope bL)
    (sessionNums I.V ivScope bL relO) (tableErrNonneg I.V ivScope bL) (sessionRounds I ivScope bL relO)
    (sessionAccepts I.V ivScope bL) (sessionRuns_sha512 I ivScope bL relO)
    (innerViewRun_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO)) hTc hLn hFk hAn hM
    (callsNil I.V ivScope bL FlockDeploy.sha512)
  exact ksLin_of_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hKSn

/-- **The linear route at `γ`'s weight `DE`'s own** (`w := wE`), where `KSBoundRouteLinMerged`'s weight premise
holds. -/
theorem ksBoundDeployedLin_merged_wE (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) :
    let Bd := bdDeployed rel nextE colsE wE join rowIn final wE
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Bd.PlainKSAbort FlockDeploy.sha512 → Bd.Binds → Bd.LastBindsEC FlockDeploy.sha512 →
    Bd.LastCoinFree FlockDeploy.sha512 →
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
    Claims.Recursion.KSBoundLin FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO) :=
  fun hP1 hBinds hEC hCF hF hst hT hL hRRP =>
    ksBoundRouteLin_merged rel relO b L hRows nextE colsE wE join rowIn final wE hP1 hBinds (fun _ => le_rfl) hEC hCF hF
      hst hT hL hRRP

end RecDeploy
