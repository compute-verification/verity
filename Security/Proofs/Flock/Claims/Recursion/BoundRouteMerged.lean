import Proofs.Flock.Claims.Recursion.BoundRouteCols
import Proofs.Flock.Claims.Recursion.BoundMerged

/-!
# `KSBoundRouteCols` with no bridge and no `WeightCovers` (route 2B)

`ksBoundRoute_of_bridge_cols` takes the bridge (`Claims.Recursion.Bridge`) and reads (P3, `Bound.WeightCovers`) there:
`γ`'s weight at `IVStar` must cover `DE`'s plus the bridge's second plurality, `N_I·√(k_I/32)`.
`Bound.KS_of_plainKSAbort_binds` reaches the inner stage from (P1) at `DE` through one plurality, so the route needs
neither: `γ`'s weight at `IVStar` need only be at least `DE`'s.

`RecDeploy.KSBoundRouteMerged` is `KSBoundRouteCols` with `WeightCovers` replaced by `∀ x, wE x ≤ w x`, and
`RecDeploy.ksBoundRoute_merged` proves it outright, with every other premise `ksBoundRoute_of_bridge_cols`'s.
`RecDeploy.ksBoundDeployed_merged_wE` is the route at `w := wE`, where that premise holds.
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay findColl Collision)
open FlockDeploy (Coin)
open Claims.Recursion (Bound Inner)

/-- **`KSBoundRouteCols` with `DE`'s weight for (P3)**: its premises with `Bound.WeightCovers` replaced by `γ`'s
weight at `IVStar` (`w`) at least `DE`'s (`wE`). -/
def KSBoundRouteMerged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
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
    KSBoundDeployed rel relO b L nextE colsE wE join rowIn final w

/-- **`KSBoundRouteMerged`, with no bridge**: `ksBoundRoute_of_bridge_cols`'s argument with
`Bound.KS_of_plainKSAbort_binds` at `deployRecC` for `KS_of_flockKS_binds`, which reads (P1) at `DE` directly and
`DE`'s weight at most `w` for the bridge and (P3). -/
theorem ksBoundRoute_merged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) : KSBoundRouteMerged rel relO b L := by
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
  have hKSn := Claims.Recursion.Bound.KS_of_plainKSAbort_binds (Bd := Bd) (decI := decI) (H := FlockDeploy.sha512)
    (D := deployRecC Bd ivScope bL) (DO := deployZkRec relO) FlockDeploy.decCoin hF (hP1 decI hdec) hw hdecC hBinds
    (refuses I.V ivScope bL FlockDeploy.sha512) (innerRounds I ivScope bL) (sessionNums I.V ivScope bL relO)
    (tableErrNonneg I.V ivScope bL) (sessionRounds I ivScope bL relO) (sessionAccepts I.V ivScope bL)
    (sessionRuns_sha512 I ivScope bL relO) (innerViewRun_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO))
    hTc hLn hFk hAn hM (callsNil I.V ivScope bL FlockDeploy.sha512)
  exact ksA_of_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hKSn

/-- **The route at `γ`'s weight `DE`'s own** (`w := wE`), where `KSBoundRouteMerged`'s weight premise holds. -/
theorem ksBoundDeployed_merged_wE (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
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
    KSBoundDeployed rel relO b L nextE colsE wE join rowIn final wE :=
  fun hP1 hBinds hEC hCF hF hst hT hL hRRP =>
    ksBoundRoute_merged rel relO b L hRows nextE colsE wE join rowIn final wE hP1 hBinds (fun _ => le_rfl) hEC hCF hF
      hst hT hL hRRP

end RecDeploy
