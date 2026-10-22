import Proofs.Flock.Claims.Recursion.KSLin2Plain
import Proofs.Flock.Claims.Recursion.VStarBDeploy

/-!
# `KSLin2` with V*'s staging read off the run

`ksLin2_route` takes V*'s staging as `KSBoundRoute` does: `VstarShowsB`, the staging shown at every list of pairs, and
`RunPairs`. `VstarShowsB` is false at `deployRec` (`vstarShowsB_deployed_rejects`: with `HeldB`, it leaves the compiled
verifier no accepted run at any statement with a part). What the route reads of it is `StagedB`, the staging at an
accepted run's pairs, which holds at the coin server's transcript wherever that transcript is injective
(`stagedB_deployRec`), and `RunPairs` holds at `decCoin` (`runPairs_deployRec`).

* `ksLin2_route_staged`: `ksLin2_route` with `StagedB` for `VstarShowsB` and no `RunPairs`, as `ksBoundRouteStaged` is
  `ksBoundRoute_of_bridge` (`vstarAccepts_bound_staged` for `vstarAccepts_bound`).
* `ksLin2_deployed_parts`: at the intended choice (`ksLin2_deployed`'s), with the staging at `showsTr`, `cmtTr`.
* `ksLin2_deployed_plain_parts`: at the deployed relation and caps `CpDep`, with `KSFlockLin` at `DE` from
  `plain_KSFlockLin_DE`. What stays is the decoder's canonicity (`CanonicalProofFiles`), the calls half of the plain
  verifier's last-message binding (`PlainCallsBindOrSlack`), the transcript's injectivity at parsed statements, at the
  dense form `HeldB`, `PartChecksB`, `PartOpensB` and `DecompB` for some inner check, and the readers' three facts.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`KSLin2` at the deployed records' dense statements with the staging read off the run**: `ksLin2_route`'s premises
with `StagedB` (at an accepted run) for `VstarShowsB` (at every staging, false at the deployment) and `RunPairs` dropped
(it holds at `decCoin`), proved as `ksLin2_route` is with `vstarAccepts_bound_staged`. -/
theorem ksLin2_route_staged (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (Cp : Claims.Recursion.Caps Coin) :
    ∀ (nextE : Msg → List Msg → List Coin → Claims.OC Out)
      (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
      (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ),
      let Bd := bdDeployed rel nextE colsE wE join rowIn final w
      let D := recBound rel b L Bd.IVStar
      let bL : Brings := { b with rd := L.readers }
      Claims.Recursion.KSFlockLin FlockDeploy.sha512 Bd.DE Cp → Bd.Binds → Bd.LastBindsEC FlockDeploy.sha512 →
      Bd.LastCoinFree FlockDeploy.sha512 →
      Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO) →
      (∀ shape ∈ [Claims.Recursion.Form.Dense],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.StagedB FlockDeploy.sha512 D
            (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec relO) L shape shows chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec relO) L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
      ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec relO) →
      Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd (deployZkRec relO) Cp DenseAt := by
  intro nextE colsE wE join rowIn final w Bd D bL hKS hBinds hEC hCF hF hshapes hT hL hRRP decI hdI decO hdO x hx Ω _
    _ P M R
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let ivScope := (FlockDeploy.deployPlain rel).scopeCheck
  have hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (Claims.run FlockDeploy.sha512 (Claims.replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x) :=
    fun x ms c => Bd.run_ivstar_length FlockDeploy.sha512 (xIOf x) ms c
  have hacc : ∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarAccepts FlockDeploy.sha512 (deployRec I.V ivScope bL) I (deployZkRec relO) shape := by
    intro shape hs
    obtain ⟨chk, shows, cmt, hHe, hSt, hCk, hOp, hDc⟩ := hshapes shape hs
    exact vstarAccepts_bound_staged FlockDeploy.sha512 I ivScope b L (deployZkRec relO) shape FlockDeploy.decCoin chk
      shows cmt hplay (runPairs_deployRec FlockDeploy.sha512 I ivScope bL) (hHe decI) (hSt decI) hCk hOp hDc hRRP
  have hTc := callsTrue FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT
  have hA := Claims.Recursion.innerAcceptsCollAt_of_dense (refuses I.V ivScope bL FlockDeploy.sha512) hTc
    (hacc _ (by simp)) hx
  have hLc := callsLogColl FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hT hL
  have hFk := forksBindAt_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hEC hCF hTc hLc
    (fun decO x Ω P ω r c ws hg => readersRun_played FlockDeploy.sha512 I ivScope bL (deployZkRec relO) hplay hT hRRP
      hg) x hA
  have hdecC : ∀ x c, FlockDeploy.decCoin x (Bd.ans c) = c := fun _ c =>
    (congrArg (FlockDeploy.decCoin _) (bdDeployed_ans rel nextE colsE wE join rowIn final w c)).trans
      (Function.leftInverse_invFun Claims.Bridge.coin_bytes_injective c)
  have hM := roundsMargin_deployRec FlockDeploy.sha512 I ivScope bL fun x => V_k_pos_flock false x
  have hLn := callsLogColl_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF hLc
  have hAn := innerAcceptsCollAt_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) hCF x hA
  have hbody := Claims.Recursion.Bound.ksLin2_of_flockLin_binds (Bd := Bd) (decI := decI)
    (H := FlockDeploy.sha512) (D := deployRecC Bd ivScope bL) (DO := deployZkRec relO) (decO := decO) (P := P)
    (x := x) FlockDeploy.decCoin Cp hF hKS hdI hdO hdecC hBinds (refuses I.V ivScope bL FlockDeploy.sha512)
    (innerRounds I ivScope bL) (sessionNums I.V ivScope bL relO) (tableErrNonneg I.V ivScope bL)
    (sessionRounds I ivScope bL relO) (sessionAccepts I.V ivScope bL) (sessionRuns_sha512 I ivScope bL relO)
    (innerViewRun_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO)) hTc hLn hFk hAn hM
    (callsNil I.V ivScope bL FlockDeploy.sha512) M R
  exact ksLin2_of_C FlockDeploy.sha512 Bd decI ivScope bL (deployZkRec relO) decO hCF Cp x P M R hbody

/-- **`KSLin2` at the intended choice with V\*'s staging discharged**: `ksLin2_route_staged` at `ksLin2_deployed`'s
choice, with the staging at the coin server's transcript (`showsTr`, `cmtTr`), where `StagedB` holds wherever that
transcript is injective (`stagedB_deployRec`). What V*'s parts still owe at the dense form is `HeldB`, `PartChecksB`,
`PartOpensB` and `DecompB`, for some inner check `chk`. -/
theorem ksLin2_deployed_parts (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (Cp : Claims.Recursion.Caps Coin) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (hC : CanonicalProofFiles)
    (hK : PlainCallsBindOrSlack FlockDeploy.sha512)
    (htr : ∀ x st, parse x = some st → Function.Injective (b.tr x)) :
    let Bd := bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed
      final w
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Claims.Recursion.KSFlockLin FlockDeploy.sha512 Bd.DE Cp →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) (cmtTr bL) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) Cp DenseAt := by
  intro Bd D bL hKS hparts hT hL hRRP
  refine ksLin2_route_staged rel _root_.Flock.Readable.relation b L Cp (FlockDeploy.V false).next
    (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final w hKS
    (binds_deployed rel _ _ wE join final w) (lastBindsEC_deployed_sha512 rel wE join final w hC hK)
    (lastCoinFree_deployed rel _ wE join rowInDeployed final w FlockDeploy.sha512) FlockDeploy.KS.zk_KS_outer
    ?_ hT hL hRRP
  intro shape hs
  obtain ⟨chk, hHe, hCk, hOp, hDc⟩ := hparts shape hs
  exact ⟨chk, showsTr bL, cmtTr bL, hHe,
    fun decI => stagedB_deployRec FlockDeploy.sha512 (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
      (FlockDeploy.deployPlain rel).scopeCheck bL L shape (fun x st hst _ => htr x st hst) (hHe decI),
    hCk, hOp, hDc⟩

/-- **`KSLin2` at the deployed relation and caps with V\*'s staging discharged**: `ksLin2_deployed_parts` at
`Flock.Readable.relation` and `CpDep`, with `KSFlockLin` at `DE` from `plain_KSFlockLin_DE`. -/
theorem ksLin2_deployed_plain_parts (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hC : CanonicalProofFiles) (hK : PlainCallsBindOrSlack FlockDeploy.sha512)
    (htr : ∀ x st, parse x = some st → Function.Injective (b.tr x)) :
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) (cmtTr bL) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_deployed_parts _root_.Flock.Readable.relation b L FlockDeploy.KS.CpDep wE join final w hC hK htr
    (plain_KSFlockLin_DE wE)

end RecDeploy
