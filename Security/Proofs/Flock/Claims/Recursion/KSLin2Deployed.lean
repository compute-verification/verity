import Proofs.Flock.Claims.Recursion.BoundRouteKSLin2
import Proofs.Flock.Claims.Recursion.BoundDeployOuter
import Proofs.Flock.Claims.Recursion.LastColsDeployed

/-!
# `KSLin2` at the intended choice

`RecDeploy.ksLin2_route` at the deployed bound record whose element-level verifier's steps and columns are the deployed
plain verifier's own and whose rows' digest input is the deployed one (`rowInDeployed`), at the deployed outer relation.
`Binds` is `binds_deployed`, `LastCoinFree` is `lastCoinFree_deployed`, `LastBindsEC` (the calls compared as sets) is
`lastBindsEC_deployed_sha512`, and Flock's KS at the outer record is `zk_KS_outer`. What stays is `KSFlockLin` at `DE`,
the decoder's canonicity (`CanonicalProofFiles`), the calls half of the plain verifier's last-message binding
(`PlainCallsBindOrSlack`), and the code facts of the staging, V*'s parts at the dense form and the readers, and the
bound holds at the dense statements (`DenseAt`).
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`KSLin2` at the intended choice's dense statements, at the deployed outer relation**: `ksLin2_route` at the
deployed plain verifier's own steps and columns and the deployed rows' digest input, at any inner relation, `γ` weight,
join, `final` and weight at `DE`, with (P2), (P4) as sets, `LastCoinFree` and Flock's KS at the outer record discharged.
-/
theorem ksLin2_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (Cp : Claims.Recursion.Caps Coin) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (hC : CanonicalProofFiles)
    (hK : PlainCallsBindOrSlack FlockDeploy.sha512) :
    let Bd := bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed
      final w
    let D := recBound rel b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    Claims.Recursion.KSFlockLin FlockDeploy.sha512 Bd.DE Cp →
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense],
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape) ∧
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) Cp DenseAt := by
  intro Bd D bL hKS hst hT hRRP
  exact ksLin2_route rel _root_.Flock.Readable.relation b L hRows Cp (FlockDeploy.V false).next
    (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final w hKS
    (binds_deployed rel _ _ wE join final w) (lastBindsEC_deployed_sha512 rel wE join final w hC hK)
    (lastCoinFree_deployed rel _ wE join rowInDeployed final w FlockDeploy.sha512) FlockDeploy.KS.zk_KS_outer
    hst hT hRRP

end RecDeploy
