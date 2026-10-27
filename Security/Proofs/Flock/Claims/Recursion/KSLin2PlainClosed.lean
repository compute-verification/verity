import Proofs.Flock.Claims.Recursion.KSLin2Plain
import Proofs.Flock.Claims.Recursion.LastBindsClosed

/-!
# `KSLin2` at the intended choice, with the decoder's canonicity and the calls half proved

`ksLin2_deployed_plain` with `CanonicalProofFiles` from `canonicalProofFiles` (`Flock.decodeProof_canonical`) and
`PlainCallsBindOrSlack` at SHA-512 from `plainCallsBindOrSlack_sha512`. What stays is the staging and V*'s parts at
the dense shape, and the readers at `deployZkRec`.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`KSLin2` at the intended choice's dense statements, at the deployed relations and caps, with no decoder or
calls premise**:
`ksLin2_deployed_plain` at `canonicalProofFiles` and `plainCallsBindOrSlack_sha512`. -/
theorem ksLin2_deployed_plain_closed (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
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
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt :=
  ksLin2_deployed_plain b L hRows wE join final w canonicalProofFiles plainCallsBindOrSlack_sha512

end RecDeploy
