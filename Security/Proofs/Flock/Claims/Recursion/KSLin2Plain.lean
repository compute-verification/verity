import Proofs.Flock.Claims.Recursion.KSLin2Deployed
import Proofs.Flock.Claims.KS.PlainKSLin

/-!
# `KSLin2` at the intended choice, with `KSFlockLin` at `DE` proved

`KSFlockLin` reads no weight of its record (`Flock.κ`, `decOK` and `Inner.PrBadAbort` don't), so the bound record's
`DE` at the deployed plain verifier's own steps and columns meets it at the deployed caps `CpDep` at every weight `wE`,
as `deployPlainTrim` does (`FlockDeploy.KS.plain_KSFlockLin`): `plain_KSFlockLin_DE`. `ksLin2_deployed_plain` is
`ksLin2_deployed` at the deployed relation and `CpDep` with that premise discharged, so what stays is the decoder's
canonicity (`CanonicalProofFiles`), the calls half of the plain verifier's last-message binding
(`PlainCallsBindOrSlack`), and the code facts of the staging, V*'s parts and the readers.
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`KSFlockLin` at `DE`**, at the deployed relation, the deployed plain verifier's own steps and columns, any weight
and the deployed caps `CpDep`: `plain_KSFlockLin`, as `DE` differs from `deployPlainTrim` only in a weight
`KSFlockLin` doesn't read. -/
theorem plain_KSFlockLin_DE (wE : Msg → ℝ) :
    Claims.Recursion.KSFlockLin FlockDeploy.sha512
      (deE _root_.Flock.Readable.relation (FlockDeploy.V false).next
        (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE) FlockDeploy.KS.CpDep :=
  fun decode hdec => FlockDeploy.KS.plain_KSFlockLin decode hdec

/-- **`KSLin2` at the intended choice's dense statements, at the deployed relations and caps**: `ksLin2_deployed` at
`Flock.Readable.relation` and `CpDep`, with `KSFlockLin` at `DE` from `plain_KSFlockLin_DE`. -/
theorem ksLin2_deployed_plain (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (hRows : L.ReadsRows FlockDeploy.sha512) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hC : CanonicalProofFiles) (hK : PlainCallsBindOrSlack FlockDeploy.sha512) :
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
  ksLin2_deployed _root_.Flock.Readable.relation b L hRows FlockDeploy.KS.CpDep wE join final w hC hK
    (plain_KSFlockLin_DE wE)

end RecDeploy
