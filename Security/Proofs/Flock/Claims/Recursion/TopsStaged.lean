import Proofs.Flock.Claims.Recursion.CapPorts
import Proofs.Flock.Claims.Recursion.KSLin2Staged
import Proofs.Flock.Claims.Recursion.BitsLin2

/-!
# `KSLin2` at `layoutBTops` with V\*'s staging read off the run

`ksLin2_deployed_caps` takes V*'s staging as `VstarShowsB`, the staging shown at every list of pairs, and that is false
at the deployment: with `HeldB`, which `layoutBTops` has from `FinPieces` (`heldB_layoutBTops`), it leaves the compiled
verifier no accepted run at a statement of the shape with a part (`vstarShowsB_deployed_rejects`). Its premises so hold
only where no honest run is accepted.

`ksLin2_deployed_staged` is the same bound at `layoutBTops` on `ksLin2_deployed_plain_parts`, which reads the staging
off the run (`StagedB`, `stagedB_deployRec`) and needs no `RunPairs`. `HeldB` comes from `heldB_layoutBTops` at
`schedMatches_dense`, `capPorts` and `rowsCover`, and the decoder and calls premises from `canonicalProofFiles` and
`plainCallsBindOrSlack_sha512`. What stays open at the dense form is the last round's pieces (`FinPieces`),
`PartChecksB` and `PartOpensB` at the coin server's transcript (`showsTr`, `cmtTr`), `DecompB`, the readers' three
facts, and that transcript being injective at the compiled statements (`b.tr x`, a fact about the brings).
`bitsLin2_deployed_staged` is its `BitsLin2On` form (`bitsLin2On_of_ksLin2On`, `sessionAccepts`). At `finOpen`,
`FinPieces` holds (`finPieces_finOpen`), and `ksLin2_deployed_open` and `bitsLin2_deployed_open` (`FinOpen.lean`)
drop it.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops`, with the staging read off the run**:
`ksLin2_deployed_plain_parts` at `layoutBTops`, with `HeldB` from `heldB_layoutBTops`. What stays at the dense form is
`FinPieces`, `PartChecksB` and `PartOpensB` at `showsTr`/`cmtTr`, and `DecompB`; then the readers' three facts and the
coin server's transcript being injective (`htr`). This bounds no number by itself: those premises are open. -/
theorem ksLin2_deployed_staged (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htr : ∀ x st, parse x = some st → Function.Injective (b.tr x)) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      FinPieces fin shape ∧
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) (cmtTr bL) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hP hRT hRL hRP
  refine ksLin2_deployed_plain_parts b L wE join final w canonicalProofFiles plainCallsBindOrSlack_sha512 htr ?_
    hRT hRL hRP
  intro shape hsh
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨hF, chk, hPC, hPO, hD⟩ := hP _ hsh
  exact ⟨chk, fun decI => heldB_layoutBTops _ _ _ _ (fun _ => rfl) fin _ schedMatches_dense (capPorts _)
    (rowsCover _) hF, hPC, hPO, hD⟩

/-- **`BitsLin2` at the deployment's dense statements, at `layoutBTops`**: `ksLin2_deployed_staged` through
`bitsLin2On_of_ksLin2On`, where each V* session accepting is the compiled record's (`sessionAccepts`). It reads
`StrictBirthday` and `ExpectedCR`, the named SHA-512 collision assumptions, and `ksLin2_deployed_staged`'s open
premises: a bound on a number under named premises, not a proved security level. -/
theorem bitsLin2_deployed_staged (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (htr : ∀ x st, parse x = some st → Function.Injective (b.tr x)) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      FinPieces fin shape ∧
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) (cmtTr bL) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hP hRT hRL hRP
  exact Claims.Recursion.bitsLin2On_of_ksLin2On
    (sessionAccepts Bd.IVStar (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL)
    (ksLin2_deployed_staged b fin wE join final w htr hP hRT hRL hRP)

end RecDeploy
