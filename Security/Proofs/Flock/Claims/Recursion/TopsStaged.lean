import Proofs.Flock.Claims.Recursion.CapPorts
import Proofs.Flock.Claims.Recursion.KSLin2Staged
import Proofs.Flock.Claims.Recursion.BitsLin2

/-!
# `KSLin2` at `layoutBTops` with V\*'s staging read off the run

`ksLin2_deployed_caps` takes V*'s staging as `VstarShowsB`, the staging shown at every list of pairs, and that is false
at the deployment: with `HeldB`, which `layoutBTops` has from `FinPieces` (`heldB_layoutBTops`), it leaves the compiled
verifier no accepted run at a statement of the shape with a part (`vstarShowsB_deployed_rejects`). Its premises so hold
only where no honest run is accepted.

`ksLin2_deployed_staged` is the same bound at `layoutBTops` on `ksLin2_deployed_plain_parts`, which reads the
staging off the run (`StagedB`, `stagedB_deployRec_part`) and needs no `RunPairs`, and no `ReadersRunPlayed` at a
layout whose held rounds have no pieces, as `layoutBTops`'s don't (`heldWhole_layoutBTops`). `HeldB` comes from
`heldB_layoutBTops` at `schedMatches_dense`, `capPorts` and `rowsCover`, and the decoder and calls premises from
`canonicalProofFiles` and `plainCallsBindOrSlack_sha512`. What stays open at the dense form is the last round's pieces
(`FinPieces`), `PartChecksBV` and `PartOpensB` at the coin server's transcript read by a reader `untr` at the V\*
statements staged as V\* stages them (`showsPart`, `cmtTr`), `DecompB`, the readers' `ReadersTrue`, and `untr` reading
that transcript back at an accepted run's inner rounds (`ReadsBackB`).
`bitsLin2_deployed_staged` is its `BitsLin2On` form (`bitsLin2On_of_ksLin2On`, `sessionAccepts`). At `finOpen`,
`FinPieces` holds (`finPieces_finOpen`), and `ksLin2_deployed_open` and `bitsLin2_deployed_open` (`FinOpen.lean`)
drop it.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **`layoutBTops` holds no round with pieces before the last** (`HeldWhole`): round 0 has none, and a stream round
it holds has no tops, where `topsPiecesAt` gives none. -/
theorem heldWhole_layoutBTops (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) :
    (layoutBTops fin).HeldWhole := by
  intro x s i hi hs
  have hne : i + 1 ≠ layoutOf.rounds x := Nat.ne_of_lt hi
  change stepPieces fin x i = []
  unfold stepPieces
  rcases i with _ | j
  · simp [hne]
  · simp only [hne, ↓reduceIte, Nat.add_one_ne_zero, Nat.add_sub_cancel]
    unfold topsPieces
    simp only [layoutBTops, streamRound] at hs
    rcases hr : readX x with _ | r
    · rfl
    simp only [hr, Option.bind_some] at hs ⊢
    rcases hrs : repSched r.shape with _ | rs
    · rfl
    simp only [hrs, Option.bind_some, Option.map_some, Option.getD_some] at hs ⊢
    unfold topsPiecesAt
    by_cases ht : (rs.getD (j % rs.size) default).2.1.isEmpty = true
    · simp only [ht, ↓reduceIte]
    · exfalso
      by_cases hj : j < 2 * rs.size
      · simp only [hj, ↓reduceIte, Option.bind_some, ht, Option.isSome_none, Bool.false_eq_true] at hs
      · simp only [hj, ↓reduceIte, Option.bind_none, Option.isSome_none, Bool.false_eq_true] at hs

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops`, with the staging read off the run**:
`ksLin2_deployed_plain_parts` at `layoutBTops`, with `HeldB` from `heldB_layoutBTops` and `HeldWhole` from
`heldWhole_layoutBTops`. What stays at the dense form is
`FinPieces`, `PartChecksBV` and `PartOpensB` at `showsPart`/`cmtTr`, and `DecompB`; then the readers' two facts and
`untr` reading the coin server's transcript back (`ReadsBackB`). This bounds no number by itself: those premises are
open. -/
theorem ksLin2_deployed_staged (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (untr : TrReader) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      FinPieces fin shape ∧
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  refine ksLin2_deployed_plain_parts b L (Claims.Recursion.VStarB.LayoutB.readsRows_of_rowIn fun _ _ => rfl)
    (heldWhole_layoutBTops fin) wE join final w canonicalProofFiles plainCallsBindOrSlack_sha512 untr hrb ?_ hRT
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
    (untr : TrReader) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      FinPieces fin shape ∧
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact Claims.Recursion.bitsLin2On_of_ksLin2On
    (sessionAccepts Bd.IVStar (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck bL)
    (ksLin2_deployed_staged b fin wE join final w untr hrb hP hRT)

end RecDeploy
