import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.RecRecord

/-!
# `KSLin2` and `BitsLin2` at the coin server's record

`ksLin2_deployed_open` and `bitsLin2_deployed_open` take `ReadsBack`: a reader `untr` reads the brings' transcript of
an accepted run's inner rounds back to those rounds. Where the transcript is the coin server's record (`recTr`, each
inner round's message as its one commit string and its coin's first `n` values, whatever its other fields) and the
reader is the record's own total reader (`recRead`), that holds at every accepted run (`readsBack_recTr`), so
`ksLin2_deployed_rec` and `bitsLin2_deployed_rec` drop it.

That the deployed coin server writes `recTr`'s bytes is a difftest on the honest fixtures (`test_lean_rec_record.py`:
`recordW` writes `rec_live.record_bytes`' bytes), not a theorem. What stays open at the dense form is `PartChecksB` and
`PartOpensB` at `showsTr recRead`/`cmtTr recRead`, `DecompB`, and the readers' three facts.

The reader is not vacuous: `recRead_honest_round` reads an honest one-round run back, and `Flock.Test.RecRecord`'s
`#guard`s run `readRecord` on the plain session's honest record bytes. Neither witnesses the open premises, which no
concrete record can until they are proved.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **The reader at an honest one-round run**: an inner round answered with its coin's bytes, as the compiled verifier
answers it, is read back to its message and the coin's first `n` values, whatever the record's other fields. -/
theorem recRead_honest_round (c : Coin) (m : Msg) (a : Flock.RecRecord.Round) (n : ℕ) :
    recRead [] (recTr (fun _ _ => a) (fun _ _ => n) (fun _ _ => []) [] [(m, .msg c.bytes)]) = some [(m, c.vals n)] := by
  simp only [recRead, recTr, List.toList_toArray, Flock.RecRecord.readRecord_recordW, Option.map_some,
    List.mapIdx_cons, List.mapIdx_nil, List.map_cons, List.map_nil]
  simp [roundOf, flatten_commits, outCoin, coinOf_bytes]

/-- **`KSLin2` at the deployment's dense statements, at the coin server's record**: `ksLin2_deployed_open` at the
brings whose transcript is `recTr aux nOf rest`, read by `recRead`, with `ReadsBack` from `readsBack_recTr`. What stays
open at the dense form is `PartChecksB` and `PartOpensB` at `showsTr recRead`/`cmtTr recRead`, and `DecompB`; then the
readers' three facts. This bounds no number by itself: those premises are open. -/
theorem ksLin2_deployed_rec (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hRT hRL hRP
  exact ksLin2_deployed_open bR fin wE join final w recRead
    (readsBack_recTr _ _ bL _ aux nOf rest rfl) hP hRT hRL hRP

/-- **`BitsLin2` at the deployment's dense statements, at the coin server's record**: `bitsLin2_deployed_open` with
`ReadsBack` from `readsBack_recTr`. It reads `StrictBirthday` and `ExpectedCR`, the named SHA-512 collision
assumptions, and `ksLin2_deployed_rec`'s open premises: a bound on a number under named premises, not a proved
security level. -/
theorem bitsLin2_deployed_rec (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr recRead) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hRT hRL hRP
  exact bitsLin2_deployed_open bR fin wE join final w recRead
    (readsBack_recTr _ _ bL _ aux nOf rest rfl) hP hRT hRL hRP

end RecDeploy
