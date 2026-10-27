import Proofs.Flock.Claims.Recursion.FinOpen
import Proofs.Flock.Claims.Recursion.RecRecord

/-!
# `KSLin2` and `BitsLin2` at the model of the coin server's record

`ksLin2_deployed_open` and `bitsLin2_deployed_open` take `ReadsBackB`: a reader `untr` reads the brings' transcript of
an accepted run's inner rounds back to those rounds before the last. Where the transcript is the model of the coin
server's record (`recTr`: round 0's message as the link's level-0 top and its coin's values as the link points, each
later round before the last as a recorded round, its tops' nodes and its `c_k`, with its coin's values, and no last
round) and the reader is the record's own total reader (`recRead`), that holds at every accepted run
(`readsBack_recTr`), so `ksLin2_deployed_rec` and `bitsLin2_deployed_rec` drop it.

The difftest on the honest fixtures (`test_lean_rec_record.py`) ties `recTr` to the coin server's record: `recordFullW`
writes `rec_live.record_bytes`' bytes up to the link points, and the rounds `recRead` shows are round 0 from the link
and the recorded rounds, each message the schedule's count of commit strings (`nAtSched`), one fewer than the inner
verifier's rounds. What stays open at the dense form is `PartChecksBV` and `PartOpensB` at
`showsPart recRead`/`cmtTr recRead` (the staged V\* statements), `DecompB`, and the readers' `ReadersTrue`
(the inner verifier's own calls are searched at each draw's coins, `callsLogCollAt_deployRec`).

The reader is not vacuous: `recRead_honest` reads an honest three-round run back, and `Flock.Test.RecRecord`'s
`#guard`s run `readRecord` on the plain session's honest record bytes. Neither witnesses the open premises, which no
concrete record can until they are proved.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **The reader at an honest run**: inner rounds answered with their coins' bytes, as the compiled verifier answers
them, read back to their messages and their coins' first `n` values before the last round, whatever the record's
other fields. -/
theorem recRead_honest (c₀ c₁ c₂ : Coin) (m₀ m₁ m₂ : Msg) (a : Flock.RecRecord.Round) (hd : Flock.RecRecord.Record)
    (n : ℕ) :
    recRead [] (recTr (fun _ _ => a) (fun _ _ => n) (fun _ => hd) (fun _ _ => []) []
      [(m₀, .msg c₀.bytes), (m₁, .msg c₁.bytes), (m₂, .msg c₂.bytes)]) = some [(m₀, c₀.vals n), (m₁, c₁.vals n)] := by
  simp only [recRead, recTr, List.toList_toArray, Flock.RecRecord.readFull_recordFullW, Option.map_some]
  simp [Flock.RecRecord.recOf, Flock.RecRecord.Record.msgs, Flock.RecRecord.flatten_nodesOf,
    Flock.RecRecord.msg_roundD, Flock.RecRecord.coins_roundD, coinVals, outCoin, coinOf_bytes]

/-- **`KSLin2` at the deployment's dense statements, at the model of the coin server's record (`recTr`)**:
`ksLin2_deployed_open` at the brings whose transcript is `recTr aux nOf hd rest`, read by `recRead`, with
`ReadsBackB` from `readsBack_recTr`. What stays open at the dense form is `PartChecksBV` and `PartOpensB` at
`showsPart recRead`/`cmtTr recRead`, and `DecompB`; then the readers' two facts. This bounds no number by itself: those
premises are open. -/
theorem ksLin2_deployed_rec (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hRT
  exact ksLin2_deployed_open bR fin wE join final w recRead
    (readsBack_recTr _ _ bL _ aux nOf hd rest rfl) hP hRT

/-- **`BitsLin2` at the deployment's dense statements, at the model of the coin server's record (`recTr`)**:
`bitsLin2_deployed_open` with `ReadsBackB` from `readsBack_recTr`. It reads `StrictBirthday` and `ExpectedCR`, the named
SHA-512 collision assumptions, and `ksLin2_deployed_rec`'s open premises: a bound on a number under named premises, not
a proved security level. -/
theorem bitsLin2_deployed_rec (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hRT
  exact bitsLin2_deployed_open bR fin wE join final w recRead
    (readsBack_recTr _ _ bL _ aux nOf hd rest rfl) hP hRT

end RecDeploy
