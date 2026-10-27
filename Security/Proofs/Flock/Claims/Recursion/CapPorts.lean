import Proofs.Flock.Claims.Recursion.RowsCover
import Proofs.Flock.Claims.Recursion.SchedDense

/-!
# RecCap's part has its port `top` (`CapPorts`), from the reader's check

`CapPorts` (`LayoutBCap.lean`) asks, at every compiled statement, that the last part's circuit, as the setup parses it,
have RecCap's port `top`, the one fact `heldB_layoutBTops` still takes at round 0 (`capAt_readX` puts RecCap last).
`readX` refuses a record whose last part's circuit has no port `top` (`capPorted`), as RecCap's session does
(`checkFile`, `checkParsed`).

`capPorts`: `CapPorts` at every shape. `ksLin2_deployed_caps` is `ksLin2_deployed_rows` with it discharged, and
with the schedule at the dense shape (`schedMatches_dense`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **`CapPorts` at every shape**: `readX` read the last part's circuit with a port `top` (`readX_listed`), the part
`capAt_readX` names RecCap's. -/
theorem capPorts (shape : Claims.Recursion.Form → Prop) : CapPorts shape := by
  intro x st _ _ r hr
  obtain ⟨-, -, -, -, hcp⟩ := readX_listed hr
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  rw [hn]
  unfold capPorted at hcp
  cases hc : r.w.parts[r.w.parts.length - 1]?.bind partCircuit with
  | none =>
    rw [hc] at hcp
    cases hcp
  | some c =>
    rw [hc] at hcp
    refine ⟨c, rfl, ?_⟩
    unfold capPort
    rw [Array.findIdx?_isSome]
    exact hcp

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops`, with the rows' and RecCap's ports discharged**:
`ksLin2_deployed_rows` with `CapPorts` from `capPorts` and the schedule from `schedMatches_dense`. What stays at the
dense form is the last round's pieces (`FinPieces`, which holds at `finOpen`: `finPieces_finOpen`), and `VstarShowsB`,
`PartChecksBV`, `PartOpensB` and `DecompB`; then `RunPairs` and the readers' three facts. This bounds no number by
itself: those premises are open, and `VstarShowsB` with `HeldB` leaves no accepted run (`vstarShowsB_deployed_rejects`),
so they hold only where no honest run is accepted. `ksLin2_deployed_staged` reads the staging off the run instead. -/
theorem ksLin2_deployed_caps (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense],
        FinPieces fin shape ∧
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
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
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hP hRT hRP
  refine ksLin2_deployed_rows b fin wE join final w ?_ hRT hRP
  obtain ⟨decC, hRun, hS⟩ := hP
  refine ⟨decC, hRun, fun shape hsh => ?_⟩
  obtain rfl := List.mem_singleton.1 hsh
  obtain ⟨hF, rest⟩ := hS _ hsh
  exact ⟨schedMatches_dense, capPorts _, hF, rest⟩

end RecDeploy
