import Proofs.Flock.Claims.Recursion.HpEarly
import Proofs.Flock.Claims.Recursion.VStarPin

/-!
# hh at the deployed builder (`hh_of_fit`)

`honest_partAccepts_layR`'s `hh`: what part `s` holds whole of an inner round at `layR elemR`, read off its outer
witness `witFin`, is the run's message there. The rounds part `s` holds whole are round 0 at RecCap's part
(`capHold`: port `top`'s hm96 slot at each of its `N` instances) and the stream rounds with no tops that absorb a row
(`heldRows`: the one slot of the port reading it). Each held segment is an hm96 slot `([], hmCommit c q n)`.

* **Before the last round**: where the trace holds at part `s` (`TraceOk`, `SlotsOk`) and its held slots are fitted to
  the round's rows (the `k`-th slot's row, in part `s`'s port inputs, is round `i`'s `k`-th row, as many slots as
  rows), each slot reads its row's commit string (`readSegs_commit`), so the held message is the rows' commit strings
  in order (`flatMap_fit`), which is the run's message there wherever the inner prover's is (`hmsg`).
* **The last round** (`kI x - 1 = 2R + 1`, at the dense form's schedule, `schedMatches_dense`) is past the rep streams
  (`streamRound` is `none` at `2R`), so no part holds it (`msgAt_last`).

The assigner is the trace's at port inputs `g c pin`: `g := fun _ pin => pin` is `VStarTrace.A mask`, and
`g := fitPin` is the deployed `A0 mask` (`hh_A0`).
-/

namespace RecDeploy.HhHeld

open _root_.Flock
open Claims (Msg)
open Claims.Recursion.VStarB (LayoutB readSegs readForms)
open RecDeploy.VStarBuild
open RecDeploy.HpEarly

/-- **No part holds the last inner round**: at a statement whose rep schedule has `R` rounds, inner round `2R + 1` is
past the rep streams' `2R` rounds. -/
theorem msgAt_last {x : Msg} {r : XR} {rs : Array RecVStar.SchedRound} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (s : ℕ) :
    (DeployedR.layR OpenRows.elemR).msgAt x s (2 * rs.size + 1) = none := by
  have h : streamRound x (2 * rs.size) = none := by simp [streamRound, hr, hrs]
  show (layoutBTops (finOpen (DeployedR.restR OpenRows.elemR))).msgAt x s (2 * rs.size + 1) = none
  simp [layoutBTops, h]

/-- **hh at the trace assigner at port inputs `g c pin`**: where `x`'s inner rounds are `2 + 2R` at its rep schedule
(`hK`, `schedMatches_dense`), the trace holds at part `s`'s port inputs (`hT`), the slots part `s` holds of each round
before the last are fitted to that round's rows (`hfit`), and the run's message at each such round is its rows' commit
strings (`hmsg`, `hmsg_run`), every round part `s` holds whole at its outer witness `witFin` is the run's message there. -/
theorem hh_of_fit {mask : ℕ → ℕ → Bool} {g : Circuit → _root_.VStar.In.PartIn → _root_.VStar.In.PartIn}
    {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {s : ℕ}
    {ms : List Msg}
    (hK : ∃ (r : XR) (rs : Array RecVStar.SchedRound), readX x = some r ∧ repSched r.shape = some rs ∧
      RestRead.kI x = 2 + 2 * rs.size)
    (hT : ∀ st c src pin, partAt x s = some (st, c) →
      srcOf I x w ρ (hist.take (RestRead.kI x - 1)) = some src →
      _root_.VStar.In.partIn src (progOf c) st = .ok pin →
      VStarTrace.TraceOk c (g c pin) ∧ SlotsOk mask c (g c pin))
    (hfit : ∀ i < RestRead.kI x - 1, ∀ segs, (DeployedR.layR OpenRows.elemR).msgAt x s i = some segs →
      ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn),
        partAt x s = some (st, c) ∧ srcOf I x w ρ (hist.take (RestRead.kI x - 1)) = some src ∧
        _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
        segs.length = (roundRows I w ρ hist i).length ∧
        ∀ k (hk : k < segs.length), ∃ (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
          segs[k] = (([] : Msg), hmCommit c q n) ∧ c.ports.findIdx? (·.name == nm) = some q ∧
          portRow (g c pin) nm n = some rw ∧ rowHidden c nm n = some ps ∧
          (roundRows I w ρ hist i)[k]? = some rw)
    (hmsg : ∀ i < RestRead.kI x - 1, ms.getD i [] = (roundRows I w ρ hist i).flatMap rowCommit) :
    ∀ i < (DeployedR.layR OpenRows.elemR).rounds x, ∀ m,
      (DeployedR.layR OpenRows.elemR).held x s
        (witFin (fun c pin => VStarTrace.A mask c (g c pin)) I x w ρ hist s) i = some m → ms.getD i [] = m := by
  intro i hi m hm
  obtain ⟨r, rs, hr, hrs, hk⟩ := hK
  have hiK : i < RestRead.kI x := hi
  unfold Claims.Recursion.VStarB.LayoutB.held at hm
  split at hm
  · exact absurd hm (by simp)
  obtain ⟨segs, hseg, rfl⟩ := Option.map_eq_some_iff.1 hm
  by_cases hlt : i < RestRead.kI x - 1
  · obtain ⟨st, c, src, pin, hp, hsrc, hpi, hlen, hks⟩ := hfit i hlt segs hseg
    obtain ⟨hT1, hT2⟩ := hT st c src pin hp hsrc hpi
    unfold witFin
    rw [vstarWit_eq hp hsrc hpi, hmsg i hlt]
    try dsimp only
    unfold Claims.Recursion.VStarB.readSegs
    refine flatMap_fit _ _ _ _ hlen.symm fun k ha hb => ?_
    obtain ⟨nm, q, n, ps, rw, hsk, hq, hrw, hps, hrk⟩ := hks k hb
    rw [List.getElem?_eq_getElem ha] at hrk
    rw [Option.some.inj hrk, hsk, ← readSegs_commit hT1 hT2 hq hrw hps]
    simp only [Claims.Recursion.VStarB.readSegs, List.flatMap_cons, List.flatMap_nil, List.append_nil]
  · exfalso
    have he : i = 2 * rs.size + 1 := by omega
    rw [he, msgAt_last hr hrs] at hseg
    cases hseg

/-- **hh at the deployed assigner `A0 mask`**: `hh_of_fit` at `g := fitPin`, the trace at the fitted port inputs. -/
theorem hh_A0 {mask : ℕ → ℕ → Bool} {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape}
    {hist : List (Msg × Msg)} {s : ℕ} {ms : List Msg}
    (hK : ∃ (r : XR) (rs : Array RecVStar.SchedRound), readX x = some r ∧ repSched r.shape = some rs ∧
      RestRead.kI x = 2 + 2 * rs.size)
    (hT : ∀ st c src pin, partAt x s = some (st, c) →
      srcOf I x w ρ (hist.take (RestRead.kI x - 1)) = some src →
      _root_.VStar.In.partIn src (progOf c) st = .ok pin →
      VStarTrace.TraceOk c (fitPin c pin) ∧ SlotsOk mask c (fitPin c pin))
    (hfit : ∀ i < RestRead.kI x - 1, ∀ segs, (DeployedR.layR OpenRows.elemR).msgAt x s i = some segs →
      ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn),
        partAt x s = some (st, c) ∧ srcOf I x w ρ (hist.take (RestRead.kI x - 1)) = some src ∧
        _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
        segs.length = (roundRows I w ρ hist i).length ∧
        ∀ k (hk : k < segs.length), ∃ (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
          segs[k] = (([] : Msg), hmCommit c q n) ∧ c.ports.findIdx? (·.name == nm) = some q ∧
          portRow (fitPin c pin) nm n = some rw ∧ rowHidden c nm n = some ps ∧
          (roundRows I w ρ hist i)[k]? = some rw)
    (hmsg : ∀ i < RestRead.kI x - 1, ms.getD i [] = (roundRows I w ρ hist i).flatMap rowCommit) :
    ∀ i < (DeployedR.layR OpenRows.elemR).rounds x, ∀ m,
      (DeployedR.layR OpenRows.elemR).held x s (witFin (A0 mask) I x w ρ hist s) i = some m →
        ms.getD i [] = m :=
  hh_of_fit (g := fitPin) hK hT hfit hmsg

end RecDeploy.HhHeld
