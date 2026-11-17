module

public import Proofs.Flock.Claims.Recursion.HpEarly
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! `honest_partAccepts_layR`'s `hh` at the deployed builder: what a part holds whole of an inner round is the run's
message there (`hh_of_fit`, `hh_A0`). -/

@[expose] public section

namespace RecDeploy.HhHeld

open _root_.Flock
open Claims (Msg)
open Claims.Recursion.VStarB (LayoutB readSegs readForms)
open RecDeploy.VStarBuild
open RecDeploy.HpEarly

theorem msgAt_last {x : Msg} {r : XR} {rs : Array RecVStar.SchedRound} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (s : ℕ) :
    (DeployedR.layR OpenRows.elemR).msgAt x s (2 * rs.size + 1) = none := by
  have h : streamRound x (2 * rs.size) = none := by simp [streamRound, hr, hrs]
  show (layoutBTops (finOpen (DeployedR.restR OpenRows.elemR))).msgAt x s (2 * rs.size + 1) = none
  simp [layoutBTops, h]

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
