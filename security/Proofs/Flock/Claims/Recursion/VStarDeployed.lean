module

public import Proofs.Flock.Claims.Recursion.HpLast
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

/-! The deployed reference prover's pieces with the last round as the deployed inner prover sends it, `finMsg` rather than an hm96 commit string (`refPartsFin`). -/

@[expose] public section

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)
open RecDeploy.HpLast

/-- **The outer witness at the deployed builder**: `vstarWit` at the inner rounds before the last. -/
def witFin (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) (s : ℕ) : Claims.Flock.Wit :=
  vstarWit A I x w ρ (hist.take (RestRead.kI x - 1)) s

/-- **The deployed inner prover at `x`**: `InnerRec.P` before the last round, `finMsg` at it. -/
def PFin (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) : Claims.Prover I.Tape :=
  ⟨fun xI h ρ => if h.length + 1 = RestRead.kI x then .ret (finMsg A I x w ρ h) else (I.P w).next xI h ρ⟩

/-- **The deployed reference prover's pieces at the compiled statement `x`**: the inner prover `PFin`, the staged
values' digests `refPartsAt`'s, and the outer witnesses `witFin`. -/
def refPartsFin (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg) : RefParts where
  TI := I.Tape
  PI := PFin A I x
  digests := (refPartsAt A I x).digests
  wit := witFin A I x

theorem PFin_last (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (xI : Msg) (h : List (Msg × Msg)) (ρ : I.Tape) (hl : h.length + 1 = RestRead.kI x) :
    (PFin A I x w).next xI h ρ = .ret (finMsg A I x w ρ h) := by
  simp [PFin, hl]

theorem PFin_before (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (xI : Msg) (h : List (Msg × Msg)) (ρ : I.Tape) (hl : h.length + 1 ≠ RestRead.kI x) :
    (PFin A I x w).next xI h ρ = (I.P w).next xI h ρ := by
  simp [PFin, hl]

theorem witFin_take (A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit) (I : InnerRec) (x : Msg)
    (w : Claims.Flock.Wit) (ρ : I.Tape) (hist h : List (Msg × Msg)) (hh : hist.take (RestRead.kI x - 1) = h)
    (s : ℕ) : witFin A I x w ρ hist s = vstarWit A I x w ρ h s := by
  rw [witFin, hh]

theorem hpLast_fin {A : Circuit → _root_.VStar.In.PartIn → Claims.Flock.Wit} {I : InnerRec} {x : Msg}
    {w : Claims.Flock.Wit} {ρ : I.Tape} {hist : List (Msg × Msg)} {s : ℕ} (hs : s < nS x) {ms : List Msg}
    (hfin : ms.getD (RestRead.kI x - 1) [] = finMsg A I x w ρ (hist.take (RestRead.kI x - 1))) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s (witFin A I x w ρ hist s) (RestRead.kI x - 1) →
      ((ms.getD (RestRead.kI x - 1) []).drop o).take m.length = m :=
  hpLast hs hfin

end RecDeploy.VStarBuild
