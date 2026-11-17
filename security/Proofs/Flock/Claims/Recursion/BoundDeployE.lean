module

public import Proofs.Flock.Claims.Recursion.BoundDeploy
public import Proofs.Flock.Claims.DeployE
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

/-! The bound record at the element-level verifier's steps (`FlockDeploy.VE`, option (b)), where the compiled verifier's
acceptance is the deployed one; (P1) and (P4) at `VE`'s steps are not proved. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

theorem recBound_V_bdDeployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE nextE' : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE colsE' : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE wE' : Msg → ℝ)
    (join join' : List Msg → Msg) (rowIn rowIn' : Msg → Msg)
    (final final' : Msg → Option (List (List (Msg × Salt)) × Msg)) (w w' : Msg → ℝ) :
    (recBound rel b L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).V =
      (recBound rel b L (bdDeployed rel nextE' colsE' wE' join' rowIn' final' w').IVStar).V :=
  rfl

theorem deE_VE (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) :
    (deE rel FlockDeploy.VE.next colsE wE).V = FlockDeploy.VE :=
  rfl

end RecDeploy
