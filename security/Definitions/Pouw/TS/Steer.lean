module

public import Definitions.Pouw.TS.Slots
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

/-! Steering a segmented consumer, STEER-PROOF's route B: zeroing coordinates spares a segment's work only once the segment has nothing left to sum. -/

@[expose] public section

namespace ResourceAccounting.TS

open Finset

variable {m G : ℕ}

def segActive (seg : Fin m → Fin G) (act Z : Finset (Fin m)) : Finset (Fin G) :=
  univ.filter fun g => ∃ i ∈ act, seg i = g ∧ i ∉ Z

def segCount (seg : Fin m → Fin G) (act : Finset (Fin m)) (g : Fin G) : ℕ :=
  (act.filter fun i => seg i = g).card

noncomputable def segConsumer (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) : Consumer m where
  act := act
  cost Z := c * (segActive seg act Z).card

def segUntouched (seg : Fin m → Fin G) (act Z : Finset (Fin m)) : Finset (Fin G) :=
  univ.filter fun g => (∃ i ∈ act, seg i = g) ∧ ∀ i ∈ Z, seg i ≠ g

noncomputable def adaptConsumer (seg : Fin m → Fin G) (act : Finset (Fin m)) (c : ℝ) : Consumer m where
  act := act
  cost Z := c * (segUntouched seg act Z).card

end ResourceAccounting.TS
