module

public import Definitions.Poub.Remote.SketchLine
public import Definitions.Poub.Assumptions.RemoteHeld
public import Definitions.Poub.Assumptions.RemoteOutside
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

/-! The receipts' sketch bound carried onto the memory line's coins (`SketchBoundLine`). -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sketch

/-- **The sketch bound on the memory line's coins.** On a trace over the line's coins whose other coins are a sketch
outcome and any record `O` of computation outside the node, the receipts' sketch bound on the trace read on the sketch
outcomes (`T.sketch O`) at `δ` gives `SketchBoundAt T O δ`. -/
def SketchBoundLine : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (n κ N m : ℕ)
    (T : Trace (ResourceAccounting.Remote.Sample.SampleCoins n κ (Outcome β Hs N m))) (O : Outside T) (δ : ℝ),
    SketchBound (T.sketch O) δ → SketchBoundAt T O δ

end ResourceAccounting.Guarantees.Remote
