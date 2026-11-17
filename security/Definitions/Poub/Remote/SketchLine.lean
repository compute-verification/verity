module

public import Definitions.Poub.Remote.Sample
public import Definitions.Poub.Remote.Outside
public import Definitions.Poub.Remote.Sketch
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

/-! The memory line's coins read as a sketch outcome (`lineCoins`). -/

@[expose] public section

namespace ResourceAccounting.Remote.Sketch

variable {β Hs : Type} {n κ N m : ℕ}

/-- A sketch outcome whose prover's coins carry the windows' openings coins, read as the memory line's coins: the other
coins the prover's own, the store and SHA-512's outputs, and the openings coins. -/
def lineCoins (ω : Outcome (β × (Fin n → Pous.Bits κ)) Hs N m) :
    ResourceAccounting.Remote.Sample.SampleCoins n κ (Outcome β Hs N m) :=
  ((ω.1.1, ω.2.1, ω.2.2), ω.1.2)

end ResourceAccounting.Remote.Sketch

namespace ResourceAccounting.Remote.Held.Trace

open ResourceAccounting.Remote.Sketch

variable {β Hs : Type} {n κ N m : ℕ}

/-- The memory line's trace over coins whose other coins are a sketch outcome, read on the sketch model's outcomes
(`lineCoins`), with the registered blocks a device outside the node computes on at a tick counted with the chips'. -/
def sketch (T : Trace (ResourceAccounting.Remote.Sample.SampleCoins n κ (Outcome β Hs N m))) (O : Outside T) :
    Trace (Outcome (β × (Fin n → Pous.Bits κ)) Hs N m) where
  blocks := T.blocks
  passes := T.passes
  accepts ω := T.accepts (lineCoins ω)
  tick ω := T.tick (lineCoins ω)
  plan ω := T.plan (lineCoins ω)
  correct ω := T.correct (lineCoins ω)
  mem ω := T.mem (lineCoins ω)
  chip ω t := T.chip (lineCoins ω) t ∪ O.blocks (lineCoins ω) t
  rebuilt ω := T.rebuilt (lineCoins ω)
  disk ω := T.disk (lineCoins ω)
  link ω := T.link (lineCoins ω)

end ResourceAccounting.Remote.Held.Trace
