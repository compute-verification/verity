module

public import Proofs.Flock.Claims.Recursion.CallsRep
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
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
import VCVio.Prelude.Core

/-!
# The fast100 schedules' Ligerito calls

The kernel runs `fast100` at each size that has a schedule, and `fast100`'s loops are core's (`Std.Legacy.Range.forIn'`,
`Array`'s), so this module imports those in full. `CallsTop` reads the result from here: with core's definitions in full
there, its walks over a verdict's runs exceed their budget.
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

/-- **Every fast100 schedule's Ligerito makes at most `2^16` calls**, `CallsTop`'s `ligC` spelled out. -/
theorem fast100_calls : ∀ m < 36, (fast100 m).all (fun s => decide ((1 + (s.levels.size - 1)) * qAll s ≤ 2 ^ 16)) = true := by
  decide +kernel

end FlockDeploy.KS
