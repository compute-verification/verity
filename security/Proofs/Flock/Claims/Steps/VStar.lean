module

import Proofs.Flock.Claims.Finder
public import Proofs.Flock.Claims.KOf
public import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.Prob
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
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-! # What V*'s bridge obligation reads of `Claims.Recursion.VStar`, without its recursion imports. -/

@[expose] public section

namespace Claims.Recursion.VStar

open Claims

/-- A byte from up to eight bits, the low bit first. -/
def byte (bs : List Bool) : UInt8 := bs.foldr (fun b a => (if b then 1 else 0) + 2 * a) 0

/-- **Bits as bytes**: eight to a byte, the low bit first, a short last byte padded with zeros. Any other bit order is
this one at permuted positions, which a layout lists. -/
def bytesOf (bs : List Bool) : Msg := (List.range ((bs.length + 7) / 8)).map fun j => byte ((bs.drop (8 * j)).take 8)

/-- The bytes a witness's hidden values hold at the positions `ps`, in order. -/
def readAt (w : Flock.Wit) (ps : List ℕ) : Msg := bytesOf (ps.map fun p => w.hid.getD p false)

end Claims.Recursion.VStar
