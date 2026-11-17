module

import Proofs.Flock.Claims.Complete.Encode
public import Proofs.Flock.Claims.DeployInputs
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
import Std.Tactic.BVDecide.Normalize.Prop
import VCVio.Prelude.Core

/-! # What the plain KS proofs read of `Claims.Recursion.VarLen`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

/-- `encV` read off the front of a message. -/
def rdVL : Claims.Msg → Option (ℕ × Claims.Msg)
  | [] => none
  | b :: t => if b.toNat < 128 then some (b.toNat, t) else (rdVL t).map fun p => (b.toNat - 128 + 128 * p.1, p.2)

def rdV : Rd ℕ := fun m => rdVL m

def rdVLs {α : Type} (r : Rd α) : Rd (List α) := do rdListN r (← rdV)

def rdVB : Rd ByteArray := do
  let n ← rdV
  pure ⟨(← rdTake n).toArray⟩

end FlockDeploy
