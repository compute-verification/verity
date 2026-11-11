module

public import Definitions.Claims
public import Mathlib.Algebra.Polynomial.Eval.Defs
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

/-!
# A counted program against a size of the proof's choosing

`DecoderComplexity decode counted size p`: `counted` computes what `decode` does, and its count is at most `p(size)`.
The proofs bound the deployed decoder's pieces with it, each at a size of its own (the root finder's at its input's
length, a level's decoder at its opened data's), on the way to `Claims.Flock.Efficient`, which reads the deployed
decoder's own count (`Claims.Flock.Deploy.dec`) at the opened word's encoded length (`Claims.Flock.wordLen`). It is no
claim: with `counted` and `size` its own arguments, a count of `0` meets it.
-/

@[expose] public section

namespace Claims

/-- **A counted program at a size**: `counted` computes what `decode` does, and the steps it reports are at most
`p(size)`. -/
def DecoderComplexity {I T : Type} (decode : I → T) (counted : I → T × ℕ) (size : I → ℕ) (p : Polynomial ℕ) : Prop :=
  ∀ i, (counted i).1 = decode i ∧ (counted i).2 ≤ p.eval (size i)

end Claims
