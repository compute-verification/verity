module

public import Definitions.NetworkCertifier.SampledAudit
public import Definitions.NetworkCertifier.SampledReg
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
# The registered inputs hold the registration's values

Definitions only. The sampled leak bounds count the accepted observation sequences at the inputs pinned to given values
at a statement's registered inputs (`AsReg`). A monitoring period's good event takes those values to be the
registration's: the extracted trace holds them at every registered input (`RegisteredInputsHeld`).

It is an open premise of a statement that reads it, and the conclusion of none. The registration commits its values
under the statement's roots (`RegOpens`), and an extracted trace that opens a registered input to another value makes,
with the registration's opening there, two passing openings of one position to different values: under the
commitment's binding the calls that check them collide, and `Breg` is that collision's finder. Where no input is
registered it holds of every trace.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- **The trace holds the registration's values at the registered inputs**: at the statement `s` and the extracted
witness `w`, the trace's committed value at every gate of `reg` is `ρ`'s. An open premise where a statement reads it,
which the registration's openings under the statement's roots (`RegOpens`) discharge under the commitment's binding;
at `reg = ∅` it holds of every trace. -/
def RegisteredInputsHeld (H : Msg → Msg) (Cm : CommitmentScheme Bool) (reg : Finset ℕ) (ρ : ℕ → Bool)
    (s : AuditStatement) (w : Wit) : Prop :=
  ∀ g ∈ reg, traceVal s.prog (w.val H Cm s) g = ρ g

end NetTiming
