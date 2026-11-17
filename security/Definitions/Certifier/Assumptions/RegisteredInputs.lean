module

public import Definitions.Certifier.SampledAudit
public import Definitions.Certifier.SampledReg
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

/-! The registered inputs hold the registration's values (`RegisteredInputsHeld`), an open premise of the statements that read it. -/

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
