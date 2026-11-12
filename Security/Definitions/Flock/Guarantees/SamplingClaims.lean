module

public import Definitions.Claims.Sampling
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
# The sampling claims are stated (trusted)

`SamplingClaimsStated` reads every claim of `Definitions.Claims.Sampling`, the shared claims they instantiate, and the
named assumptions and trusted base they use, so the Lean check records each one's text (and every definition it reads)
under `reads`: a changed claim is a changed record, which is reported when it lands. It says nothing about the
verifier. A claim leaves it when a guarantee states that claim at the deployed sampled verifier, and the guarantee's own
record holds the text from then on.
-/

@[expose] public section

namespace Flock.Guarantees

/-- **The sampling claims are stated**: the tuple of the sampling claims (`Binding`, `Universal`, `Complete`,
`ScopeMatches`, `ZK`, `Nontrivial`), the claims the compiled sampled verifier's knowledge soundness, bits and
efficiency are stated in (`KSLinOn`, `BitsLin`, `ExtractOwnLe`) and the shared ones they instantiate (`ZKmalRx`,
`Flock.ListWithin`, `Complete`, `ScopeMatches`, `ExtractRunsLe`, `ExtractRunsExpLe`, `Nontrivial`), the named
assumptions (`StrictBirthday`, `ExpectedCR`) and `trustedBase`, equal to itself. -/
def SamplingClaimsStated : Prop :=
  let claims := (@Claims.Sampling.Binding, @Claims.Sampling.Universal, @Claims.Sampling.Complete,
    @Claims.Sampling.ScopeMatches, @Claims.Sampling.ZK, @Claims.Sampling.Nontrivial,
    @Claims.Sampling.KSLinOn, @Claims.Sampling.BitsLin, @Claims.Sampling.ExtractOwnLe,
    @Claims.ZKmalRx, @Claims.Flock.ListWithin, @Claims.Complete, @Claims.ScopeMatches,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
