module

public import Definitions.Claims
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

/-! `ClaimsStated` reads every claim of `Definitions.Claims` so that the Lean check records its text; `Nontrivial`
rests on a test against recorded sessions, not a Lean proof. -/

@[expose] public section

namespace Flock.Guarantees

/-- **The claims are stated**: the tuple of Flock's claims (`KS`, `Complete`, `ScopeMatches`, `Bits`, `Efficient`,
`Nontrivial`), the list-decoding spec they read (`Flock.ListWithin`), the shared claims they instantiate (`Claims.KS`,
`KSLin`, `Complete`, `ScopeMatches`, `ZKmalR`, `SimCost`, `Bits`, `ExtractRunsLe`, `ExtractRunsExpLe`, `ERepCost`,
`Nontrivial`), the named assumptions (`StrictBirthday`, `ExpectedCR`, `Hm96Hiding`) and `trustedBase`,
equal to itself. `ZK` left it for `Flock.SecurityProofs.MaskedZK`, which states it at the deployed masked record, the
only kind of record `ZK` reads. `KS` stays: `Flock.SecurityProofs.MaskedKS` states it at the masked record only, and no
guarantee states it at the plain record (`FlockDeploy.deployPlain`, whose trim is the recursion's element-level
record). -/
def ClaimsStated : Prop :=
  let claims := (@Claims.Flock.KS, @Claims.Flock.Complete, @Claims.Flock.ScopeMatches,
    @Claims.Flock.Bits, @Claims.Flock.Efficient, @Claims.Flock.Nontrivial, @Claims.Flock.ListWithin,
    @Claims.KS, @Claims.KSLin, @Claims.Complete, @Claims.ScopeMatches, @Claims.ZKmalR, @Claims.SimCost, @Claims.Bits,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.ERepCost, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, @Claims.Hm96Hiding, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
