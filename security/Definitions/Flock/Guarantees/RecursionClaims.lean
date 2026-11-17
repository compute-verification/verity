module

public import Definitions.Claims.Recursion
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

/-! `RecursionClaimsStated` reads every claim of `Definitions.Claims.Recursion` so that the Lean check records its
text; the recursion's `KS` level is below the 100-bit floor at the deployed shape. -/

@[expose] public section

namespace Flock.Guarantees

/-- **The recursion's claims are stated**: the tuple of the compiled verifier's claims (`KS`, `Bits`, `Efficient`,
`Complete`, `ScopeMatches`, `ZK`, `Nontrivial`), the private variant's (`KSPriv`, `ZKPriv`), zero knowledge with the
reference prover's public stop and the pieces it reads (`ZKStop`, `StopDeploy`, `ZkDeploy.prover`,
`StopDeploy.halted`, `SimStop`, `εzkStop`), the claims at Flock as the inner protocol (`KSFlock`, `BitsFlock`), the
inner protocol's KS and its KS against provers that may abort (`Inner.KS`, `Inner.KSAbort`), which the proof reads,
the readable scope (`Scope`, `InScope`) and the decoders the claims quantify over (`decOK`); the KS at the
representation V* binds (`KSBound`, `KSPrivBound`), the inner claim it rests on (`Bridge`, with its premises
`Bound.PlainKSAbort`, `Bound.Binds` and `Bound.WeightCovers`), the last message's binding modulo what `E*` reads,
with the last step reading no last coin (`Bound.LastBindsEC`, `Bound.LastCoinFree`), which the recursion's proof
reads; the shared claims they instantiate (`Claims.KS`,
`ZKmalRx`, `Flock.ListWithin`, `Complete`, `ScopeMatches`, `ExtractRunsLe`, `ExtractRunsExpLe`, `Nontrivial`), the
named assumptions
(`StrictBirthday`, `ExpectedCR`, `Hm96Hiding`) and `trustedBase`, equal to itself. -/
def RecursionClaimsStated : Prop :=
  let claims := (@Claims.Recursion.KS, @Claims.Recursion.Bits, @Claims.Recursion.Efficient,
    @Claims.Recursion.Complete, @Claims.Recursion.ScopeMatches, @Claims.Recursion.ZK, @Claims.Recursion.Nontrivial,
    @Claims.Recursion.KSPriv, @Claims.Recursion.ZKPriv, @Claims.Recursion.ZKStop, @Claims.Recursion.StopDeploy,
    @Claims.Recursion.ZkDeploy.prover, @Claims.Recursion.StopDeploy.halted, @Claims.Recursion.SimStop,
    @Claims.Recursion.εzkStop,
    @Claims.Recursion.KSFlock, @Claims.Recursion.BitsFlock,
    @Claims.Recursion.Inner.KS, @Claims.Recursion.Inner.KSAbort,
    @Claims.Recursion.Scope, Claims.Recursion.InScope, @Claims.Recursion.decOK,
    @Claims.Recursion.KSBound, @Claims.Recursion.KSPrivBound, @Claims.Recursion.Bridge,
    @Claims.Recursion.Bound.PlainKSAbort, @Claims.Recursion.Bound.LastBindsEC, @Claims.Recursion.Bound.LastCoinFree,
    @Claims.KS, @Claims.ZKmalRx, @Claims.Flock.ListWithin, @Claims.Complete, @Claims.ScopeMatches,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, @Claims.Hm96Hiding, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
