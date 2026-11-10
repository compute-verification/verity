module

public import Proofs.Flock.Claims.Recursion.DeployedR
public import Proofs.Flock.Claims.Recursion.PastIn
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
import Proofs.Flock.Claims.Recursion.AccFits

/-!
# `MoreSlotsTrue` at the outer relation, unconditionally

`moreSlotsTrue` off its two leaves, both proved: `pastIn` (an output row's port is past the unit's inputs) and
`accFits` (a statement reads `acc` only below its instances). The statement is KSLin2LayR's `MoreSlotsTrue` at
`deployZkRec`, unfolded, so this file doesn't import KSLin2LayR.
-/

@[expose] public section

namespace RecDeploy.MoreSlots

open Flock
open Claims (Msg Out)
open Claims.Recursion.VStarB (readForms)

theorem moreSlotsTrue_deployed (st : Stage) :
    ∀ (x : Msg) (ms : List (Msg × Out)) (s : ℕ) (w : Claims.Flock.Wit) (r : XR),
      StagedPart st s x (staged st x ms s) →
      Claims.Flock.check (deployZkRec Flock.Readable.relation).parse (staged st x ms s) w = true →
        readX x = some r → ∀ p ∈ DeployedR.moreCalls r s,
          FlockDeploy.sha512 (p.1 ++ readForms w p.2.1) = readForms w p.2.2 :=
  moreSlotsTrue (pastIn st) accFits

end RecDeploy.MoreSlots
