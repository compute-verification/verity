module

public import Proofs.Flock.Claims.Recursion.HpEarly
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
import Proofs.Flock.Claims.Recursion.VStarTraceAt

/-!
# hp before the last round on the honest run, with every part staged (`hpEarly_staged`)

`hpEarly_deployed` takes `TraceOkAt` and `CircOkAt`. At the deployed V\* both follow from every part of `x` being
staged (`VStarTraceAt.traceAt_staged`: `traceOkAt_staged` and `SlotsDeployed.circOkAt_staged`), so hp before the last
round needs only that and the fitted rows (`RowsFit`).
-/

@[expose] public section

namespace RecDeploy.HpEarly

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild

/-- **hp before the last round on the honest run, every part staged**: `hpEarly_run` with `TraceAt` from
`VStarTraceAt.traceAt_staged`. -/
theorem hpEarly_staged (mask : ℕ → ℕ → Bool) (I : InnerRec) (IV : Claims.Verifier FlockDeploy.Coin) (b : Brings)
    (w : Claims.Flock.Wit) {x : Msg} {xr : XR} (hx : readX x = some xr) (hk : IV.k (xIOf x) = RestRead.kI x)
    (ω : RecDeploy.Tape (finB b (A0 mask) I x))
    (r : Fin ((RecDeploy.V IV (finB b (A0 mask) I x).st).k x) → FlockDeploy.Coin) {s i : ℕ}
    (hs : s < nS x) (hi : i < RestRead.kI x - 1)
    (hfit : RowsFit I x w ω.1 ((((Claims.run FlockDeploy.sha512 (refP IV (finB b (A0 mask) I x) w)
      (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω r).msgs.take (IV.k (xIOf x))).map histOf).take
        (RestRead.kI x - 1)) i)
    {st : Stage} (hS : ∀ s sv c, partAt x s = some (sv, c) → ∃ xO, StagedPart st s x xO) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s
      (witFin (A0 mask) I x w ω.1 (((Claims.run FlockDeploy.sha512
        (refP IV (finB b (A0 mask) I x) w) (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω
          r).msgs.take (IV.k (xIOf x))).map histOf) s) i →
      ((((Claims.run FlockDeploy.sha512 (refP IV (finB b (A0 mask) I x) w)
        (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω r).msgs.map Prod.fst).getD i []).drop o).take
          m.length = m :=
  hpEarly_run mask I IV b w hx hk ω r hs hi hfit (VStarTraceAt.traceAt_staged hS mask)

end RecDeploy.HpEarly
