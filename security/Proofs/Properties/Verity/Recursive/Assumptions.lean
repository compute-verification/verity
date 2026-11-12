module

public import Proofs.Flock.Claims.Recursion.RecRecord
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
# live-verifier at the recursive session: the records hold the live coins

Assumption (`live-verifier` at the recursive session), the premise the recursive component's property
(`Proofs.Properties.Verity.Recursive`) and the recursive audit (`Flock.Guarantees.RecursiveAuditC3`) take. It is
`live-verifier` as `FlockSoundness.Assumptions.InnerRecordCustody` states it at the inner session and
`Assumptions.Zk.RecordCustodyZKJ` at each session (the challenger's live-coin part's records of a session are the live history's),
restated at the check the compiled component can't make: the model verifier holds each V\* part's record to its own
live coins (`FlockDeploy.custody`), and the compiled component, which holds no live coin, reads that record as a
prover-sent blob.
-/

@[expose] public section

namespace Verity.Recursive

open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy

/-- **live-verifier** at the recursive session: the records hold the live coins. At each V\* session `s` of the run's
view `v`, wherever the outer verifier decides on session `s`'s messages and coins (`actO`), the live session the record
shows is the session's link message, first coin and round messages with their coins (`FlockDeploy.custody`). The model
verifier checks this itself. The compiled component can't, since the coins are the challenger's live-coin part's. It is admissible
until the component reads the coins through the challenger's live-coin part's signed device log; then this premise goes. -/
noncomputable def LiveCoins (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (v : Claims.Seen) : Prop :=
  ∀ s < nS x, ∀ i spec sts link c₁ mids f,
    actO (vstarAt st x (v.msgs.take (IV.k (xIOf x) + kS x)) s) ((session IV st x s v).msgs.map Prod.fst)
        ((session IV st x s v).msgs.map fun m => outCoin m.2) =
      .decide i spec sts link c₁ mids f →
    (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()

end Verity.Recursive
