module

public import Proofs.Flock.Claims.Complete.Zk.Session
public import Proofs.Flock.Claims.Complete.Honest
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

/-! Flock's honest prover at the deployed masked verifier (`honestPZk`), whose reading of the witness is the relation's
only at one table; the masked record's reference prover is `ZkTables.honestPR`. -/

@[expose] public section

namespace FlockComplete

open Flock FlockDeploy

/-- **The honest masked prover's program** on the statement `x`, the witness `w` and the tape `ω`; the empty message
where the statement sets up no session or the prover cannot run it, which the verifier refuses. -/
def progZk (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) : PI Claims.Msg :=
  match setupOf true x with
  | none => .ret []
  | some (_, spec, sts) => do
    match ← (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w) ω).run with
    | .ok (record, proofs) => pure (Final.encode ⟨x, record, proofs⟩)
    | .error _ => pure []

/-- **Flock's honest prover** at the masked verifier: at each history, its program fed the verifier's replies. -/
def honestPZk (w : Claims.Flock.Wit) : Claims.Prover TapeZk :=
  ⟨fun x hist ω => feed (progZk x w ω) (hist.map Prod.snd)⟩

end FlockComplete
