module

public import Proofs.Flock.RecGeneric.AtFlockCommitted
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
# The recursive verifier's knowledge soundness at every statement, at Hello-committed coins (trusted)

`RecursiveKSEvery` reads the composite's run at coins flipped after each message. Deployed, the challenger's live-coin
part commits them at Hello in an hm96 coin tree (`coin-tree/hm96-sha512`) and opens each round's coin with its answer,
with no coin opened where a V\* session before the last one accepts and is answered with `[]`, as
`RecursiveKSDeployed` reads it. `RecursiveKSEveryDeployed` is `RecursiveKSEvery`'s bound at that run
(`Claims.RecGeneric.KSLinkOnC` at `RecursiveKSEvery`'s record, bound record, outer record, caps and every statement):
at every leaf hiding bound `δ₁ ≠ ∞`, every hm96 tree whose leaves hide at `δ₁` (`Claims.Hm96Hiding`) has a simulator
`S` such that, for every prover that reads the root and the openings, the committed run accepts at most as often as
the composite's extractor, run on the prover with `S`'s root and openings, recovers a witness, plus
`RecursiveKSEvery`'s bound at that prover and `2 · k(x) · δ₁`.
-/

@[expose] public section

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open scoped ENNReal

/-- **`RecursiveKSEveryAtRec` at Hello-committed coins**, at the deployed reader, weights and last part's pieces, the
brings `b` and the challenger's live-coin part's record: `KSLinkOnC` where `RecursiveKSEveryAtRec` has `KSLinkOn`, at
every statement and the leaf hiding bound `δ₁`. -/
def RecursiveKSEveryDeployedAt (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) (δ₁ : ℝ≥0∞) : Prop :=
  Claims.RecGeneric.KSLinkOnC FlockDeploy.sha512
    (RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
      RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
      RecDeploy.DeployedR.wR)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep (fun _ => True) δ₁

/-- **Knowledge soundness of the recursive verifier at every statement, at Hello-committed coins**
(`RecursiveKSEveryDeployedAt` at the deployed brings' setup), for every leaf hiding bound `δ₁`, every other field of the
brings and every record the challenger's live-coin part keeps. `Claims.RecGeneric.ksLinkOnC_of_ksLinkOn` at
`Flock.SecurityProofs.RecursiveKSEvery` proves it (`Proofs.Flock.Recursive.KSEveryCoinTreeProved`), the compiled
verifier answering each round with its coin's bytes or with nothing (`RecDeploy.answersCoinOrNil_V`). -/
def RecursiveKSEveryDeployed : Prop :=
  ∀ (δ₁ : ℝ≥0∞) (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSEveryDeployedAt (RecDeploy.deployedB b) aux nOf hd recRest δ₁

end Flock.Guarantees
