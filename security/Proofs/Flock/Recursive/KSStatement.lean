module

public import Proofs.Flock.Claims.Recursion.DeployedBrings
public import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
public import Proofs.Flock.Claims.Recursion.KSLinks
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

/-! The recursive verifier's knowledge soundness at the deployed reader (`RecursiveKSAt`), stated at the deployment's parameters so that `RecursiveKS` is it at the deployed values (trusted). -/

@[expose] public section

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **Knowledge soundness of the recursive verifier at a compiled record** `D`: `KSLin2On` at SHA-512, the deployed
relation's bound at the reader `rd` (its steps `FlockDeploy.nextR rd`, its trimmed columns, V\*'s last message's reader
`final` and the weights `wE`, `w`), the deployed caps and the dense statements. `RecursiveKSAt` is this at the
deployment's record (`recBoundAt`); a statement about another compiled record (one that accepts every message, say)
names its own. -/
def RecursiveKSAtRec (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) : Prop :=
  Claims.Recursion.KSLin2On FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt

/-- **Knowledge soundness of the recursive verifier at the deployed reader**, at the deployment's parameters: for
every inner and outer decoder meeting the list-decoding spec, every dense statement and every prover, the probability
that the recursive verifier accepts and the extractor recovers no witness is at most `κ + √(γ·Pr[B])` plus the inner
verifier's bad-abort, forked-coin and first-coin terms (`KSLin2On`, through `RecursiveKSAtRec` at the deployment's
compiled record). -/
def RecursiveKSAt (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  RecursiveKSAtRec
    (RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR rest)) rd final wE w

/-- **Knowledge soundness of the recursive verifier at the deployed reader** (`RecursiveKSAt` at the deployed values),
for every other field of the brings and every record the challenger's live-coin part keeps. -/
def RecursiveKS : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSAt RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.OpenRows.elemR RecDeploy.DeployedR.wER
      RecDeploy.DeployedR.wR (RecDeploy.deployedB b) aux nOf hd recRest

def RecursiveKSLinkAtRec {Tb : Type} (D : Claims.Recursion.RecAt Coin) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ)
    (Q : Msg → Claims.Flock.Wit → Tb → Prop) (rows : Msg → List Claims.Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Claims.Flock.Wit) → Claims.Flock.Wit → Tb →
      List Msg) : Prop :=
  Claims.Recursion.KSLin2LinkOn FlockDeploy.sha512 D
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt Q rows ins

def RecursiveKSLinkAt {Tb : Type} (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (Q : Msg → Claims.Flock.Wit → Tb → Prop) (rows : Msg → List Claims.Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Claims.Flock.Wit) → Claims.Flock.Wit → Tb →
      List Msg) : Prop :=
  RecursiveKSLinkAtRec
    (RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR rest)) rd final wE w Q rows ins

end Flock.Guarantees
