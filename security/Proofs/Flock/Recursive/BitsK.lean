module

public import Proofs.Flock.Claims.Recursion.KappaEpsDeployed
public import Proofs.Flock.Recursive.KSStatement
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
import Proofs.Flock.Recursive.KSProved

/-! `Pr[acc ∧ ¬R] ≤ κ + t·ε` at the deployed reader, stated at the brings `b` because a statement writing `{ deployedB b with … }` elaborates past the heartbeat limit. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)

def RecursiveBitsKAt (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  Claims.Recursion.BitsLin2KOn FlockDeploy.sha512
    (recBoundAt _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
      (DeployedR.layR OpenRows.elemR))
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR)
    (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt

def RecursiveBitsKBoundAt (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  Claims.Recursion.BitsKAt FlockDeploy.sha512
    (recBoundAt _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
      (DeployedR.layR OpenRows.elemR))
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
      FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR)
    (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep
    (fun x => DenseAt x ∧ SizesAt { b with tr := recTr aux nOf hd recRest } x)
    (2 ^ 32) (2 ^ 32) (2 ^ (-(167.7 : ℝ))) (2 ^ (-(174.95 : ℝ)))

theorem recursiveBitsKAt_of (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (h : _root_.Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR DeployedR.wER
      DeployedR.wR b aux nOf hd recRest) :
    RecursiveBitsKAt b aux nOf hd recRest :=
  Claims.Recursion.bitsLin2KOn_of_ksLin2On
    (sessionAccepts
      (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
        FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR).IVStar
      (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck
      { b with tr := recTr aux nOf hd recRest, rd := (DeployedR.layR OpenRows.elemR).readers }) h

theorem recursiveBitsKBoundAt_of (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (h : RecursiveBitsKAt b aux nOf hd recRest) : RecursiveBitsKBoundAt b aux nOf hd recRest := by
  refine Claims.Recursion.bitsKAt_of h (fun _ hx => hx.1) ?_
  rintro decI x n v ⟨-, ⟨s, spec, sts, hin, h1⟩, hO, hS⟩ hU hV
  exact kappa_eps_deployed _root_.Flock.Readable.relation { b with tr := recTr aux nOf hd recRest }
    (DeployedR.layR OpenRows.elemR) (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER
    FlockDeploy.joinR rowInDeployed DeployedR.finalR DeployedR.wR decI hin h1 hO hS hU hV

theorem recursiveBitsK (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveBitsKAt (deployedB b) aux nOf hd recRest :=
  recursiveBitsKAt_of (deployedB b) aux nOf hd recRest (_root_.Flock.SecurityProofs.RecursiveKS b aux nOf hd recRest)

theorem recursiveBitsK_deployed (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveBitsKBoundAt (deployedB b) aux nOf hd recRest :=
  recursiveBitsKBoundAt_of (deployedB b) aux nOf hd recRest (recursiveBitsK b aux nOf hd recRest)

end RecDeploy
