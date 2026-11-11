module

public import Proofs.Flock.Claims.Recursion.ResAccEnd
public import Proofs.Flock.Claims.Recursion.ResFramePorts
public import Proofs.Flock.Claims.Recursion.ResLowerDD
public import Proofs.Flock.Claims.Recursion.RestOpenedSized
public import Proofs.Flock.Claims.Recursion.RestVerdictBasesHold
public import Proofs.Flock.Recursive.KSDeployed
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

/-!
# KS's residual leaf at the deployed readers (`hres_deployed`)

`hres_deployed` is the `ResLoweredPD` that `recursiveKSAt_of_leaves_deployed` takes as `hres`: at the deployed reader
`DeployedR.rdR`, its last message's reading `DeployedR.finalR`, the elements `OpenRows.elemR`, the deployed brings
`bB (deployedB b) …`, the dense form, `recRead` and `cmtTr recRead`. It is `resLowered_DB` there, with

* the base checks (`basesHold_spec`, at every reader);
* `ReadsAt` (`RestElem.readsAt_elemR_closed`);
* P2, the frame ports (`framePorts_D`);
* P3, the accumulator ports (`accPorts_deployed`: the steps half `StepsWG … XsW` and the ends half `endsW_of_reg`).

It takes no hypothesis. `hres_deployed_all` is the form `recursiveKS_of_resLowered` takes, at the deployed weights, and
`recursiveKS_deployed` is `RecursiveKS` from it.
-/

@[expose] public section

namespace RecDeploy.ResLower

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

/-- **KS's residual leaf at the deployed readers**: `ResLoweredPD` at `DeployedR.rdR`, `DeployedR.finalR`,
`OpenRows.elemR` and the deployed brings, with no hypothesis. -/
theorem hres_deployed (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) :
    ResLoweredPD DeployedR.rdR wE DeployedR.finalR w (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR
      Claims.Recursion.Form.Dense recRead (cmtTr recRead) DecompBR.resPD DecompBR.ownRD
      (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead :=
  resLowered_DB DeployedR.rdR key wE DeployedR.finalR w b aux nOf hd recRest OpenRows.elemR
    (basesHold_spec DeployedR.rdR) RestElem.readsAt_elemR_closed
    (framePorts_D DeployedR.rdR wE w _ OpenRows.elemR)
    (accPorts_deployed key wE w b aux nOf hd recRest (cmtTr recRead))

/-- **`hres_deployed` for every brings and record**, at the deployed weights and one key: the hypothesis
`recursiveKS_of_resLowered` takes. -/
theorem hres_deployed_all (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) :
    ∀ (b : Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      ∃ key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ,
        ResLoweredPD DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
          (DecompBR.bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead
          (cmtTr recRead) DecompBR.resPD DecompBR.ownRD (Rows.elemReadC (Top.accRC DeployedR.rdR key))
          Rows.resVRead :=
  fun b aux nOf hd recRest => ⟨key, hres_deployed DeployedR.wER DeployedR.wR b aux nOf hd recRest key⟩

/-- **`RecursiveKS`** from KS's residual leaf (`hres_deployed_all`, at any key). -/
theorem recursiveKS_deployed : Flock.Guarantees.RecursiveKS :=
  Flock.SecurityProofs.recursiveKS_of_resLowered (hres_deployed_all fun _ _ _ _ => (0, 0, 0, 0))

end RecDeploy.ResLower
