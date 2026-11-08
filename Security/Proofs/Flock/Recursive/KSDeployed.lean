module

public import Proofs.Flock.Claims.Recursion.DeployedBringsLam
public import Proofs.Flock.Claims.Recursion.ResLowerDD
public import Proofs.Flock.Recursive.KS
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
import Proofs.Flock.Claims.Recursion.RestFinal
import Proofs.Flock.Claims.Recursion.RestOpenedSized

/-!
# `RecursiveKSAt` from chain B's leaves at the deployed brings

`recursiveKSAt_of_leaves_deployed` is `recursiveKSAt_of_leaves` at `RecDeploy.deployedB b`, whose setup is the
compiled verifier's own evaluations (`deployedSetup`: check 5's and the registered reads'), with `hset` discharged by
`resSetup_deployedB`, the last message's reading by `finalReads` (`hF`) and `readsAt_elemR_closed` (`hRd`,
which also closes the cap tie's `sideCapR_leaf`). The other fields of the brings stay `b`'s.
-/

@[expose] public section

namespace Flock.SecurityProofs

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)
open RecDeploy RecDeploy.DecompBR RecDeploy.RestLow

/-- **`RecursiveKSAt` from chain B's leaves at the deployed brings**: `recursiveKSAt_of_leaves` at `deployedB b`, with
the setup's listing (`hset`) proved (`resSetup_deployedB`) and the last message's reading (`hF`, `hRd`) closed
(`finalReads`, `readsAt_elemR_closed`). Its other leaves are `recursiveKSAt_of_leaves`', at
`deployedB b`. -/
theorem recursiveKSAt_of_leaves_deployed (wE w : Msg → ℝ)
    (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hres : ResLower.ResLoweredPD DeployedR.rdR wE DeployedR.finalR w (bB (deployedB b) aux nOf hd recRest)
      OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead) resPD ownRD
      (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead) :
    Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w (deployedB b) aux nOf hd
      recRest :=
  recursiveKSAt_of_leaves wE w (deployedB b) aux nOf hd recRest key (RestRead.finalReads OpenRows.elemR)
    RestElem.readsAt_elemR_closed hres (resSetup_deployedB b)

/-- **`RecursiveKS` from chain B's residual leaf**: `recursiveKSAt_of_leaves_deployed` at the deployed weights, for every
brings and record, wherever the residual's lowering (`ResLoweredPD`, at each part's decision) holds there at some key. -/
theorem recursiveKS_of_resLowered
    (hres : ∀ (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      ∃ key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ,
        ResLower.ResLoweredPD DeployedR.rdR DeployedR.wER DeployedR.finalR DeployedR.wR
          (bB (deployedB b) aux nOf hd recRest) OpenRows.elemR Claims.Recursion.Form.Dense recRead (cmtTr recRead) resPD
          ownRD (Rows.elemReadC (Top.accRC DeployedR.rdR key)) Rows.resVRead) :
    Flock.Guarantees.RecursiveKS := by
  intro b aux nOf hd recRest
  obtain ⟨key, h⟩ := hres b aux nOf hd recRest
  exact recursiveKSAt_of_leaves_deployed DeployedR.wER DeployedR.wR b aux nOf hd recRest key h

end Flock.SecurityProofs
