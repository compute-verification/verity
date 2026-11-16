module

public import Proofs.Flock.Claims.Recursion.DeployNil
public import Proofs.Flock.Recursive.KSCoinTreeStatement
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

/-!
# `RecursiveKSDeployed`, proved

`RecursiveKS` at Hello-committed coins (`Flock.Guarantees.RecursiveKSDeployed`): `Claims.Recursion.ksLin2OnC_of_ksLin2On`
at `RecursiveKS`, whose compiled verifier answers each round with its coin's bytes or with nothing
(`RecDeploy.answersCoinOrNil_V`), and the bytes determine the coin (`Claims.Bridge.coin_bytes_injective`).
-/

@[expose] public section

namespace Flock.SecurityProofs

/-- **Knowledge soundness of the recursive verifier at Hello-committed coins.** -/
theorem RecursiveKSDeployed : Flock.Guarantees.RecursiveKSDeployed := by
  intro δ₁ b aux nOf hd recRest
  refine Claims.Recursion.ksLin2OnC_of_ksLin2On (RecursiveKS b aux nOf hd recRest) FlockDeploy.Coin.bytes
    Claims.Bridge.coin_bytes_injective (fun x => ?_) δ₁
  exact RecDeploy.answersCoinOrNil_V _ _ _ x

end Flock.SecurityProofs
