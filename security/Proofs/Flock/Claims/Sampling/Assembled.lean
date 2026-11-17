module

public import Proofs.Flock.Claims.DeployEfficient
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Runs
public import Proofs.Flock.Claims.Sampling.RunsExp
public import Proofs.Flock.Claims.Sampling.Select.Scope
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Universal
public import Proofs.Flock.SamplingBoundary
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

/-! The sampling block's claims at a deployment, with what remains as hypotheses. -/

@[expose] public section

namespace Claims.Sampling

def Deploy.mask {Coin Col T : Type} (D : Deploy Coin Col T) (Nonce Rest : Type) [Fintype Nonce] [Nonempty Nonce]
    [Fintype Rest] [Nonempty Rest] (tapeOf : Nonce × Rest → D.Tape) (S : SimLeak) : ZkDeploy Coin Col T :=
  { toDeploy := D, Nonce := fun _ => Nonce, Rest := fun _ => Rest, tapeOf := fun _ => tapeOf, S := S }

set_option synthInstance.maxSize 1024 in
theorem sampling_assembled_bind {T : Type}
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (H : Msg → Msg) (hbind : Binding D)
    (hslots : DrawSlots D) (hprog : ProgramScope D) (huniv : UniversalIs D) (hComplete : Complete H D) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ Complete H D :=
  ⟨scopeMatches_of D hslots hprog, hbind, Universal_of huniv, faultyBoundary, hComplete⟩

set_option synthInstance.maxSize 1024 in
theorem sampling_assembled {T : Type}
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (H : Msg → Msg)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hslots : DrawSlots D) (hprog : ProgramScope D) (huniv : UniversalIs D) (hComplete : Complete H D) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ Complete H D :=
  sampling_assembled_bind D H (Binding_of_hm96 D own at_ row salt π layer hcommit hrow) hslots hprog huniv hComplete

theorem listWithin_deployZk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Flock.ListWithin (FlockDeploy.deployZk rel).len (FlockDeploy.deployZk rel).enc (FlockDeploy.deployZk rel).θ
      (FlockDeploy.deployZk rel).decode := by
  rw [show (FlockDeploy.deployZk rel).decode = Claims.GS.zkDecodeRD from FlockDeploy.decode_zk rel]
  exact FlockDeploy.listWithin_zk rel

theorem listWithin_deployPlainTrim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Flock.ListWithin (FlockDeploy.deployPlainTrim rel).len (FlockDeploy.deployPlainTrim rel).enc
      (FlockDeploy.deployPlainTrim rel).θ (FlockDeploy.deployPlainTrim rel).decode := by
  rw [FlockDeploy.decode_plain rel]
  exact FlockDeploy.listWithin_plainTrim rel

end Claims.Sampling
