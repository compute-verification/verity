module

public import Proofs.Flock.Claims.DeployEfficient
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Runs
public import Proofs.Flock.Claims.Sampling.Select.Scope
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Universal
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
import Proofs.Flock.Claims.KS.PlainLeafKS
import Proofs.Flock.Claims.Sampling.Assembled

/-!
# The sampling block's claims at the plain verifier, at SHA-512

`sampling_assembled_plain_sha512`: `sampling_assembled_plain` at SHA-512 at the deployed relation, with Flock's plain KS
at the trimmed record from `FlockDeploy.KS.plain_KS`, which takes no hypothesis.
-/

@[expose] public section

namespace Claims.Sampling

open FlockDeploy (sha512) in
set_option synthInstance.maxSize 1024 in
/-- **The sampling claims at the plain verifier, at SHA-512 at the deployed relation**: `sampling_assembled_plain` with
Flock's plain KS from `FlockDeploy.KS.plain_KS`. -/
theorem sampling_assembled_plain_sha512
    (D : Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace)
    (hflock : D.flock = FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation)
    (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns sha512 D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens sha512 D iroots) (hfix : InteriorFixed sha512 D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope sha512 D)
    (hopen : OpenCallsLe sha512 D) (huniv : UniversalIs D)
    (hComplete : Complete sha512 D) :
    ScopeMatches D ∧ Binding D ∧ Universal D ∧ FaultyBoundary ∧ KS sha512 D ∧ KSCount sha512 D ∧ Bits sha512 D ∧
      Efficient sha512 D FlockDeploy.plainStepPoly ∧ Complete sha512 D :=
  sampling_assembled_plain _root_.Flock.Readable.relation D sha512 hflock FlockDeploy.KS.plain_KS hε
    own at_ row salt π layer hcommit hrow hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks hopen huniv
    hComplete

end Claims.Sampling
