module

public import Proofs.Flock.Masked
public import Proofs.Flock.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Deployed
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

/-! The sampled compiler over Flock's deployed masked record at SHA-512, with Flock's KS, the decoder's list bound and the commitment's binding proved. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Sampling (Opening pathCommit hm96Leaf regPath RowPlaces pathCommit_binds hm96Leaf_binds regPath_binds
  listWithin_deployZk)

variable (L : Compiled FlockDeploy.Coin (Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)))
  {own : Array _root_.Flock.Registered.Port} {at_ : List Msg → ℕ → Option (Msg × ℕ)}
  {row salt π : Opening Bool → Msg} {layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg}

theorem ks_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KS FlockDeploy.sha512 := by
  refine flockKS FlockDeploy.sha512 _ L _root_.Flock.SecurityProofs.MaskedKS (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

theorem ksReg_masked (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer)
    (hrow : RowPlaces row) (hS : L.Sessions FlockDeploy.sha512) : L.KSReg FlockDeploy.sha512 := by
  refine flockKSReg FlockDeploy.sha512 _ L _root_.Flock.SecurityProofs.MaskedKS (listWithin_deployZk _) ?_ hS
  rw [hcommit]
  exact pathCommit_binds (hm96Leaf_binds _ hrow salt) (regPath_binds _ own)

end Claims.SamplingGeneric

namespace SamplingDeploy

variable (c : Code) (iroots : Claims.Msg → Claims.Seen → ℕ → List Claims.Msg)

theorem ks_masked (hd : c.SampleDesign) (hr : c.Refuses FlockDeploy.sha512)
    (hi : c.InteriorFixed FlockDeploy.sha512 iroots) (hf : c.SampleFixed)
    (hruns : Claims.Sampling.SessionRuns FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c))
    (hnum : SessionsIn c.atInnerN₀)
    (hopens : Claims.Sampling.SessionOpens FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c) iroots) :
    (compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots).KS FlockDeploy.sha512 :=
  ks_compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots FlockDeploy.sha512
    _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _) (commit_binds c _) hd hr hi hf hruns
    (innerNumbers_deploy _ c hnum) hopens

theorem ksReg_masked (hd : c.SampleDesign) (hr : c.Refuses FlockDeploy.sha512)
    (hi : c.InteriorFixed FlockDeploy.sha512 iroots) (hf : c.SampleFixed)
    (hruns : Claims.Sampling.SessionRuns FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c))
    (hnum : SessionsIn c.atInnerN₀)
    (hopens : Claims.Sampling.SessionOpens FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation c) iroots) :
    (compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots).KSReg FlockDeploy.sha512 :=
  ksReg_compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) c iroots FlockDeploy.sha512
    _root_.Flock.SecurityProofs.MaskedKS (Claims.Sampling.listWithin_deployZk _) (commit_binds c _) hd hr hi hf hruns
    (innerNumbers_deploy _ c hnum) hopens

end SamplingDeploy
