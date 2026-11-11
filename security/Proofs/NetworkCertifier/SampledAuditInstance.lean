module

public import Definitions.NetworkCertifier.SampledAudit
public import Proofs.Flock.Claims.DeployCode
public import Proofs.Flock.Claims.Sampling.Leaf
public import Proofs.Flock.Claims.Sampling.Runs
public import Proofs.Flock.Claims.Sampling.Select.Scope
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Universal
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
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.SamplingGeneric
import Proofs.NetworkCertifier.SampledAuditAccepted

/-!
# The sampled audit at the assembled sampling records

`SampledAuditAccepted` takes `Compiled.KS` at any compiled sampled verifier. At a sampling record whose sessions are a
Flock record with Flock's KS, the record as a compiled verifier (`Claims.SamplingGeneric.ofDeploy`) meets it: the
compiler at Flock (`Claims.SamplingGeneric.flockKS`), from the commitment's binding (`Binding_of_hm96`) and the
contract from the record's hypotheses (`Claims.SamplingGeneric.sessions_ofDeploy`). Composed, the sampled audit implies
`Accepted` at every such record (`sampledAuditAccepted_of`), and at the masked and plain records
(`sampledAuditAccepted_zk`, `sampledAuditAccepted_plain`), their decoders within the radius.
-/

@[expose] public section

namespace NetTiming.SampledAuditProof

open Claims Claims.Sampling
open Claims.SamplingGeneric (ofDeploy)

set_option synthInstance.maxSize 1024 in
/-- **The sampled audit implies `Accepted` at an assembled sampling record** over any Flock record `F`:
`SampledAuditAccepted` at the record as a compiled verifier, its `Compiled.KS` from `flockKS`. -/
theorem sampledAuditAccepted_of {T : Type}
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (H : Msg → Msg)
    (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (hflock : D.flock = F) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) :
    KSAccepted H (ofDeploy D iroots) L K S lay k := by
  subst hflock
  exact NetTiming.SecurityProofs.SampledAuditAccepted H _ L K S lay k
    (Claims.SamplingGeneric.flockKS H D.flock _ hKS hlist (Binding_of_hm96 D own at_ row salt π layer hcommit hrow H)
      (Claims.SamplingGeneric.sessions_ofDeploy D iroots H (drawLaw_of_slots D hslots)
        (refuses_of H D (scopeMatches_of D hslots hprog) hchecks) hfix hfixed hsessions hnum hopens))

set_option synthInstance.maxSize 1024 in
/-- **The sampled audit implies `Accepted` at the masked record** `deployZk`, its decoder the masked radius decoder
(`listWithin_deployZk`). -/
theorem sampledAuditAccepted_zk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace) (H : Msg → Msg)
    (hflock : D.flock = FlockDeploy.deployZk rel) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) :
    KSAccepted H (ofDeploy D iroots) L K S lay k :=
  sampledAuditAccepted_of D H (FlockDeploy.deployZk rel) hflock hKS (listWithin_deployZk rel) own at_ row salt π
    layer hcommit hrow hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks L K S lay k

set_option synthInstance.maxSize 1024 in
/-- **The sampled audit implies `Accepted` at the plain record** `deployPlainTrim`, its decoder the plain radius decoder
(`listWithin_deployPlainTrim`). -/
theorem sampledAuditAccepted_plain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (H : Msg → Msg)
    (hflock : D.flock = FlockDeploy.deployPlainTrim rel) (hKS : Claims.Flock.KS H (FlockDeploy.deployPlainTrim rel))
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) :
    KSAccepted H (ofDeploy D iroots) L K S lay k :=
  sampledAuditAccepted_of D H (FlockDeploy.deployPlainTrim rel) hflock hKS (listWithin_deployPlainTrim rel) own at_
    row salt π layer hcommit hrow hfixed hsessions hnum hslots iroots hopens hfix hprog hchecks L K S lay k

end NetTiming.SampledAuditProof
