import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.Flock.Claims.Sampling.Assembled

/-!
# The sampled audit at the assembled sampling records

`SampledAuditAccepted` takes `Claims.Sampling.KS` at any deployment; `Claims.Sampling.sampling_assembled_of` proves it,
as a conjunct, at any deployment whose sessions are a Flock record with Flock's KS. Composed, the sampled audit implies
`Accepted` at every such deployment (`sampledAuditAccepted_of`), and at the masked and plain records
(`sampledAuditAccepted_zk`, `sampledAuditAccepted_plain`) as `sampling_assembled` and `sampling_assembled_plain` are.
-/

namespace NetTiming.SampledAuditProof

open Claims Claims.Sampling

set_option synthInstance.maxSize 1024 in
/-- **The sampled audit implies `Accepted` at an assembled sampling deployment** over any Flock record `F`:
`SampledAuditAccepted` at the `KS` conjunct of `sampling_assembled_of`. -/
theorem sampledAuditAccepted_of {T : Type}
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (H : Msg → Msg)
    (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (p : Polynomial ℕ)
    (hflock : D.flock = F) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode)
    (hcount : ∀ os, (F.dec os).2 ≤ p.eval (Claims.Flock.wordLen os))
    (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D)
    (hopen : OpenCallsLe H D) (huniv : UniversalIs D)
    (hComplete : Complete H D) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) :
    KSAccepted H D L K S lay k :=
  NetTiming.SecurityProofs.SampledAuditAccepted H D L K S lay k
    (sampling_assembled_of D H F p hflock hKS hlist hcount hε own at_ row salt π layer hcommit hrow hfixed hsessions
      hnum hslots iroots hopens hfix hprog hchecks hopen huniv hComplete).2.2.2.2.1

set_option synthInstance.maxSize 1024 in
/-- **The sampled audit implies `Accepted` at the masked record** `deployZk`, its decoder facts as
`sampling_assembled` takes them. -/
theorem sampledAuditAccepted_zk (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace) (H : Msg → Msg)
    (hflock : D.flock = FlockDeploy.deployZk rel) (hKS : Claims.Flock.KS H (FlockDeploy.deployZk rel))
    (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D)
    (hopen : OpenCallsLe H D) (huniv : UniversalIs D)
    (hComplete : Complete H D) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) :
    KSAccepted H D L K S lay k :=
  sampledAuditAccepted_of D H (FlockDeploy.deployZk rel) FlockDeploy.zkStepPoly hflock hKS
    (by
      rw [show (FlockDeploy.deployZk rel).decode = Claims.GS.zkDecodeRD from FlockDeploy.decode_zk rel]
      exact FlockDeploy.listWithin_zk rel)
    (FlockDeploy.efficient_zk H rel).2.2.1 hε own at_ row salt π layer hcommit hrow hfixed hsessions hnum hslots
    iroots hopens hfix hprog hchecks hopen huniv hComplete L K S lay k

set_option synthInstance.maxSize 1024 in
/-- **The sampled audit implies `Accepted` at the plain record** `deployPlainTrim`: `SampledAuditAccepted` at the `KS`
conjunct of `sampling_assembled_plain`. -/
theorem sampledAuditAccepted_plain (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace) (H : Msg → Msg)
    (hflock : D.flock = FlockDeploy.deployPlainTrim rel) (hKS : Claims.Flock.KS H (FlockDeploy.deployPlainTrim rel))
    (hε : ∀ x, 0 ≤ D.εtable x)
    (own : Array _root_.Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ))
    (row salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hcommit : D.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hfixed : DrawFixed D) (hsessions : SessionRuns H D) (hnum : InnerNumbers D) (hslots : DrawSlots D)
    (iroots : Msg → Seen → ℕ → List Msg) (hopens : SessionOpens H D iroots) (hfix : InteriorFixed H D iroots)
    (hprog : ProgramScope D) (hchecks : ChecksScope H D)
    (hopen : OpenCallsLe H D) (huniv : UniversalIs D)
    (hComplete : Complete H D) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) :
    KSAccepted H D L K S lay k :=
  NetTiming.SecurityProofs.SampledAuditAccepted H D L K S lay k
    (sampling_assembled_plain rel D H hflock hKS hε own at_ row salt π layer hcommit hrow hfixed hsessions hnum hslots
      iroots hopens hfix hprog hchecks hopen huniv hComplete).2.2.2.2.1

end NetTiming.SampledAuditProof
