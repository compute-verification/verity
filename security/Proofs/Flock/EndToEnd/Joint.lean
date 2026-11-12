module

public import Proofs.Flock.EndToEnd.Joint.Bound
public import Proofs.Flock.Soundness.Audit.StratifiedProj

/-! The sampled proofs' joint stratified bound, per sampling block and pooled over a period:
`Flock.SecurityProofs.JointStratified : Flock.Guarantees.JointStratified` and
`Flock.SecurityProofs.JointStratifiedHorizon : Flock.Guarantees.JointStratifiedHorizon` (`Joint/Bound.lean`), stated in
`Joint/Statement.lean`. The stratified sampling design one stratum at a time, for a consumer that samples one stratum by itself:
`Law.stratified_proj_escape` (`Soundness/Audit/StratifiedProj.lean`). -/

@[expose] public section
