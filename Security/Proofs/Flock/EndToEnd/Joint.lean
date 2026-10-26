import Proofs.Flock.EndToEnd.Joint.Bound
import Proofs.Flock.Soundness.Audit.StratifiedProj

/-! The sampled proofs' joint stratified bound, per sampling block and pooled over a period:
`Flock.SecurityProofs.JointStratified : Flock.Guarantees.JointStratified` and
`Flock.SecurityProofs.JointStratifiedHorizon : Flock.Guarantees.JointStratifiedHorizon` (`Joint/Bound.lean`), stated in
`Joint/Statement.lean`. The stratified law one stratum at a time, for a consumer that draws one stratum by itself:
`Law.stratified_proj_escape` (`Soundness/Audit/StratifiedProj.lean`). -/
