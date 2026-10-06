import Proofs.Flock.Soundness.Audit.Partitioning.Model
import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.FreeCompute
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfiltration
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.TwoStage
import Proofs.Flock.Soundness.Audit.Partitioning.Witness

/-!
# Optimal partitioning

Which division of a computation into proof units minimizes what the verifier spends to bound an adversary's gain, over
every draw law: free compute (`ConverseExact` to `UnequalCost`), exfiltration as `audit_exfiltration` bounds it
(`ExfiltrationOptimum`, `RefinementRule`), and the two-stage detection risk (`RiskConcave`, `Concentration`). The model
is in `Partitioning.Model`, the statements in `Partitioning.Guarantees`, their proofs in `Partitioning.SecurityProofs`,
and instances showing they are not vacuous in `Partitioning.Witness`. They read only the Flock-free part of `Audit`.
-/
