import FlockSoundness.Audit.Partitioning.Model
import FlockSoundness.Audit.Partitioning.Guarantees
import FlockSoundness.Audit.Partitioning.SecurityProofs.FreeCompute
import FlockSoundness.Audit.Partitioning.SecurityProofs.Exfiltration
import FlockSoundness.Audit.Partitioning.SecurityProofs.TwoStage
import FlockSoundness.Audit.Partitioning.Witness

/-!
# Optimal partitioning

Which division of a computation into proof units minimizes what the verifier spends to bound an adversary's gain, over
every draw law: free compute (`ConverseExact` to `UnequalCost`), exfiltration as `audit_exfiltration` bounds it
(`ExfiltrationOptimum`, `RefinementRule`), and the two-stage detection risk (`RiskConcave`, `Concentration`). The model
is in `Partitioning.Model`, the statements in `Partitioning.Guarantees`, their proofs in `Partitioning.SecurityProofs`,
and instances showing they are not vacuous in `Partitioning.Witness`. They read only the Flock-free part of `Audit`.
-/
