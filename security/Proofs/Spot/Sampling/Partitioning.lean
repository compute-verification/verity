module

public import Proofs.Spot.Sampling.Partitioning.Model
public import Proofs.Spot.Sampling.Partitioning.Guarantees
public import Proofs.Spot.Sampling.Partitioning.SecurityProofs.FreeCompute
public import Proofs.Spot.Sampling.Partitioning.SecurityProofs.Exfiltration
public import Proofs.Spot.Sampling.Partitioning.SecurityProofs.TwoStage
public import Proofs.Spot.Sampling.Partitioning.SecurityProofs.Unequal.Witnesses
public import Proofs.Spot.Sampling.Partitioning.SecurityProofs.Exfil.Witnesses

/-!
# Optimal partitioning

Which division of a computation into proof blocks minimizes what the verifier spends to bound an adversary's gain, over
every sampling design: free compute (`ConverseExact` to `UnequalCost`), exfiltration as `audit_exfiltration` bounds it
(`ExfiltrationOptimum`, `RefinementRule`), and the two-stage detection risk (`RiskConcave`, `Concentration`). The model
is in `Partitioning.Model`, the statements in `Partitioning.Guarantees`, their proofs in `Partitioning.SecurityProofs`,
and instances showing they are not vacuous in `Partitioning.Witness`. They read only the Flock-free part of `Audit`.
-/

@[expose] public section
