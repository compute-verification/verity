module

public import Proofs.Flock.Soundness.Audit.Partitioning.Model
public import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees
public import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.FreeCompute
public import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfiltration
public import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.TwoStage
public import Proofs.Flock.Soundness.Audit.Partitioning.Witness

/-! Optimal partitioning: which division of a computation into proof blocks minimizes what the verifier spends to bound an adversary's gain, over every sampling design. -/

@[expose] public section
