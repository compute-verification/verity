module

public import Proofs.Flock.RecGeneric.Statement
public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.AtFlockProof
public import Proofs.Flock.RecGeneric.Sampled
public import Proofs.Flock.RecGeneric.SampledProof
public import Proofs.Flock.RecGeneric.Toy

/-! The recursion as a compiler, generic in the outer argument and the server, over inner protocols with repetition
extractors: its statements (`Proofs.Flock.RecGeneric.Statement`), the Flock instance's
(`Proofs.Flock.RecGeneric.AtFlock`) with its last-coin hypothesis proved at the deployment
(`Proofs.Flock.RecGeneric.AtFlockProof`), and the bridge to the sampled compiler (`Proofs.Flock.RecGeneric.Sampled`,
B1), with B1's proof and the recursive audit from L1 (`Proofs.Flock.RecGeneric.SampledProof`), and a toy outer argument
and compiler at which L2's hypotheses are stated to hold together (`Proofs.Flock.RecGeneric.Toy`). -/

@[expose] public section
