module

public import Proofs.Flock.RecGeneric.Statement
public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.Sampled
public import Proofs.Flock.RecGeneric.SampledProof

/-! The recursion as a compiler, generic in the inner protocol, the outer argument and the server: its statements
(`Proofs.Flock.RecGeneric.Statement`), the Flock instance's (`Proofs.Flock.RecGeneric.AtFlock`), and the bridge to the
sampled compiler (`Proofs.Flock.RecGeneric.Sampled`, B1), with B1's proof and the recursive audit from L1
(`Proofs.Flock.RecGeneric.SampledProof`). -/

@[expose] public section
