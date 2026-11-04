module

public import Proofs.Flock.RecGeneric.Statement
public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.AtFlockProof
public import Proofs.Flock.RecGeneric.Sampled
public import Proofs.Flock.RecGeneric.SampledProof
public import Proofs.Flock.RecGeneric.Toy
public import Proofs.Flock.RecGeneric.ToyProof
public import Proofs.Flock.RecGeneric.ServerRun
public import Proofs.Flock.RecGeneric.Plurality
public import Proofs.Flock.RecGeneric.Coupling
public import Proofs.Flock.RecGeneric.Lift
public import Proofs.Flock.RecGeneric.BadDev
public import Proofs.Flock.RecGeneric.StatementProof
public import Proofs.Flock.RecGeneric.OuterProof
public import Proofs.Flock.RecGeneric.ZKProof
public import Proofs.Flock.RecGeneric.ZKHides
public import Proofs.Flock.RecGeneric.ZKCost

/-! The recursion as a compiler, generic in the outer argument and the server, over inner protocols with repetition
extractors: its statements (`Proofs.Flock.RecGeneric.Statement`), the Flock instance's
(`Proofs.Flock.RecGeneric.AtFlock`) with its last-coin hypothesis proved at the deployment
(`Proofs.Flock.RecGeneric.AtFlockProof`), and the bridge to the sampled compiler (`Proofs.Flock.RecGeneric.Sampled`,
B1), with B1's proof and the recursive audit from L1 (`Proofs.Flock.RecGeneric.SampledProof`), and a toy outer argument
and compiler at which L2's hypotheses are stated to hold together (`Proofs.Flock.RecGeneric.Toy`), proved to hold
(`Proofs.Flock.RecGeneric.ToyProof`). L1's proof: the
compiled verifier's runs at any server (`Proofs.Flock.RecGeneric.ServerRun`) and the plurality prover on the
composition, which bounds the inner stage's failure where every outer extraction succeeds
(`Proofs.Flock.RecGeneric.Plurality`), its aborting runs as the composite's reruns
(`Proofs.Flock.RecGeneric.Coupling`), and the inner protocol's run finders at those runs read at the reruns
(`Proofs.Flock.RecGeneric.Lift`), the plurality prover's bad draws bounded by the link
(`Proofs.Flock.RecGeneric.BadDev`), and L1 with the recursive audit from it (`Proofs.Flock.RecGeneric.StatementProof`).
`E_O`'s knowledge soundness from the outer argument's (O1), and L2 from L1 and O1
(`Proofs.Flock.RecGeneric.OuterProof`). Zero knowledge (Z1) from the outer phase's and the leaves' hiding
(`Proofs.Flock.RecGeneric.ZKProof`), the hm96 leaves' hiding (Z2, `Proofs.Flock.RecGeneric.ZKHides`), and the
compiled simulator's calls (Z1-cost, `Proofs.Flock.RecGeneric.ZKCost`). -/

@[expose] public section
