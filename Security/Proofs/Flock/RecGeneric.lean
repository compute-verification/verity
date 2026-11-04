module

public import Proofs.Flock.RecGeneric.Statement
public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.PlansProof
public import Proofs.Flock.RecGeneric.LinkBits
public import Proofs.Flock.RecGeneric.AtFlockPlans
public import Proofs.Flock.RecGeneric.AtFlockServer
public import Proofs.Flock.RecGeneric.AtFlockProof
public import Proofs.Flock.RecGeneric.AtFlockEvery
public import Proofs.Flock.RecGeneric.AtFlockRecovers
public import Proofs.Flock.RecGeneric.AtFlockReruns
public import Proofs.Flock.RecGeneric.AtFlockTerms
public import Proofs.Flock.RecGeneric.AtFlockContract
public import Proofs.Flock.RecGeneric.AtFlockKS
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

/-! The recursion as a compiler, generic in the outer argument and the server, over inner protocols with repetition
extractors: its statements (`Proofs.Flock.RecGeneric.Statement`), the Flock instance's
(`Proofs.Flock.RecGeneric.AtFlock`), with its plans' reads by Wald's bound (`Proofs.Flock.RecGeneric.PlansProof`), the
link plan's expected cost and a finder's collision bound from it (`Proofs.Flock.RecGeneric.LinkBits`), its
plans at its own runs as its finders (`Proofs.Flock.RecGeneric.AtFlockPlans`), its server as the bound record's
(`Proofs.Flock.RecGeneric.AtFlockServer`), and its well-formedness, its inner KS, its outer extraction's KS from each session's and its last-coin
hypothesis at the deployment proved (`Proofs.Flock.RecGeneric.AtFlockProof`), `KSLinkOn` at the dense statements lifted
to every statement, since the deployed verifier accepts only dense ones (`Proofs.Flock.RecGeneric.AtFlockEvery`),
`RecoversLinkOn` from the three term comparisons (`Proofs.Flock.RecGeneric.AtFlockRecovers`, F9a), the composite's
reruns read as `Claims.Recursion`'s samples (`Proofs.Flock.RecGeneric.AtFlockReruns`), the term comparisons
(`Proofs.Flock.RecGeneric.AtFlockTerms`, F6), the contract at the dense statements
(`Proofs.Flock.RecGeneric.AtFlockContract`, F7), and the recursive verifier's KS at every statement from them
(`Proofs.Flock.RecGeneric.AtFlockKS`), and the bridge to the sampled compiler (`Proofs.Flock.RecGeneric.Sampled`,
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
(`Proofs.Flock.RecGeneric.OuterProof`). -/

@[expose] public section
