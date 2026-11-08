module

public import Definitions.Claims.RecursionGeneric
public import Proofs.Flock.RecGeneric.Plans
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK
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
public import Proofs.Flock.RecGeneric.AtFlockNontrivial
public import Proofs.Flock.RecGeneric.AtFlockBits
public import Proofs.Flock.RecGeneric.AtFlockCounts
public import Proofs.Flock.RecGeneric.AtFlockRecorded
public import Proofs.Flock.RecGeneric.Cost
public import Proofs.Flock.RecGeneric.AtFlockCompiled
public import Proofs.Flock.RecGeneric.AtFlockCompiledSessions
public import Proofs.Flock.RecGeneric.AtFlockCompiledContract
public import Proofs.Flock.RecGeneric.AtFlockCompiledKS
public import Proofs.Flock.RecGeneric.AtFlockCompiledNontrivial
public import Proofs.Flock.RecGeneric.Witnesses
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
public import Proofs.Flock.RecGeneric.ZKSessionsProof
public import Proofs.Flock.RecGeneric.ZKAtFlockProof
public import Proofs.Flock.RecGeneric.ZKAtRecProof
public import Proofs.Flock.RecGeneric.CostProof
public import Proofs.Flock.RecGeneric.CompiledCost
public import Proofs.Flock.RecGeneric.CompiledBits

/-! The recursion as a compiler, generic in Π_in, Π_out and the server, over Π_in with repetition extractors: its
statements (`Definitions.Claims.RecursionGeneric`: `CompositionKS`, L1, and `CompilerKS`, L2), their honest and
nontrivial witnesses (`Proofs.Flock.RecGeneric.Witnesses`), the Flock instance's
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
(`Proofs.Flock.RecGeneric.AtFlockKS`), false at the verifier that accepts everything
(`Proofs.Flock.RecGeneric.AtFlockNontrivial`, the lock's `nontrivial`), `KSLinkOn`'s bits at route 2C's five terms with
the link's at two runs (`Proofs.Flock.RecGeneric.AtFlockBits`), with a rerun's extra calls at the deployment within
route 2C's `n` and a bound on the witnesses' recorded calls (`Proofs.Flock.RecGeneric.AtFlockCounts`), that bound at
most `2^32` wherever a run of the compiled verifier accepts (`Proofs.Flock.RecGeneric.AtFlockRecorded`), and the bridge
to the sampled compiler (`Proofs.Flock.RecGeneric.Sampled`, B1), with B1's proof and the recursive audit from L1
(`Proofs.Flock.RecGeneric.SampledProof`), and a toy outer argument
and compiler at which L2's hypotheses are stated to hold together (`Proofs.Flock.RecGeneric.Toy`), proved to hold
(`Proofs.Flock.RecGeneric.ToyProof`). The plans' expected cost from one run's average
(`Proofs.Flock.RecGeneric.Cost`), proved at Flock's plans (`Proofs.Flock.RecGeneric.CostProof`), and L2 at Flock:
today's compiled record as a compiler over Flock as the outer
argument, with its targets and bound (`Proofs.Flock.RecGeneric.AtFlockCompiled`), the compiler's contract for `E_O`
at the deployment (`Proofs.Flock.RecGeneric.AtFlockCompiledSessions`), the compiler's contract at the deployment's
dense statements (`Proofs.Flock.RecGeneric.AtFlockCompiledContract`, A1d), the bound at every statement from it, with
no hypothesis (`Proofs.Flock.RecGeneric.AtFlockCompiledKS`), with `Compiled.EO`'s expected reruns at most
`1 + Σ_s (1 + m′_s·k_s)` (`Proofs.Flock.RecGeneric.CompiledCost`), the bound false at the verifier that accepts
everything (`Proofs.Flock.RecGeneric.AtFlockCompiledNontrivial`, the lock's `nontrivial`), and its bits at the loop
bounds `(·, 2^256, 2^257)` and the link's `M = 2^256`: at most `κ + t·ε`, `κ` route 2C's and `ε` five per-call terms
(`Proofs.Flock.RecGeneric.CompiledBits`). L1's proof: the
compiled verifier's runs at any server (`Proofs.Flock.RecGeneric.ServerRun`) and the plurality prover on the
composition, which bounds the inner stage's failure where every outer extraction succeeds
(`Proofs.Flock.RecGeneric.Plurality`), its aborting runs as the composite's reruns
(`Proofs.Flock.RecGeneric.Coupling`), and the inner protocol's run finders at those runs read at the reruns
(`Proofs.Flock.RecGeneric.Lift`), the plurality prover's bad samples bounded by the link
(`Proofs.Flock.RecGeneric.BadDev`), and L1 with the recursive audit from it (`Proofs.Flock.RecGeneric.StatementProof`).
`E_O`'s knowledge soundness from the outer argument's (O1), and L2 from L1 and O1
(`Proofs.Flock.RecGeneric.OuterProof`). Zero knowledge (Z1) from the outer phase's and the leaves' hiding
(`Proofs.Flock.RecGeneric.ZKProof`), the hm96 leaves' hiding (Z2, `Proofs.Flock.RecGeneric.ZKHides`), and the
compiled simulator's calls (Z1-cost, `Proofs.Flock.RecGeneric.ZKCost`). An outer phase of zero-knowledge sessions in
turn is zero knowledge per outcome, within the exact sum of the sessions' errors and counted where the session simulator
is (Z1-S and Z1-S-cost: `Proofs.Flock.RecGeneric.ZKSessions`, proved in `Proofs.Flock.RecGeneric.ZKSessionsProof`),
and so is the release of a first part and those sessions (Z1 at `ZkRelease.ofSessions`). A proof system whose simulator
samples its statement's coins is such a session system within its own error (Z1-L, `ZkSess.ofZKOut`), and with Flock's
ZK server as each session the release is zero knowledge per outcome within the sessions' summed server errors (Z1-F:
`Proofs.Flock.RecGeneric.ZKAtFlock`, proved in `Proofs.Flock.RecGeneric.ZKAtFlockProof`). The same server is zero
knowledge per outcome at the recursive outer record, whose statements are the V\* sessions'
(`Proofs.Flock.RecGeneric.ZKServerRec`), so with it as each of `deployRec`'s V\* sessions the release is zero knowledge
per outcome too (Z1-F at `deployRec`: `Proofs.Flock.RecGeneric.ZKAtRec`, proved in
`Proofs.Flock.RecGeneric.ZKAtRecProof`). -/

@[expose] public section
