module

public import Definitions.Claims.RecursionGeneric
public import Proofs.Flock.RecGeneric.Plans
public import Proofs.Flock.RecGeneric.Outer
public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.C3.Compose.HoloStreamSizes
public import Proofs.Flock.C3.Reps.LigNames
public import Proofs.Flock.C3.VStarList.Accepts
public import Proofs.Flock.Claims.KS.PlainSampleJMove
public import Proofs.Flock.Claims.Recursion.BoundWeightDeploy
public import Proofs.Flock.Claims.Recursion.DeployHonest
public import Proofs.Flock.Claims.Recursion.KSLinks
public import Proofs.Flock.Claims.Recursion.LastColsE
public import Proofs.Flock.Claims.Recursion.OpenedDecode
public import Proofs.Flock.Claims.Recursion.PartsVStar
public import Proofs.Flock.Claims.Recursion.RowSlots
public import Proofs.Flock.Claims.Recursion.VStarPartsB
public import Proofs.Flock.Claims.Recursion.VarLen
public import Proofs.Flock.Recursive.ValueInner
public import Proofs.Flock.SessionLoop.FinPoints
public import Proofs.Flock.VBridge.AlgResZero
public import Proofs.Flock.VBridge.Family.Setup
public import Proofs.Flock.Claims.Assembled
public import Proofs.Flock.Claims.KS.PlainLeafClashJK
public import Proofs.Core.QSort
public import Proofs.Flock.C3.RegLevels.OutRegs
public import Proofs.Flock.C3.RegLevels.VRows
public import Proofs.Flock.Claims.Recursion.DecodeSame
public import Proofs.Flock.Claims.Recursion.InnerSetupSame
public import Proofs.Flock.Claims.Recursion.LabelsFull
public import Proofs.Flock.Recursive.HPortTie
public import Proofs.Flock.VBridge.Family.PartKeys
public import Proofs.Flock.Claims.Recursion.Assumptions.KSFixture
public import Proofs.Flock.Claims.Recursion.PlainKSAbortFails
public import Proofs.Flock.Claims.Witnesses.RecursionLin2
public import Proofs.Flock.SamplingGeneric
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
public import Proofs.Flock.RecGeneric.OpenedDesign
public import Proofs.Flock.RecGeneric.OuterProof
public import Proofs.Flock.RecGeneric.ZKProof
public import Proofs.Flock.RecGeneric.ZKHides
public import Proofs.Flock.RecGeneric.ZKCost
public import Proofs.Flock.RecGeneric.ZKSessionsProof
public import Proofs.Flock.RecGeneric.ZKAtFlockProof
public import Proofs.Flock.RecGeneric.ZKAtRecProof

/-! The recursion as a compiler, generic in Π_in, Π_out and the server, over Π_in with repetition extractors: its
statements (`Definitions.Flock.Recursive.Generic`: `CompositionKS`, L1, and `CompilerKS`, L2), the bridge to the
sampled compiler (`Proofs.Flock.RecGeneric.Sampled`, B1), with B1's proof and the recursive audit from L1
(`Proofs.Flock.RecGeneric.SampledProof`), and a toy outer argument and compiler at which L2's hypotheses are stated to
hold together (`Proofs.Flock.RecGeneric.Toy`), proved to hold (`Proofs.Flock.RecGeneric.ToyProof`). L1's proof: the
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
