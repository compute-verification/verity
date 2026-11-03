import Proofs.Flock.Bridge.PortRoots
import Proofs.Flock.Bridge.PopMatches
import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.Lowers
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.CircuitOf
import Proofs.Flock.Bridge.Canon
import Proofs.Flock.Bridge.OpeningsCanon
import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Bridge.Honest
import Proofs.Flock.Bridge.V5
import Proofs.Flock.Bridge.Public

/-! The sampled audit's knowledge soundness, live. Its headline is `V5`:
`Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5`, at flock-verify `--zk --public-inputs` with or
without shared rows, beside its honest witness `Flock.SecurityProofs.SampledKSLiveV5Honest`. `KSLive`:
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, the no-shared-rows form, which V5 doesn't
imply: the sampled statement on the event that the verifiers accept the files (`Refused`), from the lemma
`Flock.Bridge.sampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`, the statement at `RowsOk` circuit files (row/v1, row/v2
and row-seg/v1 ports) and the constants `κ` and `canonAt` (`Flock.Bridge.assemblyTargetC` with every target it takes).
`Lowers`: the claims program `progOf C κ` lowers every unit of every circuit (`Flock.Bridge.lowers`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`). `Public`: the lemma
`Flock.Bridge.sampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, `SampledKSLiveV2` with its error
`Flock.Bridge.deltaPub`, a function of the public parameters. `PublicDeployed`:
`Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed`, `deltaPub < 1` at the deployment's
parameters. `Honest`: `Flock.SecurityProofs.SampledKSLiveHonest : Flock.Guarantees.SampledKSLiveHonest`, V3's honest
witness: the file hypotheses of V2, V3 and Pub at the deployed session, from the named premise that its files parse
(`Bridge.Assumptions.ServedFilesParse`). -/
