module

public import Proofs.Flock.Bridge.PortRoots
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.Targets
public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Bridge.Count
public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Bridge.Canon
public import Proofs.Flock.Bridge.OpeningsCanon
public import Proofs.Flock.Bridge.Assembly
public import Proofs.Flock.Bridge.EndToEndHiddenC
public import Proofs.Flock.Bridge.FaultyRightC
public import Proofs.Flock.Bridge.RelSGuards
public import Proofs.Flock.Bridge.Refused
public import Proofs.Flock.Bridge.KSLive
public import Proofs.Flock.Bridge.PublicDeployed
public import Proofs.Flock.Bridge.Honest
public import Proofs.Flock.Bridge.V5
public import Proofs.Flock.Bridge.Public

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded. `KSLive`: `Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`,
the sampled statement at `RowsOk` circuit files (row/v1, row/v2 and row-seg/v1 ports) and the constants `κ` and
`canonAt`, proved (`Flock.Bridge.assemblyTargetC` with every target it takes), and
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, V2 with its file preconditions read off the
verifiers' acceptance (`Refused`). `Lowers`: the claims program `progOf C κ` lowers every unit of every circuit
(`Flock.Bridge.lowers`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`). `Public`:
`Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, `SampledKSLiveV2` with its error
`Flock.Bridge.deltaPub`, a function of the public parameters. `PublicDeployed`:
`Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed`, `deltaPub < 1` at the deployment's
parameters. `Honest`: `Flock.SecurityProofs.SampledKSLiveHonest : Flock.Guarantees.SampledKSLiveHonest`, the file
hypotheses of V2, V3 and Pub at the deployed session, from the named premise that its files parse
(`Bridge.Assumptions.ServedFilesParse`). -/

@[expose] public section

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
