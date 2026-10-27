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
import Proofs.Flock.Bridge.RowsC
import Proofs.Flock.Bridge.Rows
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Bridge.Honest
import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.Bridge.PublicToyDesc

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded. `KSLive`: `Flock.Bridge.sampledKSLiveXr : Flock.Bridge.SampledKSLiveXr`, the sampled
statement at `RowsOk` circuit files (row/v1, row/v2 and row-seg/v1 ports) and the constants `κ` and `canonAt`, with
the claims at the extracted values, proved (`Flock.Bridge.assemblyTargetC` with every target it takes);
`Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`, the claims at the registrant's rows, from it
and the rows link (`Rows`, `RowsC`); and `Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, V2
with its file preconditions read off the verifiers' acceptance (`Refused`). `Lowers`: the claims program `progOf C κ`
lowers every unit of every circuit (`Flock.Bridge.lowers`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`), at any claims values
that are the evaluation on its wires (`Flock.Bridge.faultyRows`). `Public`:
`Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, `SampledKSLiveV2` with its error
`Flock.Bridge.deltaPubR`, a function of the public parameters. `PublicDeployed`:
`Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed`, `deltaPubR < 1` at the deployment's
parameters. `PublicWitnesses`: its honest lemma (the parameter hypotheses hold as deployed) and its nontrivial lemma
(false at the verifier that accepts everything). `PublicToyDesc`: a hand-built circuit file meeting the nontrivial
lemma's structural premises (no port, a computed returned gate, `ioSum ≤ 2^40`). `Honest`:
`Flock.SecurityProofs.SampledKSLiveHonest : Flock.Guarantees.SampledKSLiveHonest`, the file hypotheses of V2, V3 and
Pub at the deployed session, from the named premise that its files parse (`Bridge.Assumptions.ServedFilesParse`). -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
