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
import Proofs.Flock.Bridge.ScopeGate
import Proofs.Flock.Bridge.ScopeGateSanity
import Proofs.Flock.Bridge.RefusedWitnesses
import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.Bridge.PublicToyDesc
import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.PublicDeployed

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded. `KSLive`: `Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`,
the sampled statement at `RowsOk` circuit files (row/v1, row/v2 and row-seg/v1 ports) and the constants `κ` and
`canonAt`, proved (`Flock.Bridge.assemblyTargetC` with every target it takes), and
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, V2 with its file preconditions read off the
verifiers' acceptance (`Refused`), and
`Flock.SecurityProofs.SampledKSLiveV4 : Flock.Guarantees.SampledKSLiveV4`, V3 with its six remaining file and draw
preconditions refused by `verify --zk` (`ScopeGate`). `ScopeGateSanity`: `SampledKSLiveV4`'s nontrivial lemma, at
`SampledKSLivePub`'s failing instance through `SampledKSLiveV3`'s. `RefusedWitnesses`: `SampledKSLiveV2`'s and
`SampledKSLiveV3`'s nontrivial lemmas, at that instance through `SampledKSLiveV2`'s error at most `deltaPub`.
`PublicWitnesses`: `SampledKSLivePub`'s honest lemma (the parameter hypotheses hold as deployed) and its nontrivial
lemma (false at the verifier that accepts everything). `PublicToyDesc`: a hand-built circuit file meeting the
nontrivial lemma's structural premises (no port, a computed returned gate, `ioSum ≤ 2^40`).
`Lowers`: the claims program `progOf C κ` lowers every unit of every circuit
(`Flock.Bridge.lowers`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`). `Public`:
`Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, `SampledKSLiveV2` with its error
`Flock.Bridge.deltaPub`, a function of the public parameters. `PublicDeployed`:
`Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed`, `deltaPub < 1` at the deployment's
parameters. -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
