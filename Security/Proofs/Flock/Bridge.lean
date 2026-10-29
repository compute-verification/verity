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
import Proofs.Flock.Bridge.RelSGuards
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.V5

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded. `KSLive`: `Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`,
the sampled statement at `RowsOk` circuit files (row/v1, row/v2 and row-seg/v1 ports) and the constants `κ` and
`canonAt`, proved (`Flock.Bridge.assemblyTargetC` with every target it takes), and
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, V2 with its file preconditions read off the
verifiers' acceptance (`Refused`). `Lowers`: the claims program `progOf C κ` lowers every unit of every circuit
(`Flock.Bridge.lowers`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`). `Public`:
`Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, `SampledKSLiveV2` with its error
`Flock.Bridge.deltaPub`, a function of the public parameters. -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
