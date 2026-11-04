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
import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Bridge.Honest
import Proofs.Flock.Bridge.V5
import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.BitsRows
import Proofs.Flock.Bridge.BitsLower
import Proofs.Flock.Bridge.BitsKeys
import Proofs.Flock.Bridge.BitsKeyProg

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
(`Bridge.Assumptions.ServedFilesParse`). `BitsRows`: `SampledProofs.Bits`' stacked unit rows are a circuit's template
instance rows (`Flock.Bridge.rowsAgree_circuitRows`). `BitsLower`: where a Bits key program lays out a circuit and its
units, the bridge's lowering, positions and proof blocks are Bits' (`Flock.Bridge.LaysOut.progOf_eq`,
`Flock.Bridge.blockOf_eq`). `BitsKeys`: Bits' source keys are the bridge's at every input whose Bits source isn't free
(`Flock.Bridge.keyIdx_eq`). `BitsKeyProg`: where every input a computed row reads has a row key, Bits' key program
lays out a circuit file's and its units, so the bridge's program and proof blocks are Bits'
(`Flock.Bridge.progOf_keyProg`, `Flock.Bridge.blockOf_keyProg`). -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
