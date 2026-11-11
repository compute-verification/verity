module

public import Proofs.Flock.Bridge.PortRoots
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.Targets
public import Proofs.Flock.Bridge.FilePartTarget
public import Proofs.Flock.Bridge.FilePart
public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Bridge.Slots
public import Proofs.Flock.Bridge.SlotsKey
public import Proofs.Flock.Bridge.Scope
public import Proofs.Flock.Bridge.Count
public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Bridge.UnitRows
public import Proofs.Flock.Bridge.Canon
public import Proofs.Flock.Bridge.OpeningsCanon
public import Proofs.Flock.Bridge.Assembly
public import Proofs.Flock.Bridge.EndToEndHiddenC
public import Proofs.Flock.Bridge.FaultyRightC
public import Proofs.Flock.Bridge.Refused
public import Proofs.Flock.Bridge.ScopeGate
public import Proofs.Flock.Bridge.KSLive
public import Proofs.Flock.Bridge.PublicDeployed
public import Proofs.Flock.Bridge.Honest
public import Proofs.Flock.Bridge.BitsRows
public import Proofs.Flock.Bridge.BitsLower
public import Proofs.Flock.Bridge.BitsKeys
public import Proofs.Flock.Bridge.BitsKeyProg
public import Proofs.Flock.Bridge.Members
public import Proofs.Flock.Bridge.BitsRowMap
public import Proofs.Flock.Bridge.V5
public import Proofs.Flock.Bridge.LeafSeg
public import Proofs.Flock.Bridge.Public
public import Proofs.Flock.Bridge.QsRows
public import Proofs.Flock.Bridge.PublicRows
public import Proofs.Flock.Bridge.QsRows.FirstReads

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded. `KSLive`: `Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`,
the sampled statement at `RowsOk` circuit files (row/v1, row/v2 and row-seg/v1 ports) and the constants `κ` and
`canonAt`, proved (`Flock.Bridge.assemblyTargetC` with every target it takes), and
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, V2 with its statement, file and draw
preconditions read off the verifiers' acceptance (`Refused`, `ScopeGate`). `Lowers`: the claims program `progOf C κ`
lowers every unit of every circuit
(`Flock.Bridge.lowers`). `FilePart`: `FilePartTarget`, proved (`Flock.Bridge.filePartTarget`), and `RelDeployTarget`
at the code's partition (`Flock.Bridge.relDeploy_ofCircuit`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`). `Public`:
`Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, `SampledKSLiveV2` with its error
`Flock.Bridge.deltaPub`, a function of the public parameters. `PublicDeployed`:
`Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed`, `deltaPub < 1` at the deployment's
parameters. `Honest`: `Flock.SecurityProofs.SampledKSLiveHonest : Flock.Guarantees.SampledKSLiveHonest`, the file
hypotheses of V2, V3 and Pub at the deployed session, from the named premise that its files parse
(`Bridge.Assumptions.ServedFilesParse`). `LeafSeg`: the `hm96-sha512/row-seg/v1` leaf binds, and
`Flock.Bridge.commitRowL`, the deployed commitment with each port's own leaf, binds at every circuit file and opens at
SHA-512 what `commitDepL` opens. `BitsRows`: `Spot.Bits`' stacked unit rows are a circuit's template
instance rows (`Flock.Bridge.rowsAgree_circuitRows`). `BitsLower`: where a Bits key program lays out a circuit and its
units, the bridge's lowering, positions and proof blocks are Bits' (`Flock.Bridge.LaysOut.progOf_eq`,
`Flock.Bridge.blockOf_eq`). `BitsKeys`: Bits' source keys are the bridge's (`Flock.Bridge.keyIdx_eq`). `BitsKeyProg`:
Bits' key program lays out a circuit file's and its units, so the bridge's program, positions and proof blocks are
Bits' (`Flock.Bridge.progOf_keyProg`, `Flock.Bridge.posOf_keyProg`, `Flock.Bridge.blockOf_bitsKeyProg`). `BitsRowMap`:
Bits' row map is the bridge's: at an input gate (`Flock.Bridge.rowBit_keyRowOf`, `Flock.Bridge.rowBit_wire`) and at a
returned output's bit (`Flock.Bridge.outRowBit_outBitRow`, `Flock.Bridge.outRowBit_outRowOf`). `UnitRows`: each copy
of a unit of the parsed file holds the unit net's rows at the unit's local evaluation (`Flock.Bridge.rows_localEval`).
-/

@[expose] public section

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
