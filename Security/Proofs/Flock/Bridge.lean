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
import Proofs.Flock.Bridge.RowsReg
import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Bridge.Honest
import Proofs.Flock.Bridge.V5
import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.Bridge.PublicNontrivial
import Proofs.Flock.Bridge.RowsRegWitnesses
import Proofs.Flock.Bridge.PublicToyDesc
import Proofs.Flock.Bridge.HiddenReturns
import Proofs.Flock.Bridge.HiddenLayer
import Proofs.Flock.Bridge.HiddenModel
import Proofs.Flock.Bridge.Assumptions.ToySetup
import Proofs.Flock.Bridge.HiddenModelWitness

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded. `KSLive`: `Flock.Bridge.sampledKSLiveXr : Flock.Bridge.SampledKSLiveXr`, the sampled
statement at `RowsOk` circuit files (row/v1, row/v2 and row-seg/v1 ports) and the constants `κ` and `canonAt`, with
the claims at the extracted values, proved (`Flock.Bridge.assemblyTargetC` with every target it takes);
and `Flock.Bridge.sampledKSLiveV2Z`, the claims at the registrant's rows under `verify --zk`, from the count curve at
the rows link (`Rows`, `RowsC`). `RowsReg`: `Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`,
the claims at the registrant's rows under `verify --zk --registered F`, from those two, and
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, V2 with its file preconditions read off the
verifiers' acceptance (`Refused`). `Lowers`: the claims program `progOf C κ` lowers every unit of every circuit
(`Flock.Bridge.lowers`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`), at any claims values
that are the evaluation on its wires (`Flock.Bridge.faultyRows`). `Public`:
`Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, the claims at the registrant's rows in the
additive form `δ + f(J²·K·Pr[B^τ])`, from `Bridge.SampledKSLiveXr` and V2: `Flock.Bridge.deltaPubR`, a function of the
public parameters, plus `Flock.Bridge.lossPubR` at `J²·K` times the collision probability of `B^τ`, the uniform mixture
of C-Flock's `K` finders and the registered reads' (`PubFinderRows`, over `PubFinder`'s), `J` the plan's tables (the
averaged premise, `PubAvg`). `PublicDeployed`:
`Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed`, `δ + f(p) < 1` at the deployment's
parameters for `p ≤ 2^-200`. `PublicWitnesses`: its honest lemma (the parameter hypotheses hold as deployed) and the
statement at the verifier that accepts everything (`Flock.Guarantees.SampledKSLivePubAll`). `PublicNontrivial`: that
statement is false at the toy session of `Flock.Test.ToyPubFiles` (`Flock.Guarantees.sampledKSLivePub_nontrivial`),
where a registrant opens the returned row's input to `0` and the gate to `1` (`ToyPubReads`, `ToyPubGates`,
`FaultyRow`), under two interim premises: the toy's files parse (`Assumptions.ToyFilesParse`) and its setup
(`Assumptions.ToySetupPub`). `PublicToyDesc`: a hand-built circuit file meeting V2's nontrivial lemma's structural
premises (no port, a computed returned gate, `ioSum ≤ 2^40`). `HiddenReturns`: under the parse those premises are never
met, since a returned gate is an output row's bit (`Flock.Bridge.outputs_noPort`), so no file parses to the toy
(`Flock.Bridge.toyDesc_unparsed`). `HiddenLayer`: the hidden-output count curve
(`zk_flock_countPr_reg` at `planZHJ`) is nontrivial at a perfectly binding toy binding (`Flock.Bridge.countPrReg_nontrivial`). `HiddenModel`: `EndToEndHiddenModel` at
the verifier that accepts everything, and its toy, whose every table is refused. `HiddenModelWitness`: under
`toy-setup` (`Assumptions.ToySetup`), that statement is false (`Flock.Bridge.endToEndHiddenModel_nontrivial`).
`RowsC`: the count curve with "wrong" at canonical openings or off the registrant's rows, at the finders' collision
probabilities (`ZkLink.zk_flock_countPr_rowsC`). `Honest`:
`Flock.SecurityProofs.SampledKSLiveHonest : Flock.Guarantees.SampledKSLiveHonest`, the file hypotheses of V2, V3 and
Pub at the deployed session, from the named premise that its files parse (`Bridge.Assumptions.ServedFilesParse`). -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
