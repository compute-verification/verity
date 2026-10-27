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
import Proofs.Flock.Bridge.Public
import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.Bridge.PublicToyDesc
import Proofs.Flock.Bridge.HiddenReturns
import Proofs.Flock.Bridge.HiddenLayer
import Proofs.Flock.Bridge.HiddenModel
import Proofs.Flock.Bridge.Assumptions.ToySetup
import Proofs.Flock.Bridge.HiddenModelWitness

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded. `KSLive`: `Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`,
the sampled statement at `RowsOk` circuit files (row/v1, row/v2 and row-seg/v1 ports) and the constants `κ` and
`canonAt`, proved (`Flock.Bridge.assemblyTargetC` with every target it takes), and
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, V2 with its file preconditions read off the
verifiers' acceptance (`Refused`). `Lowers`: the claims program `progOf C κ` lowers every unit of every circuit
(`Flock.Bridge.lowers`).
`FaultyRightC`: a unit right on its shaped openings is not faulty (`Flock.Bridge.faultyRightC`). `Public`:
`Flock.SecurityProofs.SampledKSLivePub : Flock.Guarantees.SampledKSLivePub`, `SampledKSLiveV2` in the additive form
`δ + f(J²·K·Pr[B^τ])`: `Flock.Bridge.deltaPub`, a function of the public parameters, plus `Flock.Bridge.lossPub` at
`J²·K` times the collision probability of `B^τ`, the uniform mixture of C-Flock's `K` finders (`PubFinder`), `J` the
plan's tables (the averaged premise, `PubAvg`). `PublicDeployed`:
`Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed`, `δ + f(p) < 1` at the deployment's
parameters for `p ≤ 2^-200`. `PublicWitnesses`: its honest lemma (the parameter hypotheses hold as deployed) and its nontrivial lemma
(false at the verifier that accepts everything). `PublicToyDesc`: a hand-built circuit file meeting the nontrivial
lemma's structural premises (no port, a computed returned gate, `ioSum ≤ 2^40`). `HiddenReturns`: under the
parse those premises are never met, since a returned gate is an output row's bit (`Flock.Bridge.outputs_noPort`), so
no file parses to the toy (`Flock.Bridge.toyDesc_unparsed`). `HiddenLayer`: the hidden-output count curve
(`zk_flock_countPr_reg` at `planZHJ`) is nontrivial at a perfectly binding toy binding (`Flock.Bridge.countPrReg_nontrivial`). `HiddenModel`: `EndToEndHiddenModel` at
the verifier that accepts everything, and its toy, whose every table is refused. `HiddenModelWitness`: under
`toy-setup` (`Assumptions.ToySetup`), that statement is false (`Flock.Bridge.endToEndHiddenModel_nontrivial`). -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
