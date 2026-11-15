module

public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.V5.EndToEndHidden
public import Proofs.Flock.Bridge.V5.Assembly
public import Proofs.Flock.Bridge.V5.OpeningsCanon
public import Proofs.Flock.Bridge.V5.Transports
public import Proofs.Flock.Bridge.V5.KSLive
public import Proofs.Flock.Bridge.V5.Honest
public import Proofs.Flock.Bridge.V5.Refused
public import Proofs.Flock.Bridge.V5.Accepted
public import Proofs.Flock.Bridge.V5.Session
public import Proofs.Flock.Bridge.V5.QsRows
public import Proofs.Flock.Bridge.V5.Rows

/-! `Statement`: `Flock.Guarantees.SampledKSLiveV5`, V2 at the verifier's public inputs with or without shared rows, and
the targets its assembly takes (stated). `EndToEndHidden`: `endToEndHiddenV5_of`, from `OpeningsCanonTargetV5`.
`Assembly`: `assemblyTargetV5`. `OpeningsCanon`: `openingsCanonV5`. `Transports`: `transportsLiveV5`. `KSLive`:
`Flock.Bridge.sampledKSLiveV5R`, V5's bound under the roots guard, and `Flock.SecurityProofs.SampledKSLiveV5 :
Flock.Guarantees.SampledKSLiveV5` from it, the guard paid for by `ecr/sha-512` at the roots finder (`LeafRoots`). `Honest`:
`Flock.SecurityProofs.SampledKSLiveV5Honest`, its file hypotheses at the served registered session, from the named
premise `ServedFilesParseV5`. `Refused`: `verify --zk`'s published-roots check gives `SampledKSLiveV5R`'s
`RootFixesDomainL`, with or without shared rows (`fixesL_of_collidesPub_false`). `Accepted`:
`Flock.Bridge.sampledKSLiveV5Accepted`, `SampledKSLiveV5R` on the event that
`verify --zk` accepts the files, flat or templated, and the run
passes the registration's population, without its hypotheses on the files' rows, domains and roots or on the population
order. `Session`: the openings a witness of a sampled proof block's table reveals (`revealV5`, from the message packing
it, `packZ`), which make the unit's proof block correct (`correct_of_revealV5`), and the sampled table at one table
(`tabOfZHJP_one`). `QsRows`: `Q_s` at public inputs by the rows and public gates a sample reads, at any sampling
design on strata (`qs_vbP_execSel_le`). `Rows`: `SampledKSLiveV5`'s bound with that count in place of the average
number of commit strings a sample reads (`sampledKSLiveV5_rows`). -/

@[expose] public section
