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

/-! `Statement`: `Flock.Guarantees.SampledKSLiveV5`, V2 at the verifier's public inputs with or without shared rows, and
the targets its assembly takes (stated). `EndToEndHidden`: `endToEndHiddenV5_of`, from `OpeningsCanonTargetV5`.
`Assembly`: `assemblyTargetV5`. `OpeningsCanon`: `openingsCanonV5`. `Transports`: `transportsLiveV5`. `KSLive`:
`Flock.Bridge.sampledKSLiveV5R`, V5's bound under the roots guard, and `Flock.SecurityProofs.SampledKSLiveV5 :
Flock.Guarantees.SampledKSLiveV5` from it, the guard paid for by `ecr/sha-512` at the roots finder (`LeafRoots`). `Honest`:
`Flock.SecurityProofs.SampledKSLiveV5Honest`, its file hypotheses at the served registered session, from the named
premise `ServedFilesParseV5`. `Refused`: `verify --zk`'s form check at a shared-row file gives `SampledKSLiveV5R`'s file hypotheses
`HmIn.parse`, `HmIn.loadPublic` and `RootFixesDomainL` at the circuit it checks (`v5Files_of_check`). `Accepted`:
`Flock.Bridge.sampledKSLiveV5Accepted`, `SampledKSLiveV5R` on the event that `verify --zk` accepts the files and the run
passes the registration's population, flat or templated, without its hypotheses on the files' rows, names and roots or on
the population order. -/

@[expose] public section
