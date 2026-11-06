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
`Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5`, proved. `Honest`:
`Flock.SecurityProofs.SampledKSLiveV5Honest`, its file hypotheses at the served registered session, from the named
premise `ServedFilesParseV5`. `Refused`: `verify --zk`'s published-roots check gives V5's `RootFixesDomainL`, with or
without shared rows (`fixesL_of_collidesPub_false`). `Accepted`:
`Flock.Bridge.sampledKSLiveV5Accepted`, V5 on the event that `verify --zk` accepts the files and the run passes the
registration's population, without V5's hypotheses on the files' rows, domains and roots or on the population order. -/

@[expose] public section
