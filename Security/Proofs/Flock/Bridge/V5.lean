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

/-! `Statement`: `Flock.Guarantees.SampledKSLiveV5`, V2 at the verifier's public inputs with or without shared rows, and
the targets its assembly takes (stated). `EndToEndHidden`: `endToEndHiddenV5_of`, from `OpeningsCanonTargetV5`.
`Assembly`: `assemblyTargetV5`. `OpeningsCanon`: `openingsCanonV5`. `Transports`: `transportsLiveV5`. `KSLive`:
`Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5`, proved. `Honest`:
`Flock.SecurityProofs.SampledKSLiveV5Honest`, its file hypotheses at the served registered session, from the named
premise `ServedFilesParseV5`. `Refused`: `verify --zk`'s form check at a shared-row file gives V5's file hypotheses
`HmIn.parse`, `HmIn.loadPublic` and `RootFixesDomainL` (`v5Files_of_check`). `Accepted`:
`Flock.Bridge.sampledKSLiveV5Accepted`, V5 on the event that `verify --zk` accepts the files and the run passes the
registration's population, without V5's hypotheses on the files' rows, names and roots or on the population order.
`Session`: the openings a witness of a drawn unit's table reveals (`revealV5`, from the message packing it, `packZ`),
which make the unit's proof block correct (`correct_of_revealV5`), and the drawn table at one table (`tabOfZHJP_one`). -/

@[expose] public section
