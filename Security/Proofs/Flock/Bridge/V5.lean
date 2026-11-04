import Proofs.Flock.Bridge.V5.Statement
import Proofs.Flock.Bridge.V5.EndToEndHidden
import Proofs.Flock.Bridge.V5.Assembly
import Proofs.Flock.Bridge.V5.OpeningsCanon
import Proofs.Flock.Bridge.V5.Transports
import Proofs.Flock.Bridge.V5.KSLive
import Proofs.Flock.Bridge.V5.Honest
import Proofs.Flock.Bridge.V5.Refused
import Proofs.Flock.Bridge.V5.Accepted

/-! `Statement`: `Flock.Guarantees.SampledKSLiveV5`, V2 at the verifier's public inputs with or without shared rows, and
the targets its assembly takes (stated). `EndToEndHidden`: `endToEndHiddenV5_of`, from `OpeningsCanonTargetV5`.
`Assembly`: `assemblyTargetV5`. `OpeningsCanon`: `openingsCanonV5`. `Transports`: `transportsLiveV5`. `KSLive`:
`Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5`, proved. `Honest`:
`Flock.SecurityProofs.SampledKSLiveV5Honest`, its file hypotheses at the served registered session, from the named
premise `ServedFilesParseV5`. `Refused`: `verify --zk`'s form check at a shared-row file gives V5's file hypotheses
`HmIn.parse`, `HmIn.loadPublic` and `RootFixesDomainL` (`v5Files_of_check`). `Accepted`:
`Flock.Bridge.sampledKSLiveV5Accepted`, V5 on the event that `verify --zk` accepts the files and the run passes the
registration's population, without V5's hypotheses on the files' rows, names and roots or on the population order. -/
