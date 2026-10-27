import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.OpeningsCanon
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.ScopeGate
import Proofs.Flock.Bridge.Transports

/-!
# `SampledKSLiveV2`, `SampledKSLiveV3` and `SampledKSLiveV4`, proved

`Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`: `AssemblyTargetC` (`assemblyTargetC`) with
canonical transport (`transportsLiveC`), M1's core (`faultyRightTargetC`), the slot count (`partRus`), the count lemma
(`count`) and `EndToEndHiddenC` (`endToEndHiddenC_of`, from `openingsCanon`) plugged in.
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`: V2 with its file preconditions read off the
verifiers' acceptance (`sampledKSLiveV3_of`).
`Flock.SecurityProofs.SampledKSLiveV4 : Flock.Guarantees.SampledKSLiveV4`: V3 with its six remaining file and draw
preconditions refused by `verify --zk` (`sampledKSLiveV4_of`).
-/

/-- **The sampled audit's knowledge soundness, live, at the circuit files the verifier of record reads** (`Flock.Guarantees.SampledKSLiveV2`). -/
theorem Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2 :=
  Flock.Bridge.assemblyTargetC Flock.Bridge.transportsLiveC Flock.Bridge.faultyRightTargetC Flock.Bridge.partRus
    Flock.Bridge.count (Flock.Bridge.endToEndHiddenC_of Flock.Bridge.openingsCanon)

/-- **The sampled audit's knowledge soundness, live, on the event that the verifiers accept the files**
(`Flock.Guarantees.SampledKSLiveV3`): no hypothesis on the files' rows, names, roots or population order. -/
theorem Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3 :=
  Flock.Guarantees.sampledKSLiveV3_of Flock.SecurityProofs.SampledKSLiveV2

/-- **The sampled audit's knowledge soundness, live, on the event that `verify --zk` accepts the files and the draw**
(`Flock.Guarantees.SampledKSLiveV4`): no hypothesis on the statement's tags, the circuit's form, shared rows, the unit
or the draw's divisibility; outside them `verify --zk` accepts with probability 0. -/
theorem Flock.SecurityProofs.SampledKSLiveV4 : Flock.Guarantees.SampledKSLiveV4 :=
  Flock.Guarantees.sampledKSLiveV4_of Flock.SecurityProofs.SampledKSLiveV3
