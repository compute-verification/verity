import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.OpeningsCanon
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.Rows
import Proofs.Flock.Bridge.Transports

/-!
# `SampledKSLiveV2` and `SampledKSLiveV3`, proved

`Flock.Bridge.sampledKSLiveXr : Flock.Bridge.SampledKSLiveXr`: `AssemblyTargetC` (`assemblyTargetC`) with canonical
transport (`transportsLiveC`), M1's core (`faultyRightTargetC`), the slot count (`partRus`), the count lemma (`count`)
and `EndToEndHiddenC` (`endToEndHiddenC_of`, from `openingsCanon`) plugged in.
`Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`: the claims at the registrant's rows, from
`sampledKSLiveXr` and `EndToEndRegisteredRows`' rows link (`sampledKSLiveV2_of_xr`).
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`: V2 with its file preconditions read off the
verifiers' acceptance (`sampledKSLiveV3_of`).
-/

/-- **The sampled audit's knowledge soundness, live, at the extracted values** (`Flock.Bridge.SampledKSLiveXr`). -/
theorem Flock.Bridge.sampledKSLiveXr : Flock.Bridge.SampledKSLiveXr :=
  Flock.Bridge.assemblyTargetC Flock.Bridge.transportsLiveC Flock.Bridge.faultyRightTargetC Flock.Bridge.partRus
    Flock.Bridge.count (Flock.Bridge.endToEndHiddenC_of Flock.Bridge.openingsCanon)

/-- **The sampled audit's knowledge soundness, live, at the registrant's rows** (`Flock.Guarantees.SampledKSLiveV2`). -/
theorem Flock.SecurityProofs.SampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2 :=
  Flock.Bridge.sampledKSLiveV2_of_xr Flock.Bridge.sampledKSLiveXr Flock.Bridge.transportsLiveC Flock.Bridge.count
    Flock.Bridge.openingsCanon

/-- **The sampled audit's knowledge soundness, live, on the event that the verifiers accept the files**
(`Flock.Guarantees.SampledKSLiveV3`): no hypothesis on the files' rows, names, roots or population order. -/
theorem Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3 :=
  Flock.Guarantees.sampledKSLiveV3_of Flock.SecurityProofs.SampledKSLiveV2
