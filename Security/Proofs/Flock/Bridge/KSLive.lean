import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.FaultyRightC
import Proofs.Flock.Bridge.EndToEndHiddenC
import Proofs.Flock.Bridge.OpeningsCanon
import Proofs.Flock.Bridge.Refused
import Proofs.Flock.Bridge.Transports

/-!
# `SampledKSLiveV2` and `SampledKSLiveV3`, proved

`Flock.Bridge.sampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2`, a lemma for V3: `AssemblyTargetC` (`assemblyTargetC`)
with canonical transport (`transportsLiveC`), M1's core (`faultyRightTargetC`), the slot count (`partBlocks`), the count
lemma (`count`) and `EndToEndHiddenC` (`endToEndHiddenC_of`, from `openingsCanon`) plugged in.
`Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3`, the guarantee at the no-shared-rows form:
V2 with its file preconditions read off the verifiers' acceptance (`sampledKSLiveV3_of`).
-/

/-- **The sampled audit's knowledge soundness, live, at the circuit files the verifier of record reads**
(`Flock.Guarantees.SampledKSLiveV2`), V3's lemma. -/
theorem Flock.Bridge.sampledKSLiveV2 : Flock.Guarantees.SampledKSLiveV2 :=
  Flock.Bridge.assemblyTargetC Flock.Bridge.transportsLiveC Flock.Bridge.faultyRightTargetC Flock.Bridge.partBlocks
    Flock.Bridge.count (Flock.Bridge.endToEndHiddenC_of Flock.Bridge.openingsCanon)

/-- **The sampled audit's knowledge soundness, live, on the event that the verifiers accept the files**
(`Flock.Guarantees.SampledKSLiveV3`): no hypothesis on the files' rows, names, roots or population order. -/
theorem Flock.SecurityProofs.SampledKSLiveV3 : Flock.Guarantees.SampledKSLiveV3 :=
  Flock.Guarantees.sampledKSLiveV3_of Flock.Bridge.sampledKSLiveV2
