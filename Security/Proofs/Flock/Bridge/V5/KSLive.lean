module

public import Proofs.Flock.Bridge.V5.Assembly
public import Proofs.Flock.Bridge.V5.EndToEndHidden
public import Proofs.Flock.Bridge.V5.OpeningsCanon
public import Proofs.Flock.Bridge.V5.Transports
public import Proofs.Flock.Bridge.KSLive

/-!
# `SampledKSLiveV5`, proved

`Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5`: `AssemblyTargetV5` (`assemblyTargetV5`) with
canonical transport at public inputs (`transportsLiveV5`), M1's core (`faultyRightTargetC`) and `EndToEndHiddenV5`
(`endToEndHiddenV5_of`, from `openingsCanonV5`) plugged in.
-/

@[expose] public section

/-- **The sampled audit's knowledge soundness, live, at the verifier's public inputs, with or without shared rows**
(`Flock.Guarantees.SampledKSLiveV5`). -/
theorem Flock.SecurityProofs.SampledKSLiveV5 : Flock.Guarantees.SampledKSLiveV5 :=
  Flock.Bridge.assemblyTargetV5 Flock.Bridge.transportsLiveV5 Flock.Bridge.faultyRightTargetC
    (Flock.Bridge.endToEndHiddenV5_of Flock.Bridge.openingsCanonV5)
