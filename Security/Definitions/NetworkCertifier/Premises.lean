/-!
# The network certifier's premises

What a property of the network certifier may take besides its acceptance (the lean-proofs skill, "One shape"): one
physical premise, `DeviceIntact`. It is the listed `trusted-nebius-host` (`Verity.Assumptions.TrustedNebiusHost`) read
for the certifier device: the device runs its code as built. Nothing is assumed of SHA-512 or LMS: a property that relies
on the device's hashes or signatures concludes its claim or a break the run itself carries.
-/

namespace NetTiming

/-- **The certifier device runs as built**: on every input `x` (its configuration, its key and the trace the world gives
it) the physical `device` returns what its `code` returns. The host changes none of it, and no attestation stands
behind it: it is trust in the auditor's device and its host provider, never in a host the prover controls
(`trusted-nebius-host`). It doesn't say the device's key stays secret. It states only
that claim's part about one run. Its other part, that its host never restores an earlier copy of the state the device
keeps there, is about its runs together, and no property takes it as a premise (`Properties.NetworkCertifier`, "Other
runs of the key"). -/
def DeviceIntact {In Out : Type} (device code : In → Out) : Prop := ∀ x, device x = code x

end NetTiming
