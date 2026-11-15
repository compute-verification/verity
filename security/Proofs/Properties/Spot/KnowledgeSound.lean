module

public import Proofs.Flock.Bridge.V5.KSLive

/-!
# SPOT's knowledge soundness: the one statement the composition reads of SPOT's proofs

The sampled composition (`Proofs.Properties.Verity.Sampled`) bounds SPOT's claim at each of its sessions by SPOT[Flock]'s
knowledge soundness of the live sampled audit at the verifier's public inputs, applied at the session's binders
(`Verity.Sampled.Live.v5_at`) and at no narrower form. `SpotKnowledgeSound` is that statement, as those uses quantify
it: `Flock.Guarantees.SampledKSLiveV5`, over every circuit file, public file, registration, sampling design the
verifier samples under, strategy and finder budget. `spotKnowledgeSound` is its proof,
`Flock.SecurityProofs.SampledKSLiveV5`.

It is stated here rather than in the definitions or properties tree because what it reads (the session's binding,
tables and simulator: `ZkHidden.rsZHJP`, `planZHJP`, `simZJA`) is declared only in the proofs tree.
-/

@[expose] public section

namespace Properties.Spot

/-- **SPOT's knowledge soundness**: the live sampled audit's, at the verifier's public inputs, with or without shared
rows, under the sampling design the verifier samples under from the registration (`Flock.Guarantees.SampledKSLiveV5`).
-/
def SpotKnowledgeSound : Prop := Flock.Guarantees.SampledKSLiveV5

/-- **SPOT's knowledge soundness holds** (`Flock.SecurityProofs.SampledKSLiveV5`). -/
theorem spotKnowledgeSound : SpotKnowledgeSound := Flock.SecurityProofs.SampledKSLiveV5

end Properties.Spot
