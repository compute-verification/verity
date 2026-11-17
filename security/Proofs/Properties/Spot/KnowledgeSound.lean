module

public import Proofs.Flock.Bridge.V5.KSLive

/-! `SpotKnowledgeSound`, the one statement the sampled composition reads of SPOT's proofs, at no narrower form than its
uses. -/

@[expose] public section

namespace Properties.Spot

def SpotKnowledgeSound : Prop := Flock.Guarantees.SampledKSLiveV5

theorem spotKnowledgeSound : SpotKnowledgeSound := Flock.SecurityProofs.SampledKSLiveV5

end Properties.Spot
