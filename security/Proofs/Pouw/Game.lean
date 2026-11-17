module

public import Definitions.Pouw.Guarantees.Game
public import Proofs.Pouw.Sampling
public import Proofs.Pouw.Theorem1
public import Proofs.Pouw.Reduction
public import Proofs.Pouw.Lemmas.Game

/-! One theorem `Pouw.SecurityProofs.G : Pouw.Guarantees.G` per guarantee of the game. -/

@[expose] public section

namespace Pouw.SecurityProofs

theorem Theorem1 : Pouw.Guarantees.Theorem1 := Pouw.Proofs.theorem1
theorem WorkWeightedSampling : Pouw.Guarantees.WorkWeightedSampling := Pouw.Proofs.workWeightedSampling
theorem EndToEnd : Pouw.Guarantees.EndToEnd := Pouw.Proofs.endToEnd

end Pouw.SecurityProofs

theorem Pouw.SecurityProofs.GapFromTT : Pouw.Guarantees.GapFromTT := @Pouw.Proofs.gapFromTT
