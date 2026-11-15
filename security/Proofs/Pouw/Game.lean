module

public import Definitions.Pouw.Guarantees.Game
public import Proofs.Pouw.Sampling
public import Proofs.Pouw.Theorem1
public import Proofs.Pouw.Reduction
public import Proofs.Pouw.Lemmas.Game

/-!
# Proofs of the game's guarantees

One theorem `Pouw.SecurityProofs.G : Pouw.Guarantees.G` per guarantee, which is the theorem in `Pouw.Proofs` (`Sampling`
and `Theorem1`) that proves the same statement under its old name.
-/

@[expose] public section

namespace Pouw.SecurityProofs

theorem Theorem1 : Pouw.Guarantees.Theorem1 := Pouw.Proofs.theorem1
theorem WorkWeightedSampling : Pouw.Guarantees.WorkWeightedSampling := Pouw.Proofs.workWeightedSampling
theorem EndToEnd : Pouw.Guarantees.EndToEnd := Pouw.Proofs.endToEnd

end Pouw.SecurityProofs

theorem Pouw.SecurityProofs.GapFromTT : Pouw.Guarantees.GapFromTT := @Pouw.Proofs.gapFromTT
