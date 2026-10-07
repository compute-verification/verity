import Models.Pouw.Guarantees.Game
import Proofs.Pouw.Sampling
import Proofs.Pouw.Theorem1
import Proofs.Pouw.Reduction

/-!
# Proofs of the game's guarantees

One theorem `Pouw.SecurityProofs.G : Pouw.Guarantees.G` per guarantee, which is the theorem in `Pouw.Proofs` (`Sampling`
and `Theorem1`) that proves the same statement under its old name.
-/

namespace Pouw.SecurityProofs

theorem Theorem1 : Pouw.Guarantees.Theorem1 := Pouw.Proofs.theorem1
theorem WorkWeightedSampling : Pouw.Guarantees.WorkWeightedSampling := Pouw.Proofs.workWeightedSampling
theorem EndToEnd : Pouw.Guarantees.EndToEnd := Pouw.Proofs.endToEnd

end Pouw.SecurityProofs

theorem Pouw.SecurityProofs.GammaFromTT : Pouw.Guarantees.GammaFromTT := @Pouw.Proofs.gammaFromTT
