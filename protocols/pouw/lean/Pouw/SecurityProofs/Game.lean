import Pouw.Guarantees.Game
import Pouw.SecurityProofs.Sampling
import Pouw.SecurityProofs.Theorem1

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
