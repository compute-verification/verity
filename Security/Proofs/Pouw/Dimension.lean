import Models.Pouw.Guarantees.Dimension
import Proofs.Pouw.Dimension.OpenPipeProofs

/-!
# Proofs of the dimension argument's guarantee

One theorem `Pouw.SecurityProofs.Dimension.G : Pouw.Guarantees.Dimension.G` per guarantee, which is the theorem in
`Pouw.Dimension.OpenPipe` (`OpenPipeProofs`) that proves the same statement under its old name.
-/

namespace Pouw.SecurityProofs.Dimension

theorem A2WordsIff : Pouw.Guarantees.Dimension.A2WordsIff := Pouw.Dimension.OpenPipe.a2WordsIff

end Pouw.SecurityProofs.Dimension
