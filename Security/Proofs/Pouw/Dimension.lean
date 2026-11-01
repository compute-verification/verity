module

public import Definitions.Pouw.Guarantees.Dimension
public import Proofs.Pouw.Dimension.OpenPipeProofs

/-!
# Proofs of the dimension argument's guarantee

One theorem `Pouw.SecurityProofs.Dimension.G : Pouw.Guarantees.Dimension.G` per guarantee, which is the theorem in
`Pouw.Dimension.OpenPipe` (`OpenPipeProofs`) that proves the same statement under its old name.
-/

@[expose] public section

namespace Pouw.SecurityProofs.Dimension

theorem A2WordsIff : Pouw.Guarantees.Dimension.A2WordsIff := Pouw.Dimension.OpenPipe.a2WordsIff

end Pouw.SecurityProofs.Dimension
