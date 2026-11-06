module

public import Definitions.Pouw.Guarantees.Barrier
public import Proofs.Pouw.Barrier.Proofs
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset

/-!
# Proofs of the degenerate-input barrier's guarantee

One theorem `Pouw.SecurityProofs.Barrier.G : Pouw.Guarantees.Barrier.G` per guarantee, which is the theorem in
`Pouw.Barrier.Proofs` that proves the same statement under its old name.
-/

@[expose] public section

namespace Pouw.SecurityProofs.Barrier

theorem Generic : Pouw.Guarantees.Barrier.Generic := Pouw.Barrier.Proofs.generic

end Pouw.SecurityProofs.Barrier
