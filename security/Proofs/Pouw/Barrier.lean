module

public import Proofs.Pouw.Barrier.Proofs
public import Proofs.Pouw.Lemmas.Barrier

/-! One theorem `Pouw.SecurityProofs.Barrier.G : Pouw.Guarantees.Barrier.G` per degenerate-input barrier
guarantee. -/

@[expose] public section

namespace Pouw.SecurityProofs.Barrier

theorem Generic : Pouw.Guarantees.Barrier.Generic := Pouw.Barrier.Proofs.generic

end Pouw.SecurityProofs.Barrier
