module

public import Definitions.Core.Guarantees.Formalism.Witnesses

/-!
# The size of a `(c, k)`-partitioned circuit: proofs of the honest and nontrivial lemmas

The proofs of `Security/Definitions/Core/Guarantees/Formalism/Witnesses.lean`, at the instances of
`Security/Definitions/Core/Formalism/Instances.lean`, each by evaluation.
-/

@[expose] public section

namespace Verity.SecurityProofs

theorem CKSize_honest : Guarantees.CKSize_honest := ⟨⟨by decide, by decide⟩, by decide, by decide⟩

theorem CKSize_nontrivial : Guarantees.CKSize_nontrivial :=
  ⟨fun h => absurd (h.card 0) (by decide), by decide⟩

end Verity.SecurityProofs
