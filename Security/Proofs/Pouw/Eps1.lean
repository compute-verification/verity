import Proofs.Pouw.Eps1.Basics
import Proofs.Pouw.Eps1.WordLocal

/-!
# Dither's CUDA-core lower bounds in M3: proofs

Theorem B's two instances of `Models.Pouw.Guarantees.Eps1`, each as `Pouw.SecurityProofs.Eps1.G`, proved in
`Eps1/WordLocal.lean` from the lemmas in `Eps1/Basics.lean`. Theorem C (`ClassPPacked`) is not proved yet, so
`lean-audit.json` doesn't list it (the gap list: note:20261017T0801Z-draft-lean-stmt).
-/

namespace Pouw.SecurityProofs.Eps1

/-- Theorem B, zero instance. -/
theorem WordLocalZero : Pouw.Guarantees.Eps1.WordLocalZero := WordLocal.wordLocalZero

/-- Theorem B, general instance. -/
theorem WordLocalGeneral : Pouw.Guarantees.Eps1.WordLocalGeneral := WordLocal.wordLocalGeneral

end Pouw.SecurityProofs.Eps1
