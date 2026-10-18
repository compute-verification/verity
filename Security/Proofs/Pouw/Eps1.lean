import Proofs.Pouw.Eps1.Basics
import Proofs.Pouw.Eps1.WordLocal
import Proofs.Pouw.Eps1.ClassP

/-!
# Dither's CUDA-core lower bounds in M3: proofs

Theorem B's two instances of `Models.Pouw.Guarantees.Eps1`, each as `Pouw.SecurityProofs.Eps1.G`, proved in
`Eps1/WordLocal.lean` from the lemmas in `Eps1/Basics.lean`, and Theorem C's count over issue slots (`ClassPSlots`),
proved in `Eps1/ClassP.lean`. Theorem C itself (`ClassPPacked`) is not proved, so `lean-audit.json` doesn't list it:
its paper proof caps a writer's creations at 4, and a writer can create 8 (note:20261018T0230Z-draft-floors).
-/

namespace Pouw.SecurityProofs.Eps1

/-- Theorem B, zero instance. -/
theorem WordLocalZero : Pouw.Guarantees.Eps1.WordLocalZero := WordLocal.wordLocalZero

/-- Theorem B, general instance. -/
theorem WordLocalGeneral : Pouw.Guarantees.Eps1.WordLocalGeneral := WordLocal.wordLocalGeneral

/-- Theorem C's count over issue slots. -/
theorem ClassPSlots : Pouw.Guarantees.Eps1.ClassPSlots := ClassP.classPSlots

end Pouw.SecurityProofs.Eps1
