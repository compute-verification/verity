import Proofs.Pouw.Eps1.Basics
import Proofs.Pouw.Eps1.WordLocal

/-!
# Dither's CUDA-core lower bounds in M3: proofs

Each guarantee of `Models.Pouw.Guarantees.Eps1` as `Pouw.SecurityProofs.Eps1.G`. Theorem B's two instances are proved in
`Eps1/WordLocal.lean`; Theorem C is a `sorry` until its prover lands (the gap list: note:20261017T0801Z-draft-lean-stmt).
The lemmas they share are in `Eps1/Basics.lean`.
-/

namespace Pouw.SecurityProofs.Eps1

/-- Theorem B, zero instance. -/
theorem WordLocalZero : Pouw.Guarantees.Eps1.WordLocalZero := WordLocal.wordLocalZero

/-- Theorem B, general instance. -/
theorem WordLocalGeneral : Pouw.Guarantees.Eps1.WordLocalGeneral := WordLocal.wordLocalGeneral

/-- Theorem C. -/
theorem ClassPPacked : Pouw.Guarantees.Eps1.ClassPPacked := by
  sorry

end Pouw.SecurityProofs.Eps1
