import Definitions.Pouw.PearlC.TileCircuit

/-! The proof-only part of `Definitions.Pouw.PearlC.TileCircuit`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.TileCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact

namespace Spec

variable (S : Spec)

/-- One proof unit holding every computed gate. -/
def part : Verity.Partition S.toCircuit 1 where
  unit g := if (S.gate g).input then none else some 0
  input_iff g := by
    show (if (S.gate g).input then none else some 0) = none ↔ (S.gate g).input = true
    split <;> simp_all

end Spec

end Pouw.PearlC.TileCircuit
