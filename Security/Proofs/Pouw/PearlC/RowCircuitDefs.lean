import Definitions.Pouw.PearlC.RowCircuit

/-! The proof-only part of `Definitions.Pouw.PearlC.RowCircuit`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

/-- An operand's value, the gates' values being `V`. -/
def Arg.eval (ops : RowOps) (V : ℕ → ℕ) : Arg → ℕ
  | .g a w => w.eval ops (V a)
  | .c v => v

end Pouw.PearlC.RowCircuit
