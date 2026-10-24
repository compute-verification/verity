import Definitions.Pouw.PearlC.RowSem

/-! The proof-only part of `Definitions.Pouw.PearlC.RowSem`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

variable (ops : RowOps)

/-- `pc8`'s code: the formed word's E4M3 cast. -/
def code8W (p : Pipeline) (α x : ℕ) (et f : ℕ → ℕ) : ℕ := Prim.eval ops .toE4m3 [form8 ops p α x et f]

end Pouw.PearlC.RowCircuit
