import Definitions.Pouw.PearlC.Accum

/-! The proof-only part of `Definitions.Pouw.PearlC.Accum`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.Accum

open Pouw.Fp8Atom Pouw.PearlC Matrix

variable {m n k : Nat}

/-- `C̃ = A′·B̃ᵀ` under the pure chain on the step `p`. -/
def ctPureChain (p : Pipeline) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => pureChain p (rowAtoms (Aq i) (Bq j))

end Pouw.PearlC.Accum
