import Definitions.Pouw.PearlC.SkipP

/-! The proof-only part of `Definitions.Pouw.PearlC.SkipP`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.SkipP

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Skip Pouw.PearlC.Accum

variable (p : Pipeline)

variable {m n k : Nat} in
/-- **The unit's flags** under `p` and `G`. -/
def unitFlagsP (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Fin m → Fin n → ℕ → Bool :=
  fun i j t => wordFlagsP p G k (Aq i) (Bq j) t

end Pouw.PearlC.SkipP
