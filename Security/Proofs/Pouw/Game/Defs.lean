import Definitions.Pouw.Game.Defs

/-! The proof-only part of `Definitions.Pouw.Game.Defs`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw

open Finset

/-- The first milestone's shape `(m, k, n) = (2^12, 2^16, 2^16)` (§6, notes on D1). -/
def milestoneShape : Shape := ⟨2 ^ 12, 2 ^ 16, 2 ^ 16⟩

variable {Q R S : Type}

end Pouw
