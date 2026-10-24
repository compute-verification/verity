import Definitions.NetworkCertifier.Grid

/-! The proof-only part of `Definitions.NetworkCertifier.Grid`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace NetTiming

open Finset

/-- Occupancy: which slots hold a frame. -/
def occ {T S : ℕ} {F : Type} (g : Grid T S F) : Fin T → Fin S → Bool := fun t s => (g t s).isSome

end NetTiming
