import Definitions.Accounting.Requirement

/-! The proof-only part of `Definitions.Accounting.Requirement`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Requirement

variable {G : Type} [DecidableEq G]

section Blocks

variable (W : Wiring G) (e : Execution G) (ℓ : Layout G)

/-- The bits `W` reads from block `j` over the engine steps of window `w`. -/
def readOf (w : Window) (j : ℕ) : ℕ := ∑ s ∈ w.steps, readIn W e ℓ s j

end Blocks

end Accounting.Requirement
