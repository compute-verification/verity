import Definitions.Accounting.Confidence

/-! The proof-only part of `Definitions.Accounting.Confidence`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting

open Finset

namespace Actual.Run

variable {L : Type*} (R : Actual.Run L)

/-- **Memory** (v2.3), per run: each `X ∈ 𝒳` has a state of at most `M X` bits at every tick. -/
def Memory (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop := ∀ X ∈ 𝒳, ∀ t, R.mem X t ≤ M X

end Actual.Run

end Accounting
