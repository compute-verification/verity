import Definitions.Accounting.Actual

/-! The proof-only part of `Definitions.Accounting.Actual`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting

open Verity Finset

namespace Placed

variable {L : Type*} (A : Placed L)

open Classical in
/-- **Egress** `out_X(I)` (v2.7, at the reader's tick): the width of the gates placed in `X` that some gate placed
outside `X` during `I` reads, `in(A[X̄, I]) ∩ A[X]` (an input gate reads nothing). -/
noncomputable def egress (X : Finset L) (I : Set ℕ) : ℕ :=
  A.width (univ.filter fun g => A.x g ∈ X ∧ ∃ h, A.x h ∉ X ∧ A.t h ∈ I ∧ g ∈ A.reads h)

/-- **Wires run forward in time inside `X`**: a computed gate placed in `X` is read by gates placed in `X` only at its
tick or later. -/
def Forward (X : Finset L) : Prop :=
  ∀ g, ∀ a ∈ A.reads g, A.isInput a = false → A.x a ∈ X → A.x g ∈ X → A.t a ≤ A.t g

end Placed

end Accounting
