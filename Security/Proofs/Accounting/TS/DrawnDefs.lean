import Definitions.Accounting.TS.Drawn

/-! The proof-only part of `Definitions.Accounting.TS.Drawn`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.TS

open Pouw.RelativeSecurity

namespace DrawnLayer

variable {C B : Type} {S m s : ℕ} {J Q Pos V : Type} (pf : DrawnLayer C B S m s J Q Pos V)

/-- **Binding fails at position `ℓ`**: an accepted proof of a drawn unit opens `ℓ` to a value other than its committed
one. Two that open it differently on `x` move it. -/
def Moved [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (ℓ : Pos) : Prop :=
  ∃ v, pf.Opens x ℓ v ∧ v ≠ pf.committed x ℓ

end DrawnLayer

end Accounting.TS
