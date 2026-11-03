import Definitions.Nci.Products.Instances
import Definitions.Nci.Products.System

/-!
# Products made whole through chains: the system their witnesses are stated at (trusted)

`Pair.system θ`, `Pair`'s product (`Security/Definitions/Nci/Products/Instances.lean`) alone as a whole-making system:
one class, of chain length 2, the product at depth 0 and no value made from an output entry, so a block has each value
by paying its width or by reading it free. `freshSystem` has the operands fresh, as `Pair.fresh`, and
`registeredSystem` registered, as `Pair.registered`.
-/

namespace Nci.Products.Pair

/-- `Pair`'s product with `θ` registered, alone in one class of chain length 2, at depth 0, with no value made from an
output entry. -/
abbrev system (θ : Finset (Fin C.N)) (hθ : ∀ g ∈ θ, C.isInput g = true) : System ⟨C, θ, hθ⟩ Unit Unit where
  K _ := 2
  K_pos _ := Nat.zero_lt_two
  cls _ := ()
  ni _ := 1
  nj _ := 1
  prod _ := product θ hθ
  depth _ := 0
  src _ := ∅

/-- The system with the operands fresh. -/
abbrev freshSystem : System fresh Unit Unit := system ∅ (by decide)

/-- The system with the operands registered. -/
abbrev registeredSystem : System registered Unit Unit := system {0, 1, 2, 3} (by decide)

end Nci.Products.Pair
