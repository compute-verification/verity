import Definitions.Pouw.Assumptions.Dimension.Fp8Mma
import Proofs.Pouw.Dimension.EmbedProofs

/-!
# The RTX 4090's tensor-core shapes in `M4090`: proofs

* `shapeTable4090`, `shapeWF4090`: the table, by cases.
-/

namespace Pouw.Dimension.Fp8Mma

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Embed

theorem regPrice_ge (s : Shape4090) : 16 ≤ s.regPrice := by
  cases s <;> decide

/-- **The shape table.** -/
theorem shapeTable4090 : ShapeTable4090 := ⟨regPrice_ge, rfl, rfl, rfl⟩

/-- **Every shape is an `M4090` instruction.** -/
theorem shapeWF4090 : ShapeWF4090 := by
  intro N m n k s outs
  show 16 * outs.length ≤ s.regPrice * outs.length
  exact Nat.mul_le_mul_right _ (regPrice_ge s)

end Pouw.Dimension.Fp8Mma
