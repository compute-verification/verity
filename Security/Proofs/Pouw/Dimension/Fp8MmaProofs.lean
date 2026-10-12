import Models.Pouw.Assumptions.Dimension.Fp8Mma
import Proofs.Pouw.Dimension.EmbedProofs

/-!
# The RTX 4090's tensor-core shapes in `M4090`: proofs

* `shapeTable4090`, `shapeWF4090`: the table, by cases.
* `ceiling4090`: one `int` instruction per target word is a well-formed `M4090` program at `16·|W|`.
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

theorem cost4090_int {G : Type} (fs : List (G → ℚ)) : cost4090 (fs.map Op.int) = 16 * fs.length := by
  induction fs with
  | nil => rfl
  | cons f fs ih =>
    simp only [cost4090, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    rw [ih, Op.price]
    ring

/-- **No bound in `M4090` exceeds 16 per target word.** -/
theorem ceiling4090 : Ceiling4090 := by
  intro N m n k fpSem W b hb
  let P : List (Op (Noise N m n k)) := W.toList.map Op.int
  have hwf : ∀ o ∈ P, o.WF := fun o ho => by
    obtain ⟨f, -, rfl⟩ := List.mem_map.1 ho
    trivial
  have hout : ∀ w ∈ W, w ∈ (m4090 N m n k fpSem).outputs ⟨P, hwf⟩ := fun w hw =>
    mem_regs_of_int fpSem P _ w (List.mem_map.2 ⟨w, mem_toList.2 hw, rfl⟩)
  have h := hb ⟨P, hwf⟩ hout
  have hc : (m4090 N m n k fpSem).cost ⟨P, hwf⟩ = 16 * W.card := by
    show ((cost4090 P : ℕ) : ℝ) = _
    rw [cost4090_int, length_toList]
    push_cast
    ring
  rw [hc] at h
  refine ⟨h, ?_⟩
  have e : (Shape4090.e4m3F32.regPrice : ℝ) = 64 := by
    rw [shapeTable4090.2.2.1]
    norm_num
  rw [e]
  linarith

end Pouw.Dimension.Fp8Mma
