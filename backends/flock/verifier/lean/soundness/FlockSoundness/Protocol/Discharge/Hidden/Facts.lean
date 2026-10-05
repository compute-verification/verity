import FlockSoundness.Discharge.Hidden.Spec

/-!
# The facts read off `Spec`

What [wave2-hidden-exec] reads of `Stmt.setupHidden`, from `setupHidden_spec`: its tags, its leaf scheme
(`setupHidden_hm96`), its well-formedness (`setupHidden_wf`: `StmtWF`, `RegionsWF`) and the drawn public file; and of
`HmOut.parse`, that an untyped one is `HmRow.parse`'s circuit then `HmOut.checkRows` (`parse_untyped`).
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

/-! ## The parse -/

/-- **`parse_untyped`**: an untyped `HmOut.parse` is `HmRow.parse`'s circuit, then `HmOut.checkRows`. -/
theorem parse_untyped {c : Circuit} (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) :
    HmRow.parse tags cf tables = .ok c ∧ HmOut.checkRows c = .ok () := by
  unfold HmOut.parse at h
  simp only [ht, Bool.not_false, ↓reduceIte] at h
  obtain ⟨c₀, hc₀, h⟩ := ebind_ok h
  obtain ⟨u, hu, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact ⟨hc₀, hu⟩

/-! ## The regions -/

/-! ## The facts read off `Spec` -/

end FlockSoundness.Discharge.Hidden
